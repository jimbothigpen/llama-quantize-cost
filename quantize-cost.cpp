// quantize-cost: per-(tensor, format) MSE measurement tool.
//
// Reads a BF16 GGUF, and for each (tensor, candidate-type) pair runs
// quantize -> dequantize via the same ggml kernels llama-quantize uses,
// then writes the round-trip MSE and the quantized size to a CSV.
//
// Output schema:
//   tensor_name,n_elements,src_type,fmt,size_bytes,mse,bpw
//
// Generally useful for any mixed-precision allocator, quant-aware
// pruning, sensitivity analysis, or tensor-level format-comparison
// work. One concrete consumer: the prismaquant-style allocator
// (https://github.com/RobTand/prismaquant + the GGUF adapter at
// https://github.com/jimbothigpen/prismaquant-llama) which combines
// this CSV with a Hessian probe (Fisher H_trace per Linear) to solve a
// multi-choice knapsack picking one fmt per tensor that minimizes
// sum(0.5 * H_trace[t] * MSE[t, fmt[t]]) under a size budget.
//
// Usage:
//   quantize-cost --model SRC.gguf --types Q4_K_M,Q5_K_M,IQ4_KS,...
//                 --output costs.csv [--imatrix IMATRIX.dat]
//                 [--include-regex REGEX] [--exclude-regex REGEX]

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

struct type_entry {
    ggml_type    t;
    const char * name;
};

static bool ieq(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

// Resolve a type name against the actual ggml type catalog of the build we
// were compiled into. Iterates 0..GGML_TYPE_COUNT and matches against the
// canonical name returned by ggml_type_name(). This avoids any hardcoded
// list that could fall out of sync with whichever fork of ggml the tool
// was built against (frankenturbo2 has TURBO/TCQ/PLANAR/INNERQ types that
// don't exist in upstream; ik_llama has IQ?_K etc.).
//
// Recipe-only names from llama-quantize (e.g. IQ3_XS, IQ3_M, MXFP4_MOE,
// Q4_K_M alias, Q*_K_S aliases) are not direct ggml types; they won't
// resolve here. The caller is expected to either skip them or translate
// them to a representative ggml_type before invoking this tool.
static bool resolve_type(const std::string & name, ggml_type & out) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type t = (ggml_type) i;
        const char * nm = ggml_type_name(t);
        if (nm && ieq(name, nm)) { out = t; return true; }
    }
    return false;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Imatrix reader — kept in lockstep with tools/quantize/quantize.cpp's
// load_imatrix() / load_legacy_imatrix(). Two on-disk formats exist:
//   1. legacy binary (n_entries + per-tensor {name, ncall, nval, floats})
//   2. new GGUF format with `<tensor>.in_sum2` and `<tensor>.counts` pairs
// Both produce the same in-memory shape: tensor_name -> per-column importance
// vector, ready to pass to ggml_quantize_chunk's imatrix argument.

static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

static bool string_remove_suffix(std::string & s, const std::string & suffix) {
    if (s.size() < suffix.size()) return false;
    if (s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    s.resize(s.size() - suffix.size());
    return true;
}

static int load_legacy_imatrix(const std::string & imatrix_file,
                               std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file, std::ios::binary);
    if (!in) {
        fprintf(stderr, "[quantize-cost] failed to open imatrix: %s\n", imatrix_file.c_str());
        return -1;
    }
    int n_entries; in.read((char *)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        fprintf(stderr, "[quantize-cost] no entries in imatrix %s\n", imatrix_file.c_str());
        return -1;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_buf(len + 1);
        in.read(name_buf.data(), len);
        name_buf[len] = 0;
        const std::string name(name_buf.data());
        auto & e = imatrix_data[name];
        int ncall; in.read((char *)&ncall, sizeof(ncall));
        int nval;  in.read((char *)&nval,  sizeof(nval));
        if (in.fail() || nval < 1) {
            fprintf(stderr, "[quantize-cost] failed reading entry %d\n", i);
            return -1;
        }
        e.resize(nval);
        in.read((char *)e.data(), nval * sizeof(float));
        if (in.fail()) {
            fprintf(stderr, "[quantize-cost] failed reading floats for entry %d\n", i);
            return -1;
        }
        if (ncall > 0) for (auto & v : e) v /= (float)ncall;
    }
    return n_entries;
}

static int load_imatrix(const std::string & imatrix_file,
                        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    struct ggml_context * ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &ctx };
    gguf_context * gctx = gguf_init_from_file(imatrix_file.c_str(), gp);
    if (!gctx) {
        // Fall through to legacy format
        if (ctx) ggml_free(ctx);
        return load_legacy_imatrix(imatrix_file, imatrix_data);
    }
    const int chunk_count_idx = gguf_find_key(gctx, LLM_KV_IMATRIX_CHUNK_COUNT);
    if (chunk_count_idx < 0) {
        fprintf(stderr, "[quantize-cost] missing imatrix metadata in %s\n", imatrix_file.c_str());
        gguf_free(gctx); ggml_free(ctx);
        return -1;
    }
    const std::string sums_suffix   = ".in_sum2";
    const std::string counts_suffix = ".counts";
    std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> sums_counts_for;
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string nm = cur->name;
        if (nm.empty()) continue;
        if (string_remove_suffix(nm, sums_suffix)) {
            sums_counts_for[std::move(nm)].first = cur;
        } else if (string_remove_suffix(nm, counts_suffix)) {
            sums_counts_for[std::move(nm)].second = cur;
        }
    }
    for (const auto & sc : sums_counts_for) {
        const std::string & name = sc.first;
        const ggml_tensor * sums   = sc.second.first;
        const ggml_tensor * counts = sc.second.second;
        if (!sums || !counts) continue;  // mismatched — skip
        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];
        auto & e = imatrix_data[name];
        e.resize(ggml_nelements(sums));
        for (int64_t j = 0; j < ne1; ++j) {
            const float c = ((const float *)counts->data)[j];
            if (c > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j*ne0 + i] = ((const float *)sums->data)[j*ne0 + i] / c;
                }
            } else {
                // Tensor never saw input during calibration; uniform fallback.
                for (int64_t i = 0; i < ne0; ++i) e[j*ne0 + i] = 1.0f;
            }
        }
    }
    int last_chunk = (int)gguf_get_val_u32(gctx, chunk_count_idx);
    gguf_free(gctx); ggml_free(ctx);
    return last_chunk;
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s --model SRC.gguf --types T1,T2,... --output costs.csv\n"
        "       [--imatrix IMATRIX.dat] [--include-regex REGEX] [--exclude-regex REGEX]\n"
        "       [--fisher-sidecar DIR]\n"
        "\n"
        "Writes one row per (tensor, type) measuring round-trip MSE and quantized size.\n"
        "With --fisher-sidecar, additionally computes Fisher row-weighted output MSE\n"
        "from per-Linear .bin sidecars produced by prismaquant-llama's emitter.\n",
        argv0);
}

// Read raw tensor bytes from `fp` at the given file offset.  Caller owns dst.
static bool read_tensor_bytes(FILE * fp, size_t off, size_t nbytes, void * dst) {
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) return false;
    return fread(dst, 1, nbytes, fp) == nbytes;
}

// Fisher sidecar reader. Each sidecar file is a per-Linear binary blob
// produced by prismaquant-llama's scripts/emit_fisher_sidecar.py, layout:
//
//   offset  size                       field
//   0       4 bytes                    magic = "PQFS"
//   4       4 bytes (uint32 LE)        version = 1
//   8       4 bytes (uint32 LE)        N (sampled activation rows)
//   12      4 bytes (uint32 LE)        in_features
//   16      N * in_features * 4 bytes  X (float32, row-major [N, in_features])
//   ...     N * 4 bytes                fisher_weights (float32 per row)
//
// File naming: GGUF tensor name with `.` -> `__`, suffixed `.bin`.
// Example: `blk.0.attn_q.weight` -> `<dir>/blk__0__attn_q__weight.bin`.
//
// Loaded lazily; absent sidecars produce NaN fisher_output_mse rather than
// aborting (keeps the binary useful when no fisher pass was requested).

struct fisher_sidecar {
    uint32_t n_rows         = 0;
    uint32_t in_features    = 0;
    std::vector<float> X;
    std::vector<float> fisher_weights;
};

static std::string sidecar_safe_name(const std::string & gguf_tensor_name) {
    std::string out;
    out.reserve(gguf_tensor_name.size() + 8);
    for (char c : gguf_tensor_name) {
        if (c == '.') { out += "__"; } else { out += c; }
    }
    return out;
}

static bool load_fisher_sidecar(const std::string & dir,
                                const std::string & tensor_name,
                                fisher_sidecar & sc,
                                std::string & err) {
    const std::string path = dir + "/" + sidecar_safe_name(tensor_name) + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "open failed: " + path;
        return false;
    }
    char magic[4];
    uint32_t version = 0;
    uint32_t n_rows = 0, in_features = 0;
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PQFS", 4) != 0) {
        err = "bad magic: " + path; fclose(f); return false;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != 1u) {
        err = "bad version: " + path; fclose(f); return false;
    }
    if (fread(&n_rows, sizeof(n_rows), 1, f) != 1 ||
        fread(&in_features, sizeof(in_features), 1, f) != 1) {
        err = "header truncated: " + path; fclose(f); return false;
    }
    if (n_rows == 0 || in_features == 0) {
        err = "zero rows/features: " + path; fclose(f); return false;
    }
    sc.n_rows      = n_rows;
    sc.in_features = in_features;
    sc.X.assign((size_t)n_rows * (size_t)in_features, 0.0f);
    sc.fisher_weights.assign((size_t)n_rows, 0.0f);
    if (fread(sc.X.data(), sizeof(float), sc.X.size(), f) != sc.X.size()) {
        err = "X truncated: " + path; fclose(f); return false;
    }
    if (fread(sc.fisher_weights.data(), sizeof(float),
              sc.fisher_weights.size(), f) != sc.fisher_weights.size()) {
        err = "fisher_weights truncated: " + path; fclose(f); return false;
    }
    fclose(f);
    return true;
}

// Compute Fisher row-weighted output MSE for a single (tensor, format).
//
// Inputs:
//   src_f32    : original weights, shape [n_out, n_in] row-major
//   deq_f32    : quant-then-dequant weights, same shape
//   X          : activation samples, shape [N, n_in] row-major
//   fisher     : per-row Fisher weights, length N
//   n_out, n_in, N : dimensions
//
// Math (mirrors prismaquant/measure_quant_cost.py:404-410):
//   y_err[i,j] = sum_k X[i,k] * (src_f32[j,k] - deq_f32[j,k])
//   fisher_output_mse = sum_{i,j} fisher[i] * y_err[i,j]^2 / (N * n_out)
//
// Cost: O(N * n_out * n_in) per call. Caller is expected to invoke this
// only for direct-measure tensors (~20 per pipeline), not after-propagation.
static double compute_fisher_output_mse(
        const float * src_f32, const float * deq_f32,
        const float * X, const float * fisher,
        int64_t n_out, int64_t n_in, int64_t N) {
    // Output error per (row i, output feature j): y_err[i, j].
    // Computed inline to avoid materializing the full [N, n_out] matrix.
    double weighted_sse = 0.0;
    for (int64_t j = 0; j < n_out; ++j) {
        // err_w[j, :] = src_f32[j, :] - deq_f32[j, :]
        const float * sj = src_f32 + j * n_in;
        const float * dj = deq_f32 + j * n_in;
        for (int64_t i = 0; i < N; ++i) {
            const float * xi = X + i * n_in;
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) {
                acc += (double)xi[k] * (double)(sj[k] - dj[k]);
            }
            weighted_sse += (double)fisher[i] * acc * acc;
        }
    }
    return weighted_sse / ((double)N * (double)n_out);
}

int main(int argc, char ** argv) {
    std::string fname_in;
    std::string types_csv;
    std::string fname_out;
    std::string fname_imatrix;  // optional
    std::string include_re;
    std::string exclude_re;
    std::string fisher_sidecar_dir;  // optional; activates fisher_output_mse column

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char * tag) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", tag); exit(1); }
            return std::string(argv[++i]);
        };
        if      (a == "--model")            fname_in     = need("--model");
        else if (a == "--types")            types_csv    = need("--types");
        else if (a == "--output")           fname_out    = need("--output");
        else if (a == "--imatrix")          fname_imatrix = need("--imatrix");
        else if (a == "--include-regex")    include_re   = need("--include-regex");
        else if (a == "--exclude-regex")    exclude_re   = need("--exclude-regex");
        else if (a == "--fisher-sidecar")   fisher_sidecar_dir = need("--fisher-sidecar");
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (fname_in.empty() || types_csv.empty() || fname_out.empty()) {
        usage(argv[0]); return 1;
    }

    // Resolve types
    std::vector<type_entry> targets;
    for (const auto & nm : split_csv(types_csv)) {
        ggml_type t;
        if (!resolve_type(nm, t)) {
            // Skip-with-warning instead of fatal: prismaquant-style callers pass the full
            // calibration sweep including recipe-only names. Producing partial output is
            // strictly more useful than failing the whole run.
            fprintf(stderr, "[quantize-cost] WARN: skipping unknown type: %s\n", nm.c_str());
            continue;
        }
        // Use ggml's canonical name so the CSV is consistent regardless of
        // how the user spelled the requested type.
        const char * canon = ggml_type_name(t);
        if (!canon) canon = nm.c_str();
        targets.push_back({t, canon});
    }
    if (targets.empty()) {
        fprintf(stderr, "[quantize-cost] no known types in --types list; nothing to do\n");
        return 1;
    }

    std::regex include_rx, exclude_rx;
    bool has_include = !include_re.empty();
    bool has_exclude = !exclude_re.empty();
    if (has_include) include_rx = std::regex(include_re);
    if (has_exclude) exclude_rx = std::regex(exclude_re);

    // Read imatrix (legacy or GGUF format) if provided. The same file format
    // llama-quantize consumes; load_imatrix auto-falls back to the legacy
    // binary reader on a non-GGUF input.
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    if (!fname_imatrix.empty()) {
        const int rc = load_imatrix(fname_imatrix, imatrix_data);
        if (rc < 0) { return 1; }
        fprintf(stderr, "[quantize-cost] imatrix: loaded %zu entries from %s\n",
                imatrix_data.size(), fname_imatrix.c_str());
    }

    // Load gguf metadata + ggml tensor structs (no data allocation).
    struct ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &meta_ctx };
    gguf_context * gctx = gguf_init_from_file(fname_in.c_str(), gp);
    if (!gctx) { fprintf(stderr, "failed to open: %s\n", fname_in.c_str()); return 1; }

    FILE * fp = fopen(fname_in.c_str(), "rb");
    if (!fp) { fprintf(stderr, "failed to fopen: %s\n", fname_in.c_str()); return 1; }
    const size_t data_off = gguf_get_data_offset(gctx);

    FILE * out = fopen(fname_out.c_str(), "w");
    if (!out) { fprintf(stderr, "failed to open output: %s\n", fname_out.c_str()); return 1; }
    // Schema kept stable across modes: fisher_output_mse is always present;
    // it's `nan` when no sidecar dir was provided or no sidecar exists for
    // this tensor. Downstream parsers should treat the column as optional
    // and tolerate `nan` (and the prismaquant-llama allocator does).
    fprintf(out, "tensor_name,n_elements,src_type,fmt,size_bytes,mse,bpw,fisher_output_mse\n");
    fflush(out);

    const bool fisher_enabled = !fisher_sidecar_dir.empty();
    if (fisher_enabled) {
        fprintf(stderr, "[quantize-cost] fisher sidecar dir: %s\n",
                fisher_sidecar_dir.c_str());
    }

    // Iterate tensors via the ggml_context that gguf populated.
    int64_t n_total = 0, n_skip = 0, n_done = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(meta_ctx); t; t = ggml_get_next_tensor(meta_ctx, t)) {
        n_total++;
        const std::string name = t->name;

        if (has_include && !std::regex_search(name, include_rx)) { n_skip++; continue; }
        if (has_exclude &&  std::regex_search(name, exclude_rx)) { n_skip++; continue; }

        const ggml_type src_type = t->type;
        const int64_t n_elements = ggml_nelements(t);
        const int64_t n_per_row  = t->ne[0];
        const int64_t n_rows     = n_elements / n_per_row;
        const auto * src_traits  = ggml_get_type_traits(src_type);

        // Sanity: only quantize 2D linear weights with row-major contiguous layout.
        // Embedding / lm_head / norms etc. fall through this with caveats; we still
        // measure them but the user can filter at allocator time.
        if (!src_traits || !src_traits->to_float) {
            n_skip++;
            continue;
        }

        // Look up tensor offset by name (the meta_ctx tensor has no .data because
        // no_alloc=true, but gguf_find_tensor + gguf_get_tensor_offset gives us the
        // absolute position in the file).
        const int64_t tid = gguf_find_tensor(gctx, name.c_str());
        if (tid < 0) { n_skip++; continue; }
        const size_t off   = data_off + gguf_get_tensor_offset(gctx, tid);
        const size_t nbyte = gguf_get_tensor_size(gctx, tid);

        // Stream this tensor: raw -> f32 -> per-format quantize/dequantize/MSE.
        std::vector<uint8_t> raw(nbyte);
        if (!read_tensor_bytes(fp, off, nbyte, raw.data())) {
            fprintf(stderr, "[quantize-cost] read failed: %s\n", name.c_str());
            n_skip++;
            continue;
        }

        std::vector<float> src_f32(n_elements);
        src_traits->to_float(raw.data(), src_f32.data(), n_elements);

        // Load fisher sidecar for this tensor, if requested. A missing sidecar
        // is not fatal — most tensors won't have one (we only emit them for
        // direct-measure exemplars), and the format-loop will emit `nan` in
        // the fisher column for any tensor without a loaded sidecar.
        fisher_sidecar sc;
        bool have_sidecar = false;
        if (fisher_enabled) {
            std::string serr;
            if (load_fisher_sidecar(fisher_sidecar_dir, name, sc, serr)) {
                if ((int64_t)sc.in_features != n_per_row) {
                    fprintf(stderr, "[quantize-cost] WARN: sidecar in_features=%u "
                            "!= tensor n_per_row=%lld for %s; ignoring sidecar\n",
                            sc.in_features, (long long)n_per_row, name.c_str());
                } else {
                    have_sidecar = true;
                }
            }
            // Silently ignore missing sidecars; warn-spam helps no one.
        }

        for (const auto & tgt : targets) {
            const auto * tgt_traits = ggml_get_type_traits(tgt.t);
            if (!tgt_traits || !tgt_traits->to_float) {
                fprintf(stderr, "[quantize-cost] type has no to_float: %s\n", tgt.name);
                continue;
            }
            // Look up imatrix vector for this tensor (if loaded). NULL if no
            // imatrix file or if this tensor wasn't in the imatrix; the kernel
            // handles NULL but quality on imatrix-dependent types (IQ_K family)
            // will be much worse without it.
            const float * imatrix_ptr = nullptr;
            if (!imatrix_data.empty()) {
                auto it = imatrix_data.find(name);
                if (it != imatrix_data.end()) imatrix_ptr = it->second.data();
            }

            // Some types require an imatrix; skip cleanly with NaN if so.
            if (ggml_quantize_requires_imatrix(tgt.t) && imatrix_ptr == nullptr) {
                fprintf(out, "%s,%lld,%s,%s,0,nan,0.0,nan\n",
                        name.c_str(), (long long)n_elements,
                        ggml_type_name(src_type), tgt.name);
                continue;
            }

            // Block-size alignment check: K-quants + most IQ-K types require
            // n_per_row % blck_size == 0 (typically QK_K=256). gpt-oss-20b's
            // 2880-wide tensors fail this for any QK_K=256 type. Skip cleanly
            // with NaN instead of letting ggml_quantize_chunk hit GGML_ASSERT
            // and abort the whole process.
            const int64_t blck = ggml_blck_size(tgt.t);
            if (blck > 0 && n_per_row % blck != 0) {
                fprintf(out, "%s,%lld,%s,%s,0,nan,0.0,nan\n",
                        name.c_str(), (long long)n_elements,
                        ggml_type_name(src_type), tgt.name);
                continue;
            }

            // Allocate quantized scratch sized for full tensor.
            const size_t row_size = ggml_row_size(tgt.t, n_per_row);
            const size_t qbytes = row_size * n_rows;
            std::vector<uint8_t> qbuf(qbytes);

            ggml_quantize_init(tgt.t);
            (void)ggml_quantize_chunk(tgt.t, src_f32.data(), qbuf.data(),
                                      /*start=*/ 0, n_rows, n_per_row,
                                      /*imatrix=*/ imatrix_ptr);

            // Dequantize row-by-row. Required for types with row_meta_size > 0
            // (IQ4_KS, IQ3_KS, IQ4_KSS, IQ4_KT) — their to_float reads a per-row
            // scale from offset 0 and assumes k == n_per_row. Calling once with
            // k == n_elements silently mis-decodes rows 1..N-1 (no error, just
            // garbage).  Row-iterated decode also works for row_meta_size=0
            // types (K-quants etc.) with no measurable overhead.
            std::vector<float> deq_f32(n_elements);
            for (int64_t r = 0; r < n_rows; r++) {
                tgt_traits->to_float(
                    qbuf.data()    + r * row_size,
                    deq_f32.data() + r * n_per_row,
                    n_per_row);
            }

            double sse = 0.0;
            for (int64_t i = 0; i < n_elements; i++) {
                const double d = (double)src_f32[i] - (double)deq_f32[i];
                sse += d * d;
            }
            const double mse = sse / (double)n_elements;
            const double bpw = (double)qbytes * 8.0 / (double)n_elements;

            // Fisher row-weighted output MSE. Only computed when this tensor
            // has a loaded sidecar — for everything else (no sidecar, kernel
            // unsupported, etc.), we emit NaN.
            double fisher_out_mse = std::nan("");
            if (have_sidecar) {
                fisher_out_mse = compute_fisher_output_mse(
                    src_f32.data(), deq_f32.data(),
                    sc.X.data(), sc.fisher_weights.data(),
                    n_rows, n_per_row, (int64_t)sc.n_rows);
            }

            fprintf(out, "%s,%lld,%s,%s,%zu,%.10g,%.6f,%.10g\n",
                    name.c_str(), (long long)n_elements,
                    ggml_type_name(src_type), tgt.name,
                    qbytes, mse, bpw, fisher_out_mse);
            fflush(out);
        }

        n_done++;
        if (n_done % 50 == 0) {
            fprintf(stderr, "[quantize-cost] %lld tensors measured (%lld skipped) ...\n",
                    (long long)n_done, (long long)n_skip);
        }
    }

    fprintf(stderr, "[quantize-cost] done: %lld tensors measured, %lld skipped, %lld total\n",
            (long long)n_done, (long long)n_skip, (long long)n_total);

    fclose(out);
    fclose(fp);
    ggml_quantize_free();
    gguf_free(gctx);
    ggml_free(meta_ctx);
    return 0;
}
