# llama-quantize-cost

Per-(tensor, format) MSE measurement tool for `llama.cpp`. Round-trip quantize → dequantize against the original weights, write a CSV of reconstruction error and quantized size for every (tensor, candidate-format) pair.

Useful for any mixed-precision allocator, sensitivity analysis, or quant-aware pruning work. The original consumer is [prismaquant-llama](https://github.com/jimbothigpen/prismaquant-llama)'s allocator, which combines this CSV with a Hessian probe (Fisher H_trace per Linear) to solve a multi-choice knapsack picking one format per tensor that minimizes `sum(0.5 · H_trace[t] · MSE[t, fmt[t]])` under a size budget.

## What it produces

```
tensor_name,n_elements,src_type,fmt,size_bytes,mse,bpw
blk.0.attn_q.weight,33554432,bf16,Q4_K,18874368,1.86e-06,4.500
blk.0.attn_q.weight,33554432,bf16,IQ4_KSS,16809984,3.02e-06,4.008
...
```

One row per (tensor, candidate format). Errors are MSE under imatrix scaling when the imatrix is provided. Types that require an imatrix get `mse=nan` if no imatrix is given. Block-alignment-incompatible (tensor, format) combos likewise produce `mse=nan` rather than aborting.

## Why it isn't standalone

The tool is fundamentally an extension to a `llama.cpp` / `ggml` fork — it `#include`s `ggml.h` and `gguf.h`, calls `ggml_get_type_traits`, and iterates `GGML_TYPE_COUNT` to discover available formats dynamically. It must be built **inside** an existing llama.cpp source tree and links against that tree's `llama-common`, `llama`, and `ggml` libraries.

This means it picks up whatever quant types your fork ships — works with mainline llama.cpp's standard set, and with forks that add custom types (e.g. [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)'s `IQ?_K` family, [`jimbothigpen/frankenturbo2`](https://github.com/jimbothigpen/frankenturbo2)'s TURBO/TCQ/PLANAR/INNERQ types).

## Integration

### One-time setup: drop into your llama.cpp fork

```bash
# Clone next to your llama.cpp tree, then symlink (or copy) into tools/
git clone https://github.com/jimbothigpen/llama-quantize-cost \
    /path/to/your/llama.cpp/tools/quantize-cost

# Wire it into your fork's tools build
echo 'add_subdirectory(quantize-cost)' >> /path/to/your/llama.cpp/tools/CMakeLists.txt
```

### Build

```bash
cd /path/to/your/llama.cpp/build
cmake ..   # reconfigure to pick up the new subdir
cmake --build . --target llama-quantize-cost
```

The binary lands at `your/llama.cpp/build/bin/llama-quantize-cost`, alongside `llama-quantize`, `llama-imatrix`, etc.

### Verify

```bash
your/llama.cpp/build/bin/llama-quantize-cost --help
# expected: usage line + flag descriptions
```

## Usage

```
llama-quantize-cost --model SRC.gguf --types T1,T2,... --output costs.csv
                    [--imatrix IMATRIX.dat]
                    [--include-regex REGEX] [--exclude-regex REGEX]
                    [--threads N]
```

- `--model` — input GGUF (typically a BF16 or F16 reference).
- `--types` — comma-separated quant names to evaluate. Names that don't resolve against the `ggml` type registry of the build (e.g. recipe-only names like `Q4_K_M`, `IQ3_M`) print a skip-warning and are dropped.
- `--imatrix` — optional importance matrix file (legacy binary or new GGUF format). Required for `IQ` family types; passed through to `ggml_quantize_chunk` so MSE matches what `llama-quantize` would actually produce.
- `--include-regex` / `--exclude-regex` — restrict to specific tensors. Useful for the prismaquant-llama exemplar workflow (measure layers 0 and 3 only, propagate to peers).
- `--threads N` — parallelize the per-tensor format loop across `N` CPU threads via OpenMP. Defaults to `omp_get_max_threads()` (all logical cores). Each tensor's source weights are materialized once and shared across worker threads; per-(tensor,format) work — quantize, dequantize, MSE, optional Fisher output MSE — runs in parallel and rows are emitted in deterministic `--types` order. Lower the value when running concurrently with other heavy CPU work.

Example:

```bash
llama-quantize-cost \
    --model model-bf16.gguf \
    --types Q4_K,Q5_K,Q6_K,IQ4_K,IQ4_KS,IQ4_KSS,IQ4_NL,IQ4_XS \
    --imatrix model.imatrix \
    --include-regex '^(output|token_embd|blk\.(0|3))\.' \
    --output costs.csv
```

## Output schema (CSV columns)

| column        | type   | meaning |
|---------------|--------|---------|
| `tensor_name` | string | GGUF tensor name |
| `n_elements`  | int    | total elements in the tensor |
| `src_type`    | string | source ggml type (e.g. `bf16`, `f16`) |
| `fmt`         | string | candidate format being measured |
| `size_bytes`  | int    | quantized size for this tensor under this format |
| `mse`         | float  | round-trip mean squared error, imatrix-scaled when applicable |
| `bpw`         | float  | bits per weight (`size_bytes · 8 / n_elements`) |

`mse=nan` indicates either (a) the format requires an imatrix that wasn't supplied, or (b) the tensor's row width isn't a multiple of the format's block size — both skipped cleanly so a single bad combo doesn't abort the whole run.

## Upstream status

This tool is a candidate for upstream landing in [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). If it ever lands there, this repo will become a thin compatibility/legacy shim. Until then, this is the canonical source.

## License

MIT — see [LICENSE](LICENSE).
