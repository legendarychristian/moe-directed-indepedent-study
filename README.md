# Hybrid MoE Inference

A research prototype for running **OLMoE-1B-7B** with CPU-resident experts and a small GPU expert cache. We study whether historical expert-map prediction and asynchronous prefetching can reduce inference latency under limited GPU memory.

## System architecture

```text
Prompt → OLMoE (attention + routers on GPU)
                  │
                  ├── Router probabilities → historical expert-map predictor
                  │                           (nearest neighbor or mean)
                  │                                  │
                  │                           Next-layer expert prediction
                  │                                  │
                  │                           Confidence gate (0.98)
                  │                                  │
                  │                           Async CPU → GPU prefetch
                  │
                  └── MoE expert execution ← Shared GPU cache (MRS eviction)
                        │
                        ├── Highest-ranked expert: GPU (cache hit or demand load)
                        └── Other 7 selected experts: CPU
```

All 16 layers' expert weights reside in CPU RAM. The shared GPU cache holds a configurable number of expert weight sets (4, 16, or 32 in our initial experiments). Weights are staged in pinned CPU memory and copied to GPU on a separate CUDA stream. The predictor looks up similar historical router trajectories to suggest which expert to prefetch in the next layer. The baseline uses the mean historical router probabilities.

**Scope:** The current executor always runs the highest-ranked expert on GPU and the remaining seven on CPU; it does not implement a dynamic CPU/GPU placement scheduler. The predictor is retrieval-based, not separately trained.

## Evaluation metrics

| Metric | Why it matters |
| --- | --- |
| **TTFT (time to first token)** | Measures prompt prefill and first-token latency. |
| **TPOT (time per output token)** | Measures latency between generated tokens. |
| **Tokens per second** | Measures decoding throughput. |
| **Expert cache hit rate** | Indicates whether useful experts are already resident on GPU. |
| **CPU–GPU transfer overhead** | Tracks weight traffic, speculative loads, wasted prefetches, and copy costs. |
| **Time waiting on expert loads** | Measures exposed load stalls that prefetching aims to hide. |
| **Performance across cache sizes** | Tests the latency–memory tradeoff at different GPU cache capacities. |
| **CPU vs. GPU execution time** | Identifies compute bottlenecks and hardware utilization. |
| **Prediction accuracy and overhead** | Measures top-1 prediction quality, confidence-gated accuracy, lookup time, and memory use. |

We also track generated-token agreement with the native model, demand misses, useful/wasted prefetches, and total inference latency. Profiling is optional because instrumentation can affect timing; CPU wall times and CUDA event durations can overlap and should not be summed as exclusive latency.

## Repository layout

```text
src/moe_hybrid/     Model loading, expert executor, MRS cache, predictors, runtime
experiments/        Benchmark and data-collection scripts; archived prototypes
tests/              Smoke and profiler validation
data/prompts/       Prompt manifests
data/routing/       Saved routing-map datasets
results/            Benchmark CSV and JSON output
```

## Basic usage

Requires a CUDA-capable NVIDIA GPU, the Python dependencies used by this project, and a locally cached `allenai/OLMoE-1B-7B-0924-Instruct` checkpoint. The initial predictor history is `step7b_history.pt` in the repository root.

```bash
# Verify reference output and cache behavior.
PYTHONPATH=src python3 tests/smoke_inference.py

# Compare prefetching off vs. on, without expensive profiling.
PYTHONPATH=src python3 experiments/benchmark_hybrid.py \
  --prompt "What causes ocean tides on Earth?" \
  --cache-size 32 --predictor nn --tokens 8 \
  --warmup-pairs 2 --measured-pairs 5 \
  --output-dir results/example
```

The benchmark writes `runs.csv` and `summary.json`. Use `experiments/benchmark_diagnostics.py` with the profiling environment flags when collecting detailed cost-model measurements.

## Research status

The prototype reproduces the native model's generated tokens on validated inputs. Nearest-neighbor prefetching has improved expert cache hits in initial experiments, **but a consistent inference-latency speedup has not been demonstrated**. Current profiling indicates that CPU expert computation dominates the measured workload. Larger, independent evaluations are planned.
