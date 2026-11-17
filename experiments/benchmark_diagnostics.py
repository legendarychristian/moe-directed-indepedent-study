"""Reproducible paired latency/cache benchmark for the modular hybrid MoE runtime.

Run from the repository root with PYTHONPATH=src. Output is one CSV with all
per-run metrics and one JSON with configuration and summary statistics.
"""
import argparse
import os
import math
import csv
import json
import hashlib
import subprocess
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from moe_hybrid.inference import HybridMoE


def collect_profile_metrics(engine):
    """Return scalar diagnostic fields from the most recent generate() call.

    Disabled profilers produce None, not misleading zeros. Sums are *not*
    additive components of end-to-end latency; CPU and GPU can overlap.
    """
    cache = engine.cache
    stats = engine.stats
    profile_cpu = os.getenv("MOE_PROFILE_EXPERTS") == "1"
    profile_h2d = os.getenv("MOE_PROFILE_TRANSFERS") == "1"
    profile_pred = os.getenv("MOE_PROFILE_PREDICTIONS") == "1"
    mib = 1024 ** 2

    scored = stats.get("predictions_scored", 0)
    accepted = stats.get("accepted_predictions", 0)
    prefix_bytes = sum(x.numel() * x.element_size() for x in engine.predictor.prefixes.values())
    map_bytes = engine.predictor.history.numel() * engine.predictor.history.element_size()

    copy_events = cache.copy_events if profile_h2d else []
    wait_events = cache.load_wait_events if profile_h2d else []
    gpu_events = stats.get("gpu_expert_events", []) if profile_cpu else []

    return {
        "cpu_expert_compute_ms_sum": stats.get("cpu_expert_math_ms", 0.0) if profile_cpu else None,
        "gpu_expert_math_cuda_ms_sum": sum(a.elapsed_time(b) for a, b in gpu_events) if profile_cpu else None,
        "activation_h2d_call_ms_sum": stats.get("activation_to_gpu_call_ms", 0.0) if profile_cpu else None,
        "weight_h2d_mib": cache.weight_h2d_bytes / mib,
        "demand_weight_h2d_mib": cache.demand_h2d_bytes / mib,
        "prefetch_weight_h2d_mib": cache.prefetch_h2d_bytes / mib,
        "pinned_staging_ms_sum": cache.staging_ms,
        "cuda_weight_copy_ms_sum": sum(a.elapsed_time(b) for a, b, _ in copy_events) if profile_h2d else None,
        "host_enqueue_ms_sum": cache.host_enqueue_ms if profile_h2d else None,
        "gpu_demand_load_wait_ms_sum": sum(a.elapsed_time(b) for a, b, src in wait_events if src == "demand") if profile_h2d else None,
        "gpu_prefetch_load_wait_ms_sum": sum(a.elapsed_time(b) for a, b, src in wait_events if src == "prefetch") if profile_h2d else None,
        "host_eviction_wait_ms_sum": cache.eviction_wait_ms if profile_h2d else None,
        "unconsumed_prefetches_at_end": sum(e["source"] == "prefetch" for e in cache.entries.values()),
        "predictions_scored": scored if profile_pred else None,
        "prediction_top1_accuracy": stats.get("top1_correct", 0) / scored if profile_pred and scored else None,
        "confidence_gated_predictions": accepted if profile_pred else None,
        "confidence_gated_top1_accuracy": stats.get("accepted_correct", 0) / accepted if profile_pred and accepted else None,
        "prediction_lookup_ms_sum": stats.get("prediction_lookup_ms", 0.0) if profile_pred else None,
        "prediction_lookup_ms_mean": stats.get("prediction_lookup_ms", 0.0) / scored if profile_pred and scored else None,
        "predictor_tensor_mib": (prefix_bytes + map_bytes) / mib,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What causes ocean tides on Earth?")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--predictor", choices=["nn", "mean"], default="nn")
    parser.add_argument("--history", default="step7b_history.pt")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--measured-pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/benchmark_smoke")
    args = parser.parse_args()

    if args.cache_size < 1 or args.tokens < 1:
        parser.error("--cache-size and --tokens must be positive")
    if args.warmup_pairs < 0 or args.measured_pairs < 1:
        parser.error("Need >=0 warmup pairs and >=1 measured pairs")

    if not (0 <= args.threshold <= 1 and 0 <= args.alpha <= 1):
        parser.error("--threshold and --alpha must be between 0 and 1")

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine = HybridMoE(
        model_id=args.model,
        capacity=args.cache_size,
        predictor_method=args.predictor,
        history_path=args.history,
        threshold=args.threshold,
        alpha=args.alpha,
    )

    rows = []
    timings = {False: [], True: []}
    deltas = []
    reference_ids = None
    try:
        for i in range(args.warmup_pairs + args.measured_pairs):
            order = (False, True) if i % 2 == 0 else (True, False)
            pair = {}
            phase = "warmup" if i < args.warmup_pairs else "measured"
            for enabled in order:
                m = engine.generate(
                    args.prompt,
                    max_new_tokens=args.tokens,
                    prefetch=enabled,
                )
                ids = m.pop("new_token_ids")
                if reference_ids is None:
                    reference_ids = ids
                assert ids == reference_ids, "Prefetch changed generated tokens"

                # Store token intervals as machine-readable JSON within CSV.
                if "token_intervals_ms" in m:
                    m["token_intervals_ms"] = json.dumps(m["token_intervals_ms"])
                row = {
                    "round": i + 1,
                    "phase": phase,
                    "prefetch": "on" if enabled else "off",
                    "new_tokens": len(ids),
                    "new_token_ids": json.dumps(ids),
                    "total_transfers": m["demand_loads"] + m["prefetch_loads"],
                    "hit_rate": m["hits"] / max(m["hits"] + m["misses"], 1),
                    "generated_tokens_per_s": len(ids) / m["elapsed_s"],
                    **m,
                    **collect_profile_metrics(engine),
                }
                rows.append(row)
                pair[enabled] = m["elapsed_s"]
                if phase == "measured":
                    timings[enabled].append(m["elapsed_s"])
                print(
                    f"Round {i+1} {phase} {row['prefetch']}: "
                    f"{m['elapsed_s']:.3f}s, hits={m['hits']}, "
                    f"misses={m['misses']}, prefetch={m['prefetch_loads']}"
                )
            if phase == "measured":
                deltas.append(pair[True] - pair[False])
    finally:
        engine.close()

    off = statistics.median(timings[False])
    on = statistics.median(timings[True])
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    history_sha256 = hashlib.sha256(Path(args.history).read_bytes()).hexdigest()
    measured_rows = [r for r in rows if r["phase"] == "measured"]
    metric_fields = [
        "ttft_ms", "tpot_ms", "decode_tokens_per_s", "elapsed_s",
        "hit_rate", "weight_h2d_mib", "cpu_expert_compute_ms_sum",
        "gpu_expert_math_cuda_ms_sum", "activation_h2d_call_ms_sum",
        "pinned_staging_ms_sum", "cuda_weight_copy_ms_sum",
        "gpu_demand_load_wait_ms_sum", "gpu_prefetch_load_wait_ms_sum",
        "host_enqueue_ms_sum", "host_eviction_wait_ms_sum",
        "prediction_top1_accuracy", "confidence_gated_top1_accuracy",
        "prediction_lookup_ms_sum", "prediction_lookup_ms_mean",
        "predictor_tensor_mib", "unconsumed_prefetches_at_end",
        "hits", "misses", "demand_loads", "prefetch_loads",
        "useful_prefetches", "wasted_prefetches",
    ]
    median_by_mode = {}
    for mode in ("off", "on"):
        mode_rows = [r for r in measured_rows if r["prefetch"] == mode]
        median_by_mode[mode] = {}
        for field in metric_fields:
            vals = [r.get(field) for r in mode_rows]
            vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
            median_by_mode[mode][field] = statistics.median(vals) if vals else None

    summary = {
        "metric_medians_by_mode": median_by_mode,
        "profiling_flags": {
            "experts": os.getenv("MOE_PROFILE_EXPERTS") == "1",
            "transfers": os.getenv("MOE_PROFILE_TRANSFERS") == "1",
            "predictions": os.getenv("MOE_PROFILE_PREDICTIONS") == "1",
        },
        "measurement_note": "CUDA events, host call times, and wall times can overlap; do not sum as exclusive latency. Profiler instrumentation changes timing.",
        "git_commit": git_commit,
        "history_sha256": history_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "reference_new_token_ids": reference_ids,
        "all_output_ids_identical": True,
        "median_off_s": off,
        "median_on_s": on,
        "median_difference_on_minus_off_s": on - off,
        "percent_difference": 100 * (on / off - 1),
        "paired_differences_on_minus_off_s": deltas,
    }

    with (out / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(f"\nMedian OFF: {off:.3f}s; ON: {on:.3f}s; change: {summary['percent_difference']:+.2f}%")
    print("Outputs identical:", summary["all_output_ids_identical"])
    print("Saved:", out / "runs.csv")
    print("Saved:", out / "summary.json")
    print("Step 11F diagnostic benchmark PASSED")


if __name__ == "__main__":
    main()
