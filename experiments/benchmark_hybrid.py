"""Reproducible paired latency/cache benchmark for the modular hybrid MoE runtime.

Run from the repository root with PYTHONPATH=src. Output is one CSV with all
per-run metrics and one JSON with configuration and summary statistics.
"""
import argparse
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
    summary = {
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
    print("Step 10H benchmark runner PASSED")


if __name__ == "__main__":
    main()
