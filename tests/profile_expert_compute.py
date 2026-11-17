"""Step 11C: profile CPU and GPU expert execution on the same 8-token test."""
import os
assert os.getenv("MOE_PROFILE_EXPERTS") == "1", "Run with MOE_PROFILE_EXPERTS=1"

import torch
from moe_hybrid.inference import HybridMoE

PROMPT = "What causes ocean tides on Earth?"
EXPECTED = [187, 187, 48, 8705, 246, 1487, 403, 4269]
engine = HybridMoE(capacity=32, predictor_method="nn")

try:
    for enabled in (False, True):
        result = engine.generate(PROMPT, max_new_tokens=8, prefetch=enabled)
        torch.cuda.synchronize()
        stats = engine.stats
        pairs = stats.get("gpu_expert_events", [])
        assert len(pairs) == result["gpu_exec"] == 240
        gpu_math_ms = sum(start.elapsed_time(end) for start, end in pairs)
        cpu_math_ms = stats.get("cpu_expert_math_ms", 0.0)
        activation_ms = stats.get("activation_to_gpu_call_ms", 0.0)

        assert gpu_math_ms > 0
        assert cpu_math_ms > 0
        assert activation_ms > 0
        assert result["new_token_ids"] == EXPECTED
        assert (result["hits"], result["misses"], result["prefetch_loads"]) == (
            (62, 178, 15) if enabled else (50, 190, 0)
        )

        # CPU vs GPU sum is NOT an end-to-end latency breakdown: streams overlap,
        # CUDA kernel timing excludes other model ops, and event recording adds overhead.
        label = "ON" if enabled else "OFF"
        print(f"\nPrefetch {label}")
        print(f"  Total inference time: {result['elapsed_s'] * 1000:.1f} ms")
        print(f"  CPU expert compute (sum, wall): {cpu_math_ms:.1f} ms")
        print(f"  GPU expert math (sum, CUDA events): {gpu_math_ms:.1f} ms")
        print(f"  CPU-to-GPU activation call time (sum, wall): {activation_ms:.1f} ms")
        print(f"  CPU/GPU expert executions: {result['cpu_exec']} / {result['gpu_exec']}")
        print(f"  Cache hits / misses: {result['hits']} / {result['misses']}")

    print("\nStep 11C expert compute profiling PASSED")
finally:
    engine.close()
