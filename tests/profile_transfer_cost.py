"""Step 11D: profile transfer volumes, CUDA copies and load stalls."""
import os

assert os.getenv("MOE_PROFILE_TRANSFERS") == "1", "Run with MOE_PROFILE_TRANSFERS=1"

import torch
from moe_hybrid.inference import HybridMoE

PROMPT = "What causes ocean tides on Earth?"
EXPECTED = [187, 187, 48, 8705, 246, 1487, 403, 4269]
MIB = 1024 ** 2
engine = HybridMoE(capacity=32, predictor_method="nn")

try:
    for enabled in (False, True):
        result = engine.generate(PROMPT, max_new_tokens=8, prefetch=enabled)
        torch.cuda.synchronize()  # Every recorded CUDA event must have completed.
        cache = engine.cache
        copy_events = cache.copy_events
        wait_events = cache.load_wait_events
        copy_ms = sum(a.elapsed_time(b) for a, b, _ in copy_events)
        demand_wait_ms = sum(
            a.elapsed_time(b) for a, b, src in wait_events if src == "demand"
        )
        prefetch_wait_ms = sum(
            a.elapsed_time(b) for a, b, src in wait_events if src == "prefetch"
        )
        pending = sum(1 for e in cache.entries.values() if e["source"] == "prefetch")

        assert result["new_token_ids"] == EXPECTED
        assert (result["hits"], result["misses"], result["prefetch_loads"]) == (
            (62, 178, 15) if enabled else (50, 190, 0)
        )
        assert len(copy_events) == result["demand_loads"] + result["prefetch_loads"]
        assert cache.weight_h2d_bytes == cache.demand_h2d_bytes + cache.prefetch_h2d_bytes
        assert cache.weight_h2d_bytes // MIB == 12 * len(copy_events)
        assert cache.prefetch_h2d_bytes // MIB == 12 * result["prefetch_loads"]
        assert 0 <= len(wait_events) <= result["gpu_exec"]
        assert copy_ms >= 0 and demand_wait_ms >= 0 and prefetch_wait_ms >= 0

        print(f"\nPrefetch {'ON' if enabled else 'OFF'}")
        print(f"  Total inference: {1000 * result['elapsed_s']:.1f} ms")
        print(f"  Weight transfers: {len(copy_events)} ({result['demand_loads']} demand + {result['prefetch_loads']} prefetch)")
        print(f"  Weight H2D: {cache.weight_h2d_bytes / MIB:.0f} MiB total, "
              f"{cache.demand_h2d_bytes / MIB:.0f} MiB demand, {cache.prefetch_h2d_bytes / MIB:.0f} MiB prefetch")
        print(f"  Pinned staging (host summed): {cache.staging_ms:.1f} ms")
        print(f"  CUDA copy-stream intervals (sum): {copy_ms:.1f} ms")
        print(f"  Host enqueue (summed): {cache.host_enqueue_ms:.1f} ms")
        print(f"  GPU stream waits for demand loads (sum): {demand_wait_ms:.1f} ms")
        print(f"  GPU stream waits for useful prefetched loads (sum): {prefetch_wait_ms:.1f} ms")
        print(f"  Host eviction waits (sum): {cache.eviction_wait_ms:.1f} ms")
        print(f"  Useful / evicted-wasted / still-pending prefetches: "
              f"{cache.useful_prefetches} / {cache.wasted_prefetches} / {pending}")

    print("\nStep 11D transfer + load-wait profiling PASSED")
finally:
    engine.close()
