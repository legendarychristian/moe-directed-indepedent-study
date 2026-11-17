"""Step 11B: confirm token timing does not alter inference behavior."""
import math
from moe_hybrid.inference import HybridMoE

EXPECTED = [187, 187, 48, 8705, 246, 1487, 403, 4269]
engine = HybridMoE(capacity=32, predictor_method="nn")

try:
    for enabled in (False, True):
        r = engine.generate(
            "What causes ocean tides on Earth?",
            max_new_tokens=8,
            prefetch=enabled,
        )
        assert r["new_token_ids"] == EXPECTED
        expected_counts = (62, 178, 15) if enabled else (50, 190, 0)
        assert (r["hits"], r["misses"], r["prefetch_loads"]) == expected_counts

        intervals = r["token_intervals_ms"]
        assert len(intervals) == len(EXPECTED) - 1
        assert all(math.isfinite(v) and v > 0 for v in intervals)
        assert 0 < r["ttft_ms"] <= r["elapsed_s"] * 1000
        assert math.isclose(
            r["tpot_ms"], sum(intervals) / len(intervals), rel_tol=1e-6
        )
        assert math.isclose(
            r["decode_tokens_per_s"], 1000.0 / r["tpot_ms"],
            rel_tol=1e-6,
        )
        print(
            f"Prefetch {'ON' if enabled else 'OFF'}: "
            f"TTFT={r['ttft_ms']:.2f}ms, "
            f"TPOT={r['tpot_ms']:.2f}ms, "
            f"decode={r['decode_tokens_per_s']:.3f} tokens/s, "
            f"hits={r['hits']}, misses={r['misses']}"
        )
    print("Step 11B per-token timing PASSED")
finally:
    engine.close()
