"""Step 11E: live nearest-neighbor vs router top-1, plus lookup cost."""
import os

assert os.getenv("MOE_PROFILE_PREDICTIONS") == "1", (
    "Run with MOE_PROFILE_PREDICTIONS=1"
)

from moe_hybrid.inference import HybridMoE

PROMPT = "What causes ocean tides on Earth?"
EXPECTED_IDS = [187, 187, 48, 8705, 246, 1487, 403, 4269]
method = os.getenv("MOE_PREDICTOR", "nn")
assert method in ("nn", "mean")

engine = HybridMoE(capacity=32, predictor_method=method)
try:
    result = engine.generate(PROMPT, max_new_tokens=8, prefetch=True)
    stats = engine.stats
    scored = stats.get("predictions_scored", 0)
    correct = stats.get("top1_correct", 0)
    accepted = stats.get("accepted_predictions", 0)
    accepted_correct = stats.get("accepted_correct", 0)
    lookup_ms = stats.get("prediction_lookup_ms", 0.0)

    p = engine.predictor
    tensor_bytes = p.history.numel() * p.history.element_size() + sum(
        x.numel() * x.element_size() for x in p.prefixes.values()
    )

    print("Predictor:", method)
    print("Predictions scored:", scored)
    print("Exact top-1 matches:", correct)
    print("Exact top-1 accuracy (%):", round(100 * correct / max(1, scored), 2))
    print("Confidence-gated predictions:", accepted)
    print("Gated correct:", accepted_correct)
    print("Gated top-1 accuracy (%):", round(100 * accepted_correct / max(1, accepted), 2))
    print("Prediction lookup total (ms):", round(lookup_ms, 3))
    print("Prediction lookup mean (ms):", round(lookup_ms / max(1, scored), 4))
    print("History + prefix tensor memory (MiB):", round(tensor_bytes / 1024 ** 2, 3))
    print("Prefetch loads:", result["prefetch_loads"])
    print("Cache hits / misses:", result["hits"], result["misses"])

    assert result["new_token_ids"] == EXPECTED_IDS
    assert scored == stats["predictions"] and scored > 0
    assert 0 <= accepted_correct <= accepted <= scored
    assert 0 <= correct <= scored
    assert lookup_ms >= 0
    assert (result["hits"], result["misses"]) == (
        (62, 178) if method == "nn" else (52, 188)
    )
    print("Step 11E live prediction profiling PASSED")
finally:
    engine.close()
