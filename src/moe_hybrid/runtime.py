"""Router observation and expert prefetch hooks."""

import os
import time
import torch

PROFILE_PREDICTIONS = os.getenv("MOE_PROFILE_PREDICTIONS", "0") == "1"


def make_router_hook(
    cache, predictor, observed, get_stats,
    get_prefetch_enabled, threshold, method,
):
    def router_hook(layer_id):
        def hook(module, args, result):
            stats = get_stats()
            logits, _, _ = result
            if layer_id == 0:
                observed.clear()  # New forward / decode step.
            cache.update_mrs(layer_id, logits)
            observed[layer_id] = torch.softmax(logits[-1].float(), -1).detach().cpu()
            if PROFILE_PREDICTIONS:
                # Score a prediction only after its target-layer router runs.
                pending = stats.setdefault("_prediction_pending", {})
                candidate = pending.pop(layer_id, None)
                if candidate is not None:
                    expert_id, accepted = candidate
                    actual_top1 = int(observed[layer_id].argmax().item())
                    correct = int(expert_id == actual_top1)
                    stats["predictions_scored"] = stats.get("predictions_scored", 0) + 1
                    stats["top1_correct"] = stats.get("top1_correct", 0) + correct
                    if accepted:
                        stats["accepted_predictions"] = stats.get("accepted_predictions", 0) + 1
                        stats["accepted_correct"] = stats.get("accepted_correct", 0) + correct

            # Prefill has many tokens. Start prefetching on 1-token decode steps.
            if not get_prefetch_enabled() or logits.shape[0] != 1:
                return
            if not (3 <= layer_id < 15):
                return

            known = layer_id + 1
            lookup_start = time.perf_counter() if PROFILE_PREDICTIONS else None
            predicted_id, confidence = predictor.predict(
                observed, known, method=method
            )
            if PROFILE_PREDICTIONS:
                stats["prediction_lookup_ms"] = (
                    stats.get("prediction_lookup_ms", 0.0)
                    + 1000 * (time.perf_counter() - lookup_start)
                )
                stats.setdefault("_prediction_pending", {})[known] = (
                    int(predicted_id), confidence >= threshold
                )

            stats["predictions"] += 1

            if confidence < threshold:
                stats["skipped"] += 1
                return

            cache.prefetch((known, predicted_id))
        return hook
    return router_hook
