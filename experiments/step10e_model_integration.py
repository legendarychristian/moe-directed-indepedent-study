"""Step 9A: end-to-end OLMoE hybrid inference + MRS cache + async NN prefetch.

Two runs on one model: prefetch OFF and ON. One highest-ranked expert per
(token, layer) executes on GPU; the other seven execute on CPU.
The 4-slot GPU weight cache is shared by all 16 MoE layers.
"""
import time
import types
import statistics
import contextlib
import io
import torch
from moe_hybrid.executor import make_hybrid_forward
from moe_hybrid.model import load_model_with_cpu_experts
from moe_hybrid.predictor import ExpertMapPredictor
from moe_hybrid.cache import SharedCache
import os

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
PROMPT = os.environ.get(
    "MOE_PROMPT",
    "What causes ocean tides on Earth?",
)

MAX_NEW_TOKENS = int(
    os.environ.get("MOE_TOKENS", "8")
)
CAPACITY = int(os.environ.get("MOE_CACHE", "4"))
SIMILARITY_THRESHOLD = 0.98
ALPHA = 0.5
WARMUP_PAIRS = 1
MEASURED_PAIRS = 3

PREFETCH_PREDICTOR = os.environ.get(
    "MOE_PREDICTOR", "nn"
).lower()

assert PREFETCH_PREDICTOR in ("nn", "mean")

predictor = ExpertMapPredictor("step7b_history.pt")
assert predictor.history.shape == (48, 16, 64)

model, tokenizer, blocks, freed_bytes = (
    load_model_with_cpu_experts(MODEL_ID)
)

inputs = tokenizer(
    PROMPT, return_tensors="pt"
).to("cuda")

print("Offloaded layers:", len(blocks))
print(
    "GPU memory freed (GiB):",
    round(freed_bytes / 1024**3, 3),
)


cache = SharedCache(blocks, capacity=CAPACITY, alpha=ALPHA)
observed = {}
prefetch_enabled = False
stats = {}
hybrid_forward = make_hybrid_forward(cache, lambda: stats)


def router_hook(layer_id):
    def hook(module, args, result):
        logits, _, _ = result
        if layer_id == 0:
            observed.clear()  # New forward / decode step.
        cache.update_mrs(layer_id, logits)
        observed[layer_id] = torch.softmax(logits[-1].float(), -1).detach().cpu()

        # Prefill has many tokens. Start prefetching on 1-token decode steps.
        if not prefetch_enabled or logits.shape[0] != 1:
            return
        if not (3 <= layer_id < 15):
            return

        known = layer_id + 1
        predicted_id, confidence = predictor.predict(
            observed, known, method=PREFETCH_PREDICTOR
        )

        stats["predictions"] += 1

        if confidence < SIMILARITY_THRESHOLD:
            stats["skipped"] += 1
            return

        cache.prefetch((known, predicted_id))
    return hook




hooks = []
for i, mlp in enumerate(blocks):
    mlp.experts.hybrid_layer_id = i
    mlp.experts.forward = types.MethodType(hybrid_forward, mlp.experts)
    hooks.append(mlp.gate.register_forward_hook(router_hook(i)))


def run_mode(enabled):
    global prefetch_enabled, stats
    cache.reset()
    observed.clear()
    prefetch_enabled = enabled
    stats = {"predictions": 0, "skipped": 0, "gpu_exec": 0, "cpu_exec": 0}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print("\nPrefetch", "ON" if enabled else "OFF")
    print("  Generated IDs:", result[0, inputs["input_ids"].shape[1]:].tolist())
    print("  Wall time (s):", round(elapsed, 3))
    print("  Demand hits / misses:", cache.hits, cache.misses)
    print("  Demand / prefetch loads:", cache.demand_loads, cache.prefetch_loads)
    print("  Useful / wasted prefetches:", cache.useful_prefetches, cache.wasted_prefetches)
    print("  Evictions:", cache.evictions)
    print("  Predictions / skipped:", stats["predictions"], stats["skipped"])
    print("  GPU / CPU expert executions:", stats["gpu_exec"], stats["cpu_exec"])
    print("  Host pinned-buffer staging (ms):", round(cache.staging_ms, 2))
    print("  GPU cache entries:", len(cache.entries), "/", CAPACITY)
    assert cache.hits + cache.misses == stats["gpu_exec"]
    assert stats["cpu_exec"] == 7 * stats["gpu_exec"]
    assert len(cache.entries) <= CAPACITY
    return result, elapsed


# Step 9B: warm up, then compare the same model in alternating order.
WARMUP_PAIRS = 2
MEASURED_PAIRS = 5
latencies = {False: [], True: []}
paired_deltas = []
reference_ids = None

for round_idx in range(WARMUP_PAIRS + MEASURED_PAIRS):
    order = (False, True) if round_idx % 2 == 0 else (True, False)
    pair = {}

    for enabled in order:
        # Keep the existing run's verbose diagnostics out of the timing log.
        # Redirected printing occurs after run_mode's measured region.
        with contextlib.redirect_stdout(io.StringIO()):
            result, elapsed = run_mode(enabled)

        token_ids = result[0].tolist()
        if reference_ids is None:
            reference_ids = token_ids
        assert token_ids == reference_ids, "Generated token IDs changed"

        pair[enabled] = elapsed
        if round_idx >= WARMUP_PAIRS:
            latencies[enabled].append(elapsed)

        print(
            f"Round {round_idx + 1} "
            f"({'warmup' if round_idx < WARMUP_PAIRS else 'measured'}), "
            f"prefetch {'ON' if enabled else 'OFF'}: "
            f"{elapsed:.3f}s, "
            f"hits={cache.hits}, misses={cache.misses}, "
            f"speculative_loads={cache.prefetch_loads}"
        )

    if round_idx >= WARMUP_PAIRS:
        paired_deltas.append(pair[True] - pair[False])

median_off = statistics.median(latencies[False])
median_on = statistics.median(latencies[True])

print("\nStep 9B repeated benchmark:")
print("Predictor:", PREFETCH_PREDICTOR)
print("\nPrompt:", PROMPT)
print("Maximum new tokens:", MAX_NEW_TOKENS)
print("Warmup pairs:", WARMUP_PAIRS)
print("Measured pairs:", MEASURED_PAIRS)
print("Prefetch OFF median (s):", round(median_off, 3))
print("Prefetch ON median (s):", round(median_on, 3))
print("Median difference ON - OFF (s):", round(median_on - median_off, 3))
print("Change (%):", round(100 * (median_on / median_off - 1), 2))
print("OFF range (s):", [round(min(latencies[False]), 3), round(max(latencies[False]), 3)])
print("ON range (s):", [round(min(latencies[True]), 3), round(max(latencies[True]), 3)])
print("Paired differences ON - OFF (s):", [round(x, 3) for x in paired_deltas])
print("All generated token IDs identical:", True)

assert all(mlp.experts.gate_up_proj.device.type == "cpu" for mlp in blocks)
for hook in hooks:
    hook.remove()
print("\nStep 9B repeated integrated benchmark PASSED")
