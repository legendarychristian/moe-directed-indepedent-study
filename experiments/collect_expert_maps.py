"""Smoke test: capture native OLMoE router maps by prompt and decoding step."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
PROMPTS = [
    "Why do mirrors reverse left and right but not up and down?",
    "Explain how a hash table handles collisions.",
]
NEW_TOKENS = 6
OUTPUT = Path("data/routing/smoke_expert_maps.pt")

torch.manual_seed(42)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=0,
    local_files_only=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

layers = model.model.layers
observed = {}
steps = []


def make_hook(layer_id):
    def hook(module, args, result):
        logits = result[0]
        observed[layer_id] = torch.softmax(
            logits[-1].float(), dim=-1
        ).detach().cpu()
        if layer_id == len(layers) - 1:
            assert len(observed) == len(layers)
            steps.append(torch.stack([observed[i] for i in range(len(layers))]))
            observed.clear()
    return hook


hooks = [
    layer.mlp.gate.register_forward_hook(make_hook(i))
    for i, layer in enumerate(layers)
]

maps_by_prompt = []
ids_by_prompt = []
try:
    with torch.inference_mode():
        for prompt in PROMPTS:
            observed.clear()
            steps.clear()
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            result = model.generate(
                **inputs,
                max_new_tokens=NEW_TOKENS,
                min_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            assert len(steps) == NEW_TOKENS, (
                f"Captured {len(steps)} steps, expected {NEW_TOKENS}"
            )
            maps_by_prompt.append(torch.stack(steps))
            ids_by_prompt.append(
                result[0, inputs["input_ids"].shape[1]:].tolist()
            )
            print(f"Captured {len(steps)} maps: {prompt}")
finally:
    for handle in hooks:
        handle.remove()

maps = torch.stack(maps_by_prompt)
assert maps.shape == (len(PROMPTS), NEW_TOKENS, len(layers), 64)
assert torch.isfinite(maps).all()
assert torch.allclose(maps.sum(dim=-1), torch.ones_like(maps.sum(-1)), atol=1e-5)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "expert_maps": maps,
    "prompts": PROMPTS,
    "generated_token_ids": ids_by_prompt,
    "model_id": MODEL_ID,
    "max_new_tokens": NEW_TOKENS,
    "collection_note": "One map per model forward: prefill-last-token then decode; smoke only",
}, OUTPUT)

loaded = torch.load(OUTPUT, map_location="cpu", weights_only=True)
assert torch.equal(loaded["expert_maps"], maps)
print("Shape:", tuple(maps.shape))
print("Saved:", OUTPUT)
print("Step 11A native routing-map capture PASSED")
