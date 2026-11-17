"""Load pretrained OLMoE with CPU-resident experts."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_with_cpu_experts(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=0,
        local_files_only=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    blocks = [layer.mlp for layer in model.model.layers]

    before = torch.cuda.memory_allocated()

    for mlp in blocks:
        mlp.experts.to("cpu")

    after = torch.cuda.memory_allocated()

    assert all(
        mlp.experts.gate_up_proj.device.type == "cpu"
        for mlp in blocks
    )

    return model, tokenizer, blocks, before - after
