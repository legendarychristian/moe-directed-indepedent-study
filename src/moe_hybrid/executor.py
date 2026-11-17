"""Hybrid CPU/GPU MoE expert execution."""

import os
import time
import torch

PROFILE_EXPERTS = os.getenv("MOE_PROFILE_EXPERTS", "0") == "1"
import torch.nn.functional as F


def make_hybrid_forward(cache, get_stats):
    def hybrid_forward(self, hidden_states, top_k_index, top_k_weights):
        stats = get_stats()
        layer_id = self.hybrid_layer_id
        output = torch.zeros_like(hidden_states)
        cpu_inputs = hidden_states.float().cpu()

        for token_id in range(hidden_states.shape[0]):
            x_gpu = hidden_states[token_id:token_id + 1]
            x_cpu = cpu_inputs[token_id:token_id + 1]
            for rank in range(top_k_index.shape[1]):
                expert_id = int(top_k_index[token_id, rank])
                weight = top_k_weights[token_id, rank]
                if rank == 0:
                    gate_up, down = cache.demand((layer_id, expert_id))
                    if PROFILE_EXPERTS:
                        gpu_start = torch.cuda.Event(enable_timing=True)
                        gpu_end = torch.cuda.Event(enable_timing=True)
                        gpu_start.record()
                    gate, up = F.linear(x_gpu, gate_up).chunk(2, dim=-1)
                    value = F.linear(F.silu(gate) * up, down)
                    if PROFILE_EXPERTS:
                        gpu_end.record()
                        stats.setdefault("gpu_expert_events", []).append((gpu_start, gpu_end))
                    stats["gpu_exec"] += 1
                else:
                    cpu_start = time.perf_counter() if PROFILE_EXPERTS else None
                    gate_up = self.gate_up_proj[expert_id].float()
                    down = self.down_proj[expert_id].float()
                    gate, up = F.linear(x_cpu, gate_up).chunk(2, dim=-1)
                    value = F.linear(F.silu(gate) * up, down)
                    if PROFILE_EXPERTS:
                        stats["cpu_expert_math_ms"] = (
                            stats.get("cpu_expert_math_ms", 0.0)
                            + (time.perf_counter() - cpu_start) * 1000.0
                        )
                    activation_start = time.perf_counter() if PROFILE_EXPERTS else None
                    value = value.to("cuda", dtype=hidden_states.dtype)
                    if PROFILE_EXPERTS:
                        stats["activation_to_gpu_call_ms"] = (
                            stats.get("activation_to_gpu_call_ms", 0.0)
                            + (time.perf_counter() - activation_start) * 1000.0
                        )
                    stats["cpu_exec"] += 1
                output[token_id:token_id + 1] += weight * value
        return output

    return hybrid_forward
