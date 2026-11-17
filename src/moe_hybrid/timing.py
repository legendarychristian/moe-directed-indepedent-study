"""Per-token wall-clock timing for Hugging Face greedy generation.

Only for batch size 1. Synchronizes GPU before each timestamp, which adds
measurement overhead. Interpret as instrumented latency, not pure GPU kernel time.
"""
import time
import torch
from transformers import StoppingCriteria


class TokenTimer(StoppingCriteria):
    def __init__(self, start_time):
        self.start_time = start_time
        self.timestamps = []

    def __call__(self, input_ids, scores, **kwargs):
        assert input_ids.shape[0] == 1, "TokenTimer supports batch size 1"
        if input_ids.is_cuda:
            torch.cuda.synchronize(device=input_ids.device)
        self.timestamps.append(time.perf_counter())
        # Record timing without changing the generation stopping policy.
        return torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
