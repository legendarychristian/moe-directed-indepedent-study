
"""Reusable OLMoE hybrid inference runtime."""

import time
import types

import torch

from .cache import SharedCache
from .executor import make_hybrid_forward
from .model import load_model_with_cpu_experts
from .predictor import ExpertMapPredictor
from .runtime import make_router_hook
from .timing import TokenTimer


class HybridMoE:
    def __init__(
        self,
        model_id="allenai/OLMoE-1B-7B-0924-Instruct",
        history_path="step7b_history.pt",
        capacity=32,
        predictor_method="nn",
        threshold=0.98,
        alpha=0.5,
    ):
        if predictor_method not in ("nn", "mean"):
            raise ValueError(
                "predictor_method must be 'nn' or 'mean'"
            )

        self.model, self.tokenizer, self.blocks, self.freed_bytes = (
            load_model_with_cpu_experts(model_id)
        )

        self.cache = SharedCache(
            self.blocks, capacity=capacity, alpha=alpha
        )
        self.predictor = ExpertMapPredictor(history_path)
        self.observed = {}
        self.stats = {}
        self.prefetch_enabled = False
        self.capacity = capacity

        forward = make_hybrid_forward(
            self.cache, lambda: self.stats
        )

        router_hook = make_router_hook(
            cache=self.cache,
            predictor=self.predictor,
            observed=self.observed,
            get_stats=lambda: self.stats,
            get_prefetch_enabled=lambda: self.prefetch_enabled,
            threshold=threshold,
            method=predictor_method,
        )

        self._hooks = []
        self._original_forwards = []

        for layer_id, mlp in enumerate(self.blocks):
            self._original_forwards.append(
                mlp.experts.forward
            )
            mlp.experts.hybrid_layer_id = layer_id

            mlp.experts.forward = types.MethodType(
                forward, mlp.experts
            )

            self._hooks.append(
                mlp.gate.register_forward_hook(
                    router_hook(layer_id)
                )
            )

    def generate(
        self, prompt, max_new_tokens=8, prefetch=False
    ):
        self.cache.reset()
        self.observed.clear()
        self.prefetch_enabled = prefetch

        self.stats = {
            "predictions": 0,
            "skipped": 0,
            "gpu_exec": 0,
            "cpu_exec": 0,
        }

        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to("cuda")

        torch.cuda.synchronize()
        start = time.perf_counter()
        token_timer = TokenTimer(start)

        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[token_timer],
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # TokenTimer runs after each newly generated token is appended.
        new_token_ids = result[
            0, inputs["input_ids"].shape[1]:
        ].tolist()
        timestamps = token_timer.timestamps
        assert len(timestamps) == len(new_token_ids), (
            "Token timestamp count does not match generated tokens"
        )
        intervals_ms = [
            (b - a) * 1000.0
            for a, b in zip(timestamps, timestamps[1:])
        ]
        ttft_ms = ((timestamps[0] - start) * 1000.0) if timestamps else None
        tpot_ms = (
            sum(intervals_ms) / len(intervals_ms)
            if intervals_ms else None
        )
        decode_tps = (
            1000.0 / tpot_ms
            if tpot_ms is not None and tpot_ms > 0 else None
        )

        assert (
            self.cache.hits + self.cache.misses
            == self.stats["gpu_exec"]
        )
        assert (
            self.stats["cpu_exec"]
            == 7 * self.stats["gpu_exec"]
        )
        assert len(self.cache.entries) <= self.capacity

        return {
            "new_token_ids": new_token_ids,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "decode_tokens_per_s": decode_tps,
            "token_intervals_ms": intervals_ms,
            "timing_method": "synchronized_stopping_criteria",
            "elapsed_s": elapsed,
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "demand_loads": self.cache.demand_loads,
            "prefetch_loads": self.cache.prefetch_loads,
            "useful_prefetches": self.cache.useful_prefetches,
            "wasted_prefetches": self.cache.wasted_prefetches,
            "evictions": self.cache.evictions,
            "predictions": self.stats["predictions"],
            "skipped": self.stats["skipped"],
            "gpu_exec": self.stats["gpu_exec"],
            "cpu_exec": self.stats["cpu_exec"],
            "staging_ms": self.cache.staging_ms,
        }

    def close(self):
        self.cache.copy_stream.synchronize()
        torch.cuda.synchronize()

        for hook in self._hooks:
            hook.remove()

        for mlp, original in zip(
            self.blocks, self._original_forwards
        ):
            mlp.experts.forward = original

        self._hooks.clear()
