"""Shared GPU expert cache with MRS eviction."""

import os
import time
import torch

PROFILE_TRANSFERS = os.getenv("MOE_PROFILE_TRANSFERS", "0") == "1"


class SharedCache:
    def __init__(self, blocks, capacity=4, alpha=0.5):
        self.blocks = blocks
        self.capacity = capacity
        self.alpha = alpha
        self.copy_stream = torch.cuda.Stream()
        self.priorities = {}
        self.entries = {}
        self.free_slots = list(range(self.capacity))

        # Reusable, bounded pinned-memory staging buffers: ~12 MiB per expert slot.
        sample = self.blocks[0].experts
        gate_shape = tuple(sample.gate_up_proj[0].shape)
        down_shape = tuple(sample.down_proj[0].shape)
        self.slots = [
            (
                torch.empty(gate_shape, dtype=torch.bfloat16, pin_memory=True),
                torch.empty(down_shape, dtype=torch.bfloat16, pin_memory=True),
            )
            for _ in range(self.capacity)
        ]
        self.reset()

    def reset(self):
        self.copy_stream.synchronize()
        torch.cuda.synchronize()
        self.entries.clear()
        self.priorities.clear()
        self.free_slots = list(range(self.capacity))
        self.hits = self.misses = self.evictions = 0
        self.prefetch_loads = self.useful_prefetches = 0
        self.wasted_prefetches = 0
        self.demand_loads = 0
        self.staging_ms = 0.0
        self.weight_h2d_bytes = 0
        self.demand_h2d_bytes = 0
        self.prefetch_h2d_bytes = 0
        self.eviction_wait_ms = 0.0
        self.host_enqueue_ms = 0.0
        self.copy_events = []
        self.load_wait_events = []

    def update_mrs(self, layer_id, logits):
        probabilities = torch.softmax(logits.float(), dim=-1).mean(0).cpu()
        chosen = probabilities.topk(16).indices
        scores = torch.zeros_like(probabilities)
        scores[chosen] = probabilities[chosen]
        for expert_id, score in enumerate(scores.tolist()):
            key = (layer_id, expert_id)
            self.priorities[key] = (
                self.alpha * score
                + (1 - self.alpha) * self.priorities.get(key, 0.0)
            )

    def insert(self, key, source):
        if key in self.entries:
            return self.entries[key]

        if not self.free_slots:
            # Soft protection: evict an already-used entry first.
            candidates = [
                k for k, entry in self.entries.items()
                if entry["source"] != "prefetch"
            ]
            if not candidates:
                candidates = list(self.entries)
            victim = min(candidates, key=lambda k: self.priorities.get(k, 0))
            old = self.entries.pop(victim)

            # Do not overwrite pinned staging memory while DMA is in flight.
            evict_start = time.perf_counter() if PROFILE_TRANSFERS else None
            old["ready"].synchronize()
            if PROFILE_TRANSFERS:
                self.eviction_wait_ms += (time.perf_counter() - evict_start) * 1000
            if old["source"] == "prefetch":
                self.wasted_prefetches += 1
            self.free_slots.append(old["slot"])
            self.evictions += 1

        slot = self.free_slots.pop()
        gate_pin, down_pin = self.slots[slot]
        layer_id, expert_id = key
        experts = self.blocks[layer_id].experts

        start = time.perf_counter()
        gate_pin.copy_(experts.gate_up_proj[expert_id])
        down_pin.copy_(experts.down_proj[expert_id])
        self.staging_ms += (time.perf_counter() - start) * 1000

        # Track actual expert weight H2D bytes (regardless of whether profiling is enabled).
        bytes_this_load = (
            gate_pin.numel() * gate_pin.element_size()
            + down_pin.numel() * down_pin.element_size()
        )
        self.weight_h2d_bytes += bytes_this_load
        if source == "prefetch":
            self.prefetch_h2d_bytes += bytes_this_load
        else:
            self.demand_h2d_bytes += bytes_this_load

        enqueue_start = time.perf_counter() if PROFILE_TRANSFERS else None
        with torch.cuda.stream(self.copy_stream):
            if PROFILE_TRANSFERS:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record()
            gate_gpu = gate_pin.to("cuda", non_blocking=True)
            down_gpu = down_pin.to("cuda", non_blocking=True)
            if PROFILE_TRANSFERS:
                copy_end.record()
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        if PROFILE_TRANSFERS:
            self.host_enqueue_ms += (time.perf_counter() - enqueue_start) * 1000
            self.copy_events.append((copy_start, copy_end, source))

        entry = {
            "weights": (gate_gpu, down_gpu),
            "ready": ready,
            "source": source,
            "slot": slot,
        }
        self.entries[key] = entry
        if source == "prefetch":
            self.prefetch_loads += 1
        else:
            self.demand_loads += 1
        assert len(self.entries) <= self.capacity
        return entry

    def prefetch(self, key):
        if key not in self.entries:
            self.insert(key, "prefetch")

    def demand(self, key):
        if key in self.entries:
            self.hits += 1
            entry = self.entries[key]
            if entry["source"] == "prefetch":
                self.useful_prefetches += 1
        else:
            self.misses += 1
            entry = self.insert(key, "demand")

        # A GPU stream dependency, not a device-wide host synchronization.
        stream = torch.cuda.current_stream()
        prior_source = entry["source"]
        if PROFILE_TRANSFERS and prior_source != "used":
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_end = torch.cuda.Event(enable_timing=True)
            wait_start.record(stream)
        stream.wait_event(entry["ready"])
        if PROFILE_TRANSFERS and prior_source != "used":
            wait_end.record(stream)
            self.load_wait_events.append((wait_start, wait_end, prior_source))
        for weight in entry["weights"]:
            weight.record_stream(stream)  # Cross-stream allocator safety.
        entry["source"] = "used"
        return entry["weights"]
