
"""Verify that modular inference matches our reference."""

from moe_hybrid.inference import HybridMoE

PROMPT = "What causes ocean tides on Earth?"

EXPECTED_IDS = [
    187, 187, 48, 8705, 246, 1487, 403, 4269
]

engine = HybridMoE(
    capacity=32,
    predictor_method="nn",
)

try:
    off = engine.generate(
        PROMPT,
        max_new_tokens=8,
        prefetch=False,
    )

    on = engine.generate(
        PROMPT,
        max_new_tokens=8,
        prefetch=True,
    )

    print(
        "OFF hits / misses:",
        off["hits"], off["misses"],
    )
    print(
        "ON hits / misses:",
        on["hits"], on["misses"],
    )
    print(
        "ON prefetch loads:",
        on["prefetch_loads"],
    )
    print(
        "Native-reference IDs match:",
        off["new_token_ids"] == EXPECTED_IDS,
    )
    print(
        "ON/OFF token IDs match:",
        on["new_token_ids"] == off["new_token_ids"],
    )

    assert (off["hits"], off["misses"]) == (50, 190)

    assert (
        on["hits"],
        on["misses"],
        on["prefetch_loads"],
    ) == (62, 178, 15)

    assert (
        off["new_token_ids"]
        == on["new_token_ids"]
        == EXPECTED_IDS
    )

    print("Step 10G inference entry point PASSED")

finally:
    engine.close()
