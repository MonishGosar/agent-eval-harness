"""Rough token prices used only when a provider does not report cost.

Figures are USD per million tokens and follow DeepEval's published OpenAI table.
Reports mark any total that used this table, or the per-call assumption, as estimated.
"""

from __future__ import annotations

# (input USD / 1M, output USD / 1M)
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT = _PRICES["gpt-4.1-mini"]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = _PRICES.get(model, _DEFAULT)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
