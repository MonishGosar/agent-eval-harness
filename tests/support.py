"""Tiny agent used by harness tests. Not a live model."""

from __future__ import annotations

import threading

calls = {"n": 0}
_lock = threading.Lock()


def reset() -> None:
    with _lock:
        calls["n"] = 0


def run_agent(request: object) -> dict[str, object]:
    with _lock:
        calls["n"] += 1
    return {
        "output": "Order 17 was delivered.",
        "plan": ["Plan: look up order 17"],
        "tool_calls": [
            {
                "name": "lookup_order",
                "arguments": {"order_id": "17"},
                "output": {"status": "delivered"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost_usd": 0.01},
    }
