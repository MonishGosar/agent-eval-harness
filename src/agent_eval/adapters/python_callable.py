"""Adapter for a plain Python callable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_eval.adapters.base import call_python
from agent_eval.config import AdapterConfig
from agent_eval.loading import load_callable
from agent_eval.models import AgentRequest, AgentResult


class PythonCallableAdapter:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def run(self, request: AgentRequest) -> AgentResult:
        return call_python(self._fn, request)


def build(config: AdapterConfig) -> PythonCallableAdapter:
    assert config.target is not None
    loaded = load_callable(config.target)
    if config.factory:
        built = loaded(**config.kwargs)
        if not callable(built):
            raise TypeError(f"{config.target} factory did not return a callable")
        return PythonCallableAdapter(built)
    return PythonCallableAdapter(loaded)
