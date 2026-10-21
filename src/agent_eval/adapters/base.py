"""Adapter interface. run() is invoked once per case; metrics never call the agent themselves."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from agent_eval.models import AgentRequest, AgentResult


class AgentAdapter(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        """Execute the agent for one case and return text plus the tool-call trace."""


def coerce_result(value: Any) -> AgentResult:
    if isinstance(value, AgentResult):
        return value.normalized()
    if isinstance(value, str):
        return AgentResult(output=value)
    if isinstance(value, dict):
        return AgentResult.model_validate(value).normalized()
    if hasattr(value, "output"):
        return AgentResult.model_validate(value, from_attributes=True).normalized()
    raise TypeError(
        "agent returned an unsupported value. Return str, dict, or AgentResult "
        f"(got {type(value).__name__})"
    )


def call_python(fn: Any, request: AgentRequest) -> AgentResult:
    """Call a user function. A single str parameter receives the user text; otherwise the request."""
    signature = inspect.signature(fn)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(parameters) == 1 and parameters[0].annotation is str:
        value = fn(request.input)
    else:
        value = fn(request)
    return coerce_result(value)
