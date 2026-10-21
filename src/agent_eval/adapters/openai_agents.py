"""Adapter for the OpenAI Agents SDK. The agent is built once; Runner.run_sync runs each case once."""

from __future__ import annotations

import json
import time
from typing import Any

from agent_eval.config import AdapterConfig
from agent_eval.errors import AdapterError
from agent_eval.loading import load_callable
from agent_eval.models import (
    AgentRequest,
    AgentResult,
    ToolCallRecord,
    Usage,
    render_structured_output,
)


def build(config: AdapterConfig) -> OpenAIAgentsAdapter:
    _require_sdk()
    assert config.target is not None
    factory = load_callable(config.target)
    agent = factory(**config.kwargs)
    return OpenAIAgentsAdapter(agent)


def _require_sdk() -> None:
    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise AdapterError(
            "OpenAI Agents adapter requires the openai-agents extra: "
            "pip install 'agent-eval-harness[openai-agents]'"
        ) from exc


class OpenAIAgentsAdapter:
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def run(self, request: AgentRequest) -> AgentResult:
        _require_sdk()
        from agents import Runner

        payload: str | list[dict[str, str]]
        if request.turns:
            payload = [{"role": turn.role, "content": turn.content} for turn in request.turns]
            payload.append({"role": "user", "content": request.input})
        else:
            payload = request.input
        started = time.perf_counter()
        result = Runner.run_sync(self._agent, payload)
        latency_ms = (time.perf_counter() - started) * 1000
        extracted = extract_openai_result(result)
        return extracted.model_copy(update={"latency_ms": latency_ms}).normalized()


def extract_openai_result(result: Any) -> AgentResult:
    """Read final output, tool calls, and usage from a RunResult-like object."""
    calls: dict[str, ToolCallRecord] = {}
    order: list[str] = []
    plans: list[str] = []
    for item in getattr(result, "new_items", None) or []:
        kind = getattr(item, "type", "")
        raw = getattr(item, "raw_item", None)
        if kind == "tool_call_item":
            name = _tool_name(item, raw)
            arguments = _tool_args(raw)
            identifier = _call_id(raw) or f"{name}-{len(order)}"
            calls[identifier] = ToolCallRecord(name=name, arguments=arguments)
            order.append(identifier)
        elif kind == "tool_call_output_item":
            identifier = _call_id(raw) or _call_id(getattr(item, "raw_item", None))
            output = getattr(item, "output", None)
            output = _maybe_json(output)
            if identifier and identifier in calls:
                calls[identifier] = calls[identifier].model_copy(update={"output": output})
            elif order:
                last = order[-1]
                if calls[last].output is None:
                    calls[last] = calls[last].model_copy(update={"output": output})
        elif kind in {"message_output_item", "reasoning_item"}:
            text = _message_text(raw)
            for line in text.splitlines():
                if line.strip().lower().startswith("plan:"):
                    plans.append(line.strip())
    output = render_structured_output(getattr(result, "final_output", ""))
    return AgentResult(
        output=output,
        tool_calls=[calls[key] for key in order],
        plan=plans,
        usage=_usage(result),
    )


def _tool_name(item: Any, raw: Any) -> str:
    named = getattr(item, "tool_name", None)
    if named:
        return str(named)
    if isinstance(raw, dict):
        return str(raw.get("name") or raw.get("tool_name") or "tool")
    return str(getattr(raw, "name", None) or "tool")


def _tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        value = raw.get("arguments", raw.get("args", {}))
    else:
        value = getattr(raw, "arguments", None)
        if value is None:
            value = getattr(raw, "args", {})
    if isinstance(value, str):
        return _maybe_dict(value)
    if isinstance(value, dict):
        return value
    return {} if value in (None, "") else {"value": value}


def _call_id(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(raw.get("call_id") or raw.get("id") or "")
    return str(getattr(raw, "call_id", None) or getattr(raw, "id", None) or "")


def _message_text(raw: Any) -> str:
    if raw is None:
        return ""
    content = raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    text = getattr(raw, "text", None)
    return str(text or "")


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _maybe_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _usage(result: Any) -> Usage:
    prompt = 0
    completion = 0
    for response in getattr(result, "raw_responses", None) or []:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        prompt += int(getattr(usage, "input_tokens", 0) or 0)
        completion += int(getattr(usage, "output_tokens", 0) or 0)
    return Usage(prompt_tokens=prompt, completion_tokens=completion).settled()
