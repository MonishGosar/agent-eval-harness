"""Adapter for LangChain runnables and LangGraph agents.

The runnable is invoked once. Tool calls are read from the returned messages (and from
callbacks when a legacy agent only reports tools that way).
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent_eval.adapters.base import coerce_result
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


def build(config: AdapterConfig) -> LangChainAdapter:
    _require_langchain()
    assert config.target is not None
    factory = load_callable(config.target)
    runnable = factory(**config.kwargs)
    return LangChainAdapter(runnable)


def _require_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise AdapterError(
            "LangChain adapter requires the langchain extra: "
            "pip install 'agent-eval-harness[langchain]'"
        ) from exc


class _ToolTap:
    """Callback used when the runnable does not return a message list."""

    def __init__(self) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        class Handler(BaseCallbackHandler):
            def __init__(self) -> None:
                super().__init__()
                self.calls: dict[str, ToolCallRecord] = {}
                self.order: list[str] = []
                self.usage = Usage()
                self.texts: list[str] = []

            def on_llm_end(self, response: Any, **kwargs: Any) -> None:
                self.usage = _add_usage(self.usage, _usage_from_llm_result(response))
                try:
                    generation = response.generations[0][0]
                except (AttributeError, IndexError, TypeError):
                    return
                message = getattr(generation, "message", None)
                text = _message_text(
                    getattr(message, "content", None) or getattr(generation, "text", "")
                )
                if text:
                    self.texts.append(text)

            def on_tool_start(
                self,
                serialized: dict[str, Any],
                input_str: Any,
                *,
                run_id: Any,
                **kwargs: Any,
            ) -> None:
                key = str(run_id)
                name = str(serialized.get("name") or kwargs.get("name") or "tool")
                self.calls[key] = ToolCallRecord(name=name, arguments=_coerce_args(input_str))
                self.order.append(key)

            def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
                key = str(run_id)
                record = self.calls.get(key)
                if record is None:
                    return
                self.calls[key] = record.model_copy(update={"output": _tool_output(output)})

        self.handler = Handler()


def _invoke(runnable: Any, messages: list[Any], text: str, handler: Any) -> Any:
    """Prefer the LangGraph messages payload. Fall back when the runnable wants {input: text}."""
    try:
        return runnable.invoke({"messages": messages}, config={"callbacks": [handler]})
    except TypeError:
        return runnable.invoke({"input": text}, config={"callbacks": [handler]})


class LangChainAdapter:
    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    def run(self, request: AgentRequest) -> AgentResult:
        _require_langchain()
        from langchain_core.messages import AIMessage, HumanMessage

        messages: list[Any] = []
        for turn in request.turns:
            if turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))
            else:
                messages.append(HumanMessage(content=turn.content))
        messages.append(HumanMessage(content=request.input))
        tap = _ToolTap()
        started = time.perf_counter()
        result = _invoke(self._runnable, messages, request.input, tap.handler)
        latency_ms = (time.perf_counter() - started) * 1000
        extracted = extract_langchain_result(result)
        if not extracted.tool_calls and tap.handler.order:
            extracted = extracted.model_copy(
                update={
                    "tool_calls": [tap.handler.calls[key] for key in tap.handler.order],
                }
            )
        if not extracted.plan:
            extracted = extracted.model_copy(update={"plan": _plans_from_texts(tap.handler.texts)})
        usage = extracted.usage
        if usage.total_tokens == 0 and tap.handler.usage.total_tokens:
            usage = tap.handler.usage
        return extracted.model_copy(update={"latency_ms": latency_ms, "usage": usage}).normalized()


def extract_langchain_result(result: Any) -> AgentResult:
    """Pure extraction so tests can pass a message list without a live model."""
    if isinstance(result, AgentResult):
        return result.normalized()
    if isinstance(result, str):
        return AgentResult(output=result)
    if not isinstance(result, dict):
        return coerce_result(result)

    messages = result.get("messages")
    structured = result.get("structured_response")
    if messages:
        return _from_messages(messages, structured)
    if "output" in result and not isinstance(result.get("output"), list):
        output = result.get("output")
        text = render_structured_output(structured if structured is not None else output)
        return AgentResult(
            output=text, plan=_plans_from_texts([text] if isinstance(text, str) else [])
        )
    if structured is not None:
        return AgentResult(output=render_structured_output(structured))
    return coerce_result(result)


def _from_messages(messages: list[Any], structured: Any) -> AgentResult:
    from langchain_core.messages import AIMessage, ToolMessage

    calls: dict[str, ToolCallRecord] = {}
    order: list[str] = []
    texts: list[str] = []
    usage = Usage()
    for message in messages:
        usage = _add_usage(usage, _usage_from_message(message))
        if isinstance(message, AIMessage) or message.__class__.__name__ == "AIMessage":
            text = _message_text(message.content)
            if text:
                texts.append(text)
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = str(tool_call.get("name") or "")
                identifier = str(tool_call.get("id") or f"{name}-{len(order)}")
                record = ToolCallRecord(
                    name=name,
                    arguments=_coerce_args(tool_call.get("args")),
                    reasoning=text or None,
                )
                calls[identifier] = record
                order.append(identifier)
        elif isinstance(message, ToolMessage) or message.__class__.__name__ == "ToolMessage":
            identifier = str(getattr(message, "tool_call_id", "") or "")
            output = _tool_output(message.content)
            if identifier in calls:
                calls[identifier] = calls[identifier].model_copy(update={"output": output})
            else:
                name = str(getattr(message, "name", "") or "tool")
                key = identifier or f"{name}-{len(order)}"
                calls[key] = ToolCallRecord(name=name, arguments={}, output=output)
                if key not in order:
                    order.append(key)
    output = render_structured_output(structured) if structured is not None else ""
    if not output:
        output = texts[-1] if texts else ""
    return AgentResult(
        output=output,
        tool_calls=[calls[key] for key in order if key in calls],
        plan=_plans_from_texts(texts),
        usage=usage.settled(),
    )


def _plans_from_texts(texts: list[str]) -> list[str]:
    plans: list[str] = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("plan:"):
                plans.append(stripped)
    return plans


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _coerce_args(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def _tool_output(value: Any) -> Any:
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _usage_from_message(message: Any) -> Usage:
    metadata = getattr(message, "usage_metadata", None) or {}
    if not isinstance(metadata, dict):
        return Usage()
    return Usage(
        prompt_tokens=int(metadata.get("input_tokens") or 0),
        completion_tokens=int(metadata.get("output_tokens") or 0),
    )


def _usage_from_llm_result(response: Any) -> Usage:
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = {}
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if not isinstance(token_usage, dict):
        return Usage()
    return Usage(
        prompt_tokens=int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0),
        completion_tokens=int(
            token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
        ),
    )


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        cost_usd=left.cost_usd if right.cost_usd is None else (left.cost_usd or 0) + right.cost_usd,
    )
