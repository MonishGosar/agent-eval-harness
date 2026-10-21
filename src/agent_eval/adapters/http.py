"""Adapter for an HTTP agent. POST {input, turns, session_id, metadata}; read a JSON AgentResult."""

from __future__ import annotations

import os
import re

import httpx

from agent_eval.adapters.base import coerce_result
from agent_eval.config import AdapterConfig
from agent_eval.errors import AdapterError
from agent_eval.models import AgentRequest, AgentResult

_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise AdapterError(f"header placeholder ${{{name}}} is not set")
        return os.environ[name]

    return _ENV.sub(replace, value)


class HttpAdapter:
    def __init__(self, config: AdapterConfig) -> None:
        if not config.url:
            raise AdapterError("http adapter requires url")
        self._url = config.url
        self._timeout = config.timeout_seconds
        self._headers = {key: _expand(value) for key, value in config.headers.items()}

    def run(self, request: AgentRequest) -> AgentResult:
        payload = {
            "input": request.input,
            "turns": [turn.model_dump() for turn in request.turns],
            "session_id": request.session_id,
            "metadata": request.metadata,
            "case_id": request.case_id,
        }
        try:
            response = httpx.post(
                self._url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise AdapterError(f"HTTP agent call failed: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"HTTP agent returned non-JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise AdapterError("HTTP agent JSON body must be an object")
        return coerce_result(body)


def build(config: AdapterConfig) -> HttpAdapter:
    return HttpAdapter(config)
