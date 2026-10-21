"""Build the adapter named by evals.yaml."""

from __future__ import annotations

from agent_eval.adapters.base import AgentAdapter
from agent_eval.adapters.http import build as build_http
from agent_eval.adapters.langchain import build as build_langchain
from agent_eval.adapters.openai_agents import build as build_openai_agents
from agent_eval.adapters.python_callable import build as build_python
from agent_eval.config import AdapterConfig
from agent_eval.errors import ConfigError


def build_adapter(config: AdapterConfig) -> AgentAdapter:
    if config.type == "python":
        return build_python(config)
    if config.type in {"langchain", "langgraph"}:
        return build_langchain(config)
    if config.type == "openai_agents":
        return build_openai_agents(config)
    if config.type == "http":
        return build_http(config)
    raise ConfigError(f"unknown adapter type {config.type!r}")
