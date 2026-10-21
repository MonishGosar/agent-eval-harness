"""Construct the DeepEval judge from evals.yaml. Missing credentials raise; scores are never invented."""

from __future__ import annotations

import os
from typing import Any

from agent_eval.config import JudgeConfig
from agent_eval.errors import ConfigError


def build_judge(config: JudgeConfig) -> Any:
    if config.provider == "openai":
        return _openai(config)
    if config.provider == "azure":
        return _azure(config)
    raise ConfigError(f"unknown judge provider {config.provider!r}")


def _openai(config: JudgeConfig) -> Any:
    env_name = config.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(env_name)
    if not api_key:
        raise ConfigError(
            f"OpenAI judge requires {env_name}. "
            "agent-eval does not invent scores when no key is configured."
        )
    from deepeval.models import OpenAIModel

    return OpenAIModel(model=config.model, api_key=api_key)


def _azure(config: JudgeConfig) -> Any:
    env_name = config.api_key_env or "AZURE_OPENAI_API_KEY"
    api_key = os.environ.get(env_name)
    endpoint = os.environ.get(config.endpoint_env)
    if not api_key:
        raise ConfigError(
            f"Azure OpenAI judge requires {env_name}. "
            "agent-eval does not invent scores when no key is configured."
        )
    if not endpoint:
        raise ConfigError(f"Azure OpenAI judge requires {config.endpoint_env}")
    deployment = config.deployment or config.model
    from deepeval.models import AzureOpenAIModel

    return AzureOpenAIModel(
        model=config.model,
        api_key=api_key,
        base_url=endpoint,
        api_version=config.api_version,
        deployment_name=deployment,
    )
