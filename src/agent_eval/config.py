"""Load and validate evals.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_eval.errors import ConfigError

ADAPTER_TYPES = ("python", "langchain", "langgraph", "openai_agents", "http")
JUDGED_METRICS = {
    "task_completion",
    "step_efficiency",
    "plan_quality",
    "plan_adherence",
    "argument_correctness",
    "geval",
    "answer_correctness",
}
BUILTIN_METRICS = JUDGED_METRICS | {"tool_correctness"}


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["python", "langchain", "langgraph", "openai_agents", "http"]
    target: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 60
    kwargs: dict[str, Any] = Field(default_factory=dict)
    factory: bool = False
    cache_version: str = "1"

    @model_validator(mode="after")
    def _target_or_url(self) -> AdapterConfig:
        if self.type == "http":
            if not self.url:
                raise ValueError("http adapter requires url")
        elif not self.target:
            raise ValueError(f"{self.type} adapter requires target (module:attribute)")
        elif ":" not in self.target:
            raise ValueError("target must look like 'package.module:attribute'")
        return self


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "azure"] = "openai"
    model: str = "gpt-4.1-mini"
    samples: int = 1
    disagreement_spread: float = 0.25
    fail_on_disagreement: bool = False
    api_key_env: str | None = None
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_version: str = "2024-10-21"
    deployment: str | None = None

    @field_validator("samples")
    @classmethod
    def _samples(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("judge.samples must be between 1 and 10")
        return value

    @field_validator("disagreement_spread")
    @classmethod
    def _spread(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("judge.disagreement_spread must be between 0 and 1")
        return value


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_usd: float | None = None
    max_cases: int | None = None
    max_judge_calls: int | None = None
    assume_judge_call_usd: float = 0.002

    @field_validator("max_usd", "assume_judge_call_usd")
    @classmethod
    def _non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("budget amounts must be >= 0")
        return value

    @field_validator("max_cases", "max_judge_calls")
    @classmethod
    def _positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("budget caps must be >= 1")
        return value


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = 4
    cache: bool = True
    cache_dir: str = ".agent-eval/cache"
    runs_dir: str = "runs"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    regression_delta: float = 0.05

    @field_validator("concurrency")
    @classmethod
    def _concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("run.concurrency must be >= 1")
        return value


class MetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str | None = None
    threshold: float = 0.5
    task: str | None = None
    criteria: str | None = None
    evaluation_steps: list[str] | None = None
    evaluation_params: list[str] | None = None
    exact_match: bool = False
    consider_ordering: bool = False
    input_parameters: bool = False
    available_tools: list[str] = Field(default_factory=list)
    target: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    judged: bool | None = None

    @field_validator("threshold")
    @classmethod
    def _threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("metric threshold must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _kind(self) -> MetricConfig:
        kind = self.type or self.name
        if kind == "custom" or (
            self.type is None and self.name not in BUILTIN_METRICS and self.target
        ):
            kind = "custom"
        if self.type is None:
            self.type = kind
        if self.type == "custom" and not self.target:
            raise ValueError(f"custom metric {self.name!r} requires target")
        if self.type not in BUILTIN_METRICS | {"custom"}:
            raise ValueError(
                f"unknown metric type {self.type!r} on {self.name!r}. "
                f"Known types: {', '.join(sorted(BUILTIN_METRICS | {'custom'}))}"
            )
        if self.type in {"geval", "answer_correctness"} and not (
            self.criteria or self.evaluation_steps
        ):
            if self.type == "answer_correctness" or self.name == "answer_correctness":
                self.criteria = (
                    "Decide whether the actual output reaches the same logical conclusion "
                    "as the expected output. Wording may differ. Penalize a different "
                    "decision, amount, eligibility call, or next step."
                )
            else:
                raise ValueError(
                    f"geval metric {self.name!r} requires criteria or evaluation_steps"
                )
        return self

    def is_judged(self) -> bool:
        if self.judged is not None:
            return self.judged
        if self.type == "tool_correctness":
            return bool(self.available_tools)
        if self.type == "custom":
            return False
        return True


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    project: str
    adapter: AdapterConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    dataset: DatasetConfig
    run: RunConfig = Field(default_factory=RunConfig)
    metrics: list[MetricConfig]

    @model_validator(mode="after")
    def _metrics(self) -> EvalConfig:
        if self.version != 1:
            raise ValueError("only evals.yaml version 1 is supported")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        return self


def find_project_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "evals.yaml").exists():
            return parent
    return current


def resolve_from(config_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_config(path: Path) -> EvalConfig:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    try:
        return EvalConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
