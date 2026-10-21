"""Shared records for requests, traces, and persisted run results."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def jsonable(value: Any) -> Any:
    """Return a JSON-serializable copy of a tool output or model object."""
    try:
        json.dumps(value)
    except TypeError:
        if hasattr(value, "model_dump"):
            return jsonable(value.model_dump())
        return str(value)
    else:
        return value


def render_structured_output(value: Any) -> str:
    """Render a final agent payload. Structured answers keep both fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    data: Any
    if hasattr(value, "model_dump"):
        data = value.model_dump()
    elif isinstance(value, dict):
        data = value
    else:
        return str(value)
    if isinstance(data, dict) and "answer" in data:
        answer = str(data.get("answer") or "").strip()
        conversation = str(data.get("conversation") or "").strip()
        if conversation:
            return f"{answer}\n\n{conversation}"
        return answer
    return json.dumps(data, default=str)


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None

    def settled(self) -> Usage:
        total = self.total_tokens or (self.prompt_tokens + self.completion_tokens)
        return self.model_copy(update={"total_tokens": total})


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    reasoning: str | None = None

    def normalized(self) -> ToolCallRecord:
        args = self.arguments if isinstance(self.arguments, dict) else {"value": self.arguments}
        return self.model_copy(
            update={"arguments": jsonable(args), "output": jsonable(self.output)}
        )


class AgentRequest(BaseModel):
    """One evaluation input handed to an adapter. Turns are prior context, not the current ask."""

    case_id: str
    input: str
    turns: list[Turn] = Field(default_factory=list)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What a single adapter invocation produced. Metrics are scored from this, not from a re-run."""

    model_config = ConfigDict(extra="ignore")

    output: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0
    error: str | None = None

    def normalized(self) -> AgentResult:
        return self.model_copy(
            update={
                "tool_calls": [call.normalized() for call in self.tool_calls],
                "usage": self.usage.settled(),
                "plan": [step.strip() for step in self.plan if step and step.strip()],
            }
        )


class ExpectedTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] | None = None


class Golden(BaseModel):
    """One dataset row. expected_tools omitted means tool correctness is not scored for the row."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expected_output: str
    expected_tools: list[ExpectedTool] | None = None
    turns: list[Turn] = Field(default_factory=list)
    task: str | None = None
    context: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def rendered_input(self) -> str:
        if not self.turns:
            return self.input
        lines = [f"{turn.role}: {turn.content}" for turn in self.turns]
        lines.append(f"user: {self.input}")
        return "\n".join(lines)


class JudgeSample(BaseModel):
    score: float
    reason: str
    passed: bool


class MetricResult(BaseModel):
    name: str
    score: float | None = None
    threshold: float
    passed: bool
    reason: str = ""
    judged: bool = False
    disagreement: bool = False
    spread: float = 0.0
    samples: list[JudgeSample] = Field(default_factory=list)
    skipped: bool = False
    error: str | None = None


class CaseResult(BaseModel):
    id: str
    status: Literal["passed", "failed", "error", "skipped"]
    input: str
    rendered_input: str
    expected_output: str
    actual_output: str = ""
    tags: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    plan_captured: bool = False
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    expected_tools: list[ExpectedTool] | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0
    agent_cached: bool = False
    error: str | None = None
    skip_reason: str | None = None


class MetricSummary(BaseModel):
    name: str
    threshold: float
    average: float | None = None
    pass_rate: float | None = None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    disagreements: int = 0


class RunSummary(BaseModel):
    cases: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    pass_rate: float = 0.0
    agent_runs: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False
    disagreements: int = 0
    thresholds_failed: bool = False
    budget_exceeded: bool = False


class RunRecord(BaseModel):
    schema_version: int = 1
    run_id: str
    project: str
    created_at: str
    harness_version: str
    config_path: str
    config_hash: str
    dataset_path: str
    dataset_hash: str
    judge_model: str
    judge_provider: str
    adapter_type: str
    status: Literal["passed", "failed", "budget_exceeded"]
    summary: RunSummary
    metrics: list[MetricSummary]
    cases: list[CaseResult]


class MetricDelta(BaseModel):
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    regressions: int = 0
    improvements: int = 0


class CaseDelta(BaseModel):
    case_id: str
    metric: str
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    baseline_passed: bool | None
    candidate_passed: bool | None
    kind: Literal["regression", "improvement"]
    reason: str


class CompareReport(BaseModel):
    schema_version: int = 1
    baseline_run: str
    candidate_run: str
    regression_delta: float
    regressions: list[CaseDelta] = Field(default_factory=list)
    improvements: list[CaseDelta] = Field(default_factory=list)
    metrics: list[MetricDelta] = Field(default_factory=list)
    missing_in_candidate: list[str] = Field(default_factory=list)
    missing_in_baseline: list[str] = Field(default_factory=list)
