"""In-process judge for tests. It never calls a provider."""

from __future__ import annotations

import threading
from typing import Any

from deepeval.models import DeepEvalBaseLLM


class ScriptedJudge(DeepEvalBaseLLM):
    """Return schema objects. G-Eval scores are on a 0-10 scale before normalization."""

    def __init__(self, scores: list[float] | None = None, *, advance: bool = False) -> None:
        self._scores = scores or [0.9]
        self._advance = advance
        self._index = 0
        self.calls = 0
        self._lock = threading.Lock()
        super().__init__(model="scripted-judge")

    def load_model(self) -> ScriptedJudge:
        return self

    def get_model_name(self) -> str:
        return "scripted-judge"

    async def a_generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        schema = kwargs.get("schema")
        with self._lock:
            self.calls += 1
            score = self._scores[self._index % len(self._scores)]
            if self._advance and schema is not None and _is_score_schema(schema):
                self._index += 1
        if schema is None:
            return '{"score": 0.9, "reason": "scripted"}'
        return _instance(schema, score)

    def supports_log_probs(self) -> bool:
        return False


def _is_score_schema(schema: type) -> bool:
    return schema.__name__ in {
        "ReasonScore",
        "TaskCompletionVerdict",
        "EfficiencyVerdict",
        "PlanQualityScore",
        "PlanAdherenceScore",
        "Verdicts",
    }


def _instance(schema: type, score: float) -> Any:
    name = schema.__name__
    if name == "Steps":
        return schema(steps=["Compare the decision, the amount, and the next step."])
    if name == "ReasonScore":
        return schema(reason="same logical conclusion", score=score * 10)
    if name == "TaskAndOutcome":
        return schema(task="resolve the request", outcome="the agent answered")
    if name == "TaskCompletionVerdict":
        return schema(verdict=score, reason="the task was handled")
    if name == "Task":
        return schema(task="resolve the request")
    if name == "EfficiencyVerdict":
        return schema(score=score, reason="no wasted steps")
    if name == "AgentPlan":
        return schema(plan=["look up the record", "answer from the record"])
    if name == "PlanQualityScore":
        return schema(score=score, reason="the plan is usable")
    if name == "PlanAdherenceScore":
        return schema(score=score, reason="the trace follows the plan")
    if name == "Verdicts":
        from deepeval.metrics.argument_correctness.schema import ArgumentCorrectnessVerdict
        from deepeval.metrics.base_metric import Verdict

        verdict = Verdict.YES if score >= 0.5 else Verdict.NO
        return schema(verdicts=[ArgumentCorrectnessVerdict(verdict=verdict, reason="checked")])
    if name == "ArgumentCorrectnessScoreReason":
        return schema(
            reason="arguments match the request" if score >= 0.5 else "arguments do not match"
        )
    if name == "ToolSelectionScore":
        return schema(score=score, reason="tool choice is fine")
    raise AssertionError(f"scripted judge has no fixture for schema {name}")
