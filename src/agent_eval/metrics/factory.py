"""Build fresh DeepEval metric instances from evals.yaml.

Each sample gets its own instance because DeepEval metrics store the score on self.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_eval.config import MetricConfig
from agent_eval.errors import ConfigError
from agent_eval.loading import load_callable

MetricFactory = Callable[[], Any]

_PARAMS = {
    "input": "INPUT",
    "actual_output": "ACTUAL_OUTPUT",
    "expected_output": "EXPECTED_OUTPUT",
    "context": "CONTEXT",
    "retrieval_context": "RETRIEVAL_CONTEXT",
    "tools_called": "TOOLS_CALLED",
    "expected_tools": "EXPECTED_TOOLS",
}


def build_factories(
    metrics: list[MetricConfig], judge: Any
) -> list[tuple[MetricConfig, MetricFactory]]:
    return [(metric, build_factory(metric, judge)) for metric in metrics]


def build_factory(metric: MetricConfig, judge: Any) -> MetricFactory:
    kind = metric.type or metric.name
    if kind == "task_completion":
        return lambda metric=metric: _task(metric, judge)
    if kind == "step_efficiency":
        return lambda metric=metric: _step(metric, judge)
    if kind == "plan_quality":
        return lambda metric=metric: _plan_quality(metric, judge)
    if kind == "plan_adherence":
        return lambda metric=metric: _plan_adherence(metric, judge)
    if kind == "tool_correctness":
        return lambda metric=metric: _tools(metric, judge)
    if kind == "argument_correctness":
        return lambda metric=metric: _arguments(metric, judge)
    if kind in {"geval", "answer_correctness"}:
        return lambda metric=metric: _geval(metric, judge)
    if kind == "custom":
        return lambda metric=metric: _custom(metric, judge)
    raise ConfigError(f"unknown metric type {kind!r}")


def _common(judge: Any) -> dict[str, Any]:
    return {
        "model": judge,
        "async_mode": False,
        "verbose_mode": False,
        "include_reason": True,
    }


def _task(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import TaskCompletionMetric

    return TaskCompletionMetric(threshold=metric.threshold, task=metric.task, **_common(judge))


def _step(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import StepEfficiencyMetric

    return StepEfficiencyMetric(threshold=metric.threshold, **_common(judge))


def _plan_quality(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import PlanQualityMetric

    return PlanQualityMetric(threshold=metric.threshold, **_common(judge))


def _plan_adherence(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import PlanAdherenceMetric

    return PlanAdherenceMetric(threshold=metric.threshold, **_common(judge))


def _arguments(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import ArgumentCorrectnessMetric

    return ArgumentCorrectnessMetric(threshold=metric.threshold, **_common(judge))


def _tools(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import ToolCall, ToolCallParams

    params: list[Any] = []
    if metric.input_parameters:
        params.append(ToolCallParams.INPUT_PARAMETERS)
    available = [ToolCall(name=name) for name in metric.available_tools] or None
    return ToolCorrectnessMetric(
        threshold=metric.threshold,
        evaluation_params=params,
        should_exact_match=metric.exact_match,
        should_consider_ordering=metric.consider_ordering,
        available_tools=available,
        **_common(judge),
    )


def _geval(metric: MetricConfig, judge: Any) -> Any:
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    names = metric.evaluation_params or ["input", "actual_output", "expected_output"]
    try:
        params = [getattr(SingleTurnParams, _PARAMS[name]) for name in names]
    except KeyError as exc:
        raise ConfigError(
            f"unknown evaluation param {exc.args[0]!r}. Known: {', '.join(sorted(_PARAMS))}"
        ) from exc
    return GEval(
        name=metric.name,
        criteria=metric.criteria,
        evaluation_steps=metric.evaluation_steps,
        evaluation_params=params,
        threshold=metric.threshold,
        model=judge,
        async_mode=False,
        verbose_mode=False,
    )


def _custom(metric: MetricConfig, judge: Any) -> Any:
    if not metric.target:
        raise ConfigError(f"custom metric {metric.name!r} requires target")
    builder = load_callable(metric.target)
    return builder(threshold=metric.threshold, model=judge, **metric.params)
