"""Score every configured metric from one captured trace."""

from __future__ import annotations

from typing import Any

from agent_eval.config import MetricConfig
from agent_eval.metrics.factory import MetricFactory
from agent_eval.models import Golden, JudgeSample, MetricResult
from agent_eval.pricing import estimate_cost
from agent_eval.trace import build_trace


def to_test_case(case: Golden, result: Any, task: str | None) -> Any:
    from deepeval.test_case import LLMTestCase, ToolCall

    tools_called = [
        ToolCall(
            name=call.name,
            input_parameters=dict(call.arguments),
            output=call.output,
            reasoning=call.reasoning,
        )
        for call in result.tool_calls
    ]
    expected = None
    if case.expected_tools is not None:
        expected = [
            ToolCall(
                name=tool.name,
                input_parameters=dict(tool.arguments) if tool.arguments is not None else {},
            )
            for tool in case.expected_tools
        ]
    test_case = LLMTestCase(
        input=case.rendered_input(),
        actual_output=result.output,
        expected_output=case.expected_output,
        tools_called=tools_called,
        expected_tools=expected,
        context=list(case.context) or None,
        name=case.id,
    )
    trace = build_trace(case, result)
    fields = getattr(type(test_case), "model_fields", {})
    if "trace_dict" in fields:
        test_case.trace_dict = trace
    # deepeval 4.x still reads the private attribute. Set both so either release works.
    test_case._trace_dict = trace
    if task and hasattr(test_case, "_trace_dict"):
        test_case._trace_dict = {**trace, "task": task}
        if "trace_dict" in fields:
            test_case.trace_dict = test_case._trace_dict
    return test_case


def metric_applies(metric: MetricConfig, case: Golden) -> bool:
    if (metric.type or metric.name) == "tool_correctness":
        return case.expected_tools is not None
    return True


def score_metric(
    metric: MetricConfig,
    factory: MetricFactory,
    test_case: Any,
    *,
    samples: int,
    spread: float,
    judge_model: str,
    assume_call_usd: float,
    cache_get: Any | None = None,
    cache_put: Any | None = None,
    cache_key_for: Any | None = None,
) -> tuple[MetricResult, float, bool, int]:
    """Return the metric result, added cost, whether that cost was estimated, and judge calls."""
    if not metric.is_judged():
        sample_count = 1
    else:
        sample_count = samples
    drawn: list[JudgeSample] = []
    cost = 0.0
    estimated = False
    judge_calls = 0
    errors: list[str] = []
    for index in range(sample_count):
        cached = None
        key = None
        if cache_key_for is not None:
            key = cache_key_for(index)
            cached = cache_get(key) if cache_get else None
        if cached is not None:
            drawn.append(JudgeSample.model_validate(cached))
            continue
        sample, sample_cost, sample_estimated, calls, error = _one(
            factory,
            test_case,
            judge_model=judge_model,
            assume_call_usd=assume_call_usd,
            threshold=metric.threshold,
        )
        cost += sample_cost
        estimated = estimated or sample_estimated
        judge_calls += calls
        if error:
            errors.append(error)
        drawn.append(sample)
        if cache_put and key and error is None:
            cache_put(key, sample.model_dump(mode="json"))
    if not drawn:
        return (
            MetricResult(
                name=metric.name,
                threshold=metric.threshold,
                passed=False,
                error="no samples",
                judged=metric.is_judged(),
            ),
            cost,
            estimated,
            judge_calls,
        )
    scores = [sample.score for sample in drawn]
    aggregate = _median(scores)
    spread_value = max(scores) - min(scores) if len(scores) > 1 else 0.0
    votes = {sample.passed for sample in drawn}
    disagreement = len(drawn) > 1 and (spread_value > spread or len(votes) > 1)
    closest = min(drawn, key=lambda sample: abs(sample.score - aggregate))
    passed = aggregate >= metric.threshold and not errors
    reason = closest.reason
    if errors:
        reason = errors[0]
        passed = False
    result = MetricResult(
        name=metric.name,
        score=round(aggregate, 4),
        threshold=metric.threshold,
        passed=passed,
        reason=reason,
        judged=metric.is_judged(),
        disagreement=disagreement,
        spread=round(spread_value, 4),
        samples=drawn,
        error=errors[0] if errors else None,
    )
    return result, cost, estimated, judge_calls


def _one(
    factory: MetricFactory,
    test_case: Any,
    *,
    judge_model: str,
    assume_call_usd: float,
    threshold: float,
) -> tuple[JudgeSample, float, bool, int, str | None]:
    try:
        metric = factory()
        before = _read_calls(getattr(metric, "model", None))
        try:
            metric.measure(test_case, _show_indicator=False)
        except TypeError:
            metric.measure(test_case)
    except Exception as exc:  # metric failures stay on the case instead of aborting the run
        return (
            JudgeSample(score=0.0, reason=str(exc), passed=False),
            0.0,
            False,
            0,
            f"{type(exc).__name__}: {exc}",
        )
    score = getattr(metric, "score", None)
    if score is None:
        return (
            JudgeSample(score=0.0, reason="metric returned no score", passed=False),
            0.0,
            False,
            0,
            "metric returned no score",
        )
    score_f = float(score)
    reason = str(getattr(metric, "reason", "") or "")
    passed = bool(score_f >= threshold)
    after = _read_calls(getattr(metric, "model", None))
    calls = max(after - before, 0)
    cost, estimated = _metric_cost(metric, calls, judge_model, assume_call_usd)
    return JudgeSample(score=score_f, reason=reason, passed=passed), cost, estimated, calls, None


def _metric_cost(
    metric: Any, calls: int, judge_model: str, assume_call_usd: float
) -> tuple[float, bool]:
    raw = getattr(metric, "evaluation_cost", None)
    amount = getattr(raw, "value", raw)
    if isinstance(amount, (int, float)) and amount > 0:
        return float(amount), False
    prompt = int(getattr(metric, "input_tokens", 0) or 0)
    completion = int(getattr(metric, "output_tokens", 0) or 0)
    if prompt or completion:
        return estimate_cost(judge_model, prompt, completion), True
    if calls and assume_call_usd:
        return calls * assume_call_usd, True
    return 0.0, False


def _read_calls(model: Any) -> int:
    calls = getattr(model, "calls", None)
    if isinstance(calls, int):
        return calls
    return 0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
