"""Run each golden once, then score every metric from the captured trace."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_eval import __version__
from agent_eval.adapters import build_adapter
from agent_eval.adapters.base import AgentAdapter
from agent_eval.budget import Budget
from agent_eval.cache import Cache, digest, file_digest, package_digest
from agent_eval.config import EvalConfig, MetricConfig, load_config, resolve_from
from agent_eval.dataset import load_dataset
from agent_eval.judge import build_judge
from agent_eval.loading import prepare_import_path
from agent_eval.metrics.factory import MetricFactory, build_factories
from agent_eval.models import (
    AgentRequest,
    AgentResult,
    CaseResult,
    Golden,
    MetricResult,
    MetricSummary,
    RunRecord,
    RunSummary,
    Usage,
)
from agent_eval.pricing import estimate_cost
from agent_eval.reporting.store import write_run
from agent_eval.scoring import metric_applies, score_metric, to_test_case


def execute(
    config_path: Path,
    *,
    adapter: AgentAdapter | None = None,
    judge: Any | None = None,
    concurrency: int | None = None,
    use_cache: bool | None = None,
    runs_dir: Path | None = None,
    case_ids: set[str] | None = None,
    tags: set[str] | None = None,
    budget_usd: float | None = None,
    max_cases: int | None = None,
) -> tuple[RunRecord, Path]:
    config_path = config_path.resolve()
    prepare_import_path(config_path)
    config = load_config(config_path)
    if budget_usd is not None:
        config.run.budget.max_usd = budget_usd
    if max_cases is not None:
        config.run.budget.max_cases = max_cases
    dataset_path = resolve_from(config_path, config.dataset.path)
    cases = load_dataset(dataset_path)
    if case_ids:
        cases = [case for case in cases if case.id in case_ids]
    if tags:
        cases = [case for case in cases if tags.intersection(case.tags)]
    if not cases:
        from agent_eval.errors import DatasetError

        raise DatasetError("no cases left after filters")

    agent = adapter or build_adapter(config.adapter)
    model = judge if judge is not None else build_judge(config.judge)
    factories = build_factories(config.metrics, model)
    cache_enabled = config.run.cache if use_cache is None else use_cache
    cache_dir = resolve_from(config_path, config.run.cache_dir)
    cache = Cache(cache_dir, enabled=cache_enabled)
    budget = Budget(config.run.budget)
    workers = concurrency or config.run.concurrency
    output_dir = runs_dir or resolve_from(config_path, config.run.runs_dir)

    results = _run_cases(
        cases,
        agent=agent,
        config=config,
        factories=factories,
        cache=cache,
        budget=budget,
        workers=workers,
        judge_model=config.judge.model,
    )
    record = _assemble(
        results,
        config=config,
        config_path=config_path,
        dataset_path=dataset_path,
        budget=budget,
    )
    destination = write_run(output_dir, record, config_path)
    return record, destination


def _run_cases(
    cases: list[Golden],
    *,
    agent: AgentAdapter,
    config: EvalConfig,
    factories: list[tuple[MetricConfig, MetricFactory]],
    cache: Cache,
    budget: Budget,
    workers: int,
    judge_model: str,
) -> list[CaseResult]:
    pending = list(cases)
    futures: dict[Future[CaseResult], Golden] = {}
    finished: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or futures:
            while pending and len(futures) < workers and budget.try_reserve_case():
                case = pending.pop(0)
                future = pool.submit(
                    _evaluate_case,
                    case,
                    agent=agent,
                    config=config,
                    factories=factories,
                    cache=cache,
                    budget=budget,
                    judge_model=judge_model,
                )
                futures[future] = case
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                finished.append(future.result())
    reason = budget.skip_reason() if budget.exceeded else "not scheduled"
    for case in pending:
        finished.append(_skipped(case, reason if budget.exceeded else "not scheduled"))
    order = {case.id: index for index, case in enumerate(cases)}
    finished.sort(key=lambda item: order.get(item.id, 0))
    return finished


def _evaluate_case(
    case: Golden,
    *,
    agent: AgentAdapter,
    config: EvalConfig,
    factories: list[tuple[MetricConfig, MetricFactory]],
    cache: Cache,
    budget: Budget,
    judge_model: str,
) -> CaseResult:
    request = AgentRequest(
        case_id=case.id,
        input=case.input,
        turns=list(case.turns),
        session_id=str(case.metadata.get("session_id") or "") or None,
        metadata=dict(case.metadata),
    )
    agent_key = _agent_key(config, request)
    cached = cache.get_agent(agent_key)
    started = time.perf_counter()
    agent_cached = cached is not None
    if cached is not None:
        result = cached
    else:
        try:
            result = agent.run(request).normalized()
        except Exception as exc:
            result = AgentResult(output="", error=f"{type(exc).__name__}: {exc}")
        result = result.model_copy(
            update={"latency_ms": result.latency_ms or (time.perf_counter() - started) * 1000}
        )
        cache.put_agent(agent_key, result)

    agent_cost, agent_estimated = _agent_cost(result.usage, judge_model)
    budget.charge(agent_cost, estimated=agent_estimated)

    if result.error:
        return CaseResult(
            id=case.id,
            status="error",
            input=case.input,
            rendered_input=case.rendered_input(),
            expected_output=case.expected_output,
            actual_output=result.output,
            tags=list(case.tags),
            plan=list(result.plan),
            plan_captured=bool(result.plan),
            tool_calls=list(result.tool_calls),
            expected_tools=case.expected_tools,
            usage=result.usage,
            latency_ms=result.latency_ms,
            agent_cached=agent_cached,
            error=result.error,
        )

    test_case = to_test_case(case, result, case.task or _task_default(config))
    metric_results: list[MetricResult] = []
    for metric, factory in factories:
        if not metric_applies(metric, case):
            metric_results.append(
                MetricResult(
                    name=metric.name,
                    threshold=metric.threshold,
                    passed=True,
                    skipped=True,
                    reason="case has no expected_tools; tool correctness was not scored",
                    judged=False,
                )
            )
            continue
        bound = _bind_task(factory, metric, case)

        def cache_key_for(index: int, metric: MetricConfig = metric) -> str:
            return digest(
                "metric",
                config.judge.model,
                config.judge.provider,
                metric.model_dump(mode="json"),
                case.model_dump(mode="json"),
                result.model_dump(mode="json"),
                index,
            )

        scored, cost, estimated, calls = score_metric(
            metric,
            bound,
            test_case,
            samples=config.judge.samples,
            spread=config.judge.disagreement_spread,
            judge_model=judge_model,
            assume_call_usd=config.run.budget.assume_judge_call_usd,
            cache_get=cache.get_metric,
            cache_put=cache.put_metric,
            cache_key_for=cache_key_for,
        )
        if scored.disagreement and config.judge.fail_on_disagreement:
            scored.passed = False
            scored.reason = (
                f"Judge samples disagreed (spread {scored.spread}). {scored.reason}"
            ).strip()
        budget.charge(cost, estimated=estimated, judge_calls=calls)
        metric_results.append(scored)

    applicable = [item for item in metric_results if not item.skipped]
    failed = any(not item.passed for item in applicable)
    return CaseResult(
        id=case.id,
        status="failed" if failed else "passed",
        input=case.input,
        rendered_input=case.rendered_input(),
        expected_output=case.expected_output,
        actual_output=result.output,
        tags=list(case.tags),
        plan=list(result.plan),
        plan_captured=bool(result.plan),
        tool_calls=list(result.tool_calls),
        expected_tools=case.expected_tools,
        metrics=metric_results,
        usage=result.usage,
        latency_ms=result.latency_ms,
        agent_cached=agent_cached,
    )


def _bind_task(factory: MetricFactory, metric: MetricConfig, case: Golden) -> MetricFactory:
    task = case.task or metric.task
    if (metric.type or metric.name) != "task_completion" or not task:
        return factory

    def bound() -> Any:
        instance = factory()
        instance.task = task
        if hasattr(instance, "_is_task_provided"):
            instance._is_task_provided = True
        return instance

    return bound


def _task_default(config: EvalConfig) -> str | None:
    for metric in config.metrics:
        if (metric.type or metric.name) == "task_completion" and metric.task:
            return metric.task
    return None


def _agent_key(config: EvalConfig, request: AgentRequest) -> str:
    module = ""
    if config.adapter.target:
        module = package_digest(config.adapter.target.split(":", 1)[0])
    return digest(
        "agent",
        config.adapter.model_dump(mode="json"),
        request.model_dump(mode="json"),
        module,
    )


def _agent_cost(usage: Usage, model: str) -> tuple[float, bool]:
    if usage.cost_usd is not None:
        return usage.cost_usd, False
    if usage.prompt_tokens or usage.completion_tokens:
        return estimate_cost(model, usage.prompt_tokens, usage.completion_tokens), True
    return 0.0, False


def _skipped(case: Golden, reason: str) -> CaseResult:
    return CaseResult(
        id=case.id,
        status="skipped",
        input=case.input,
        rendered_input=case.rendered_input(),
        expected_output=case.expected_output,
        tags=list(case.tags),
        expected_tools=case.expected_tools,
        skip_reason=reason,
    )


def _assemble(
    cases: list[CaseResult],
    *,
    config: EvalConfig,
    config_path: Path,
    dataset_path: Path,
    budget: Budget,
) -> RunRecord:
    summaries: list[MetricSummary] = []
    for metric in config.metrics:
        rows = []
        for case in cases:
            for item in case.metrics:
                if item.name == metric.name and not item.skipped and item.score is not None:
                    rows.append(item)
        passed = sum(1 for item in rows if item.passed)
        failed = sum(1 for item in rows if not item.passed)
        skipped = sum(
            1
            for case in cases
            for item in case.metrics
            if item.name == metric.name and item.skipped
        )
        disagreements = sum(1 for item in rows if item.disagreement)
        average = (
            sum(item.score for item in rows if item.score is not None) / len(rows) if rows else None
        )
        pass_rate = passed / len(rows) if rows else None
        summaries.append(
            MetricSummary(
                name=metric.name,
                threshold=metric.threshold,
                average=None if average is None else round(average, 4),
                pass_rate=None if pass_rate is None else round(pass_rate, 4),
                passed=passed,
                failed=failed,
                skipped=skipped,
                disagreements=disagreements,
            )
        )
    passed_cases = sum(1 for case in cases if case.status == "passed")
    failed_cases = sum(1 for case in cases if case.status == "failed")
    errors = sum(1 for case in cases if case.status == "error")
    skipped_cases = sum(1 for case in cases if case.status == "skipped")
    scored = passed_cases + failed_cases
    thresholds_failed = failed_cases > 0 or errors > 0
    summary = RunSummary(
        cases=len(cases),
        passed=passed_cases,
        failed=failed_cases,
        errors=errors,
        skipped=skipped_cases,
        pass_rate=round(passed_cases / scored, 4) if scored else 0.0,
        agent_runs=sum(1 for case in cases if case.status != "skipped" and not case.agent_cached),
        cache_hits=sum(1 for case in cases if case.agent_cached),
        prompt_tokens=sum(case.usage.prompt_tokens for case in cases),
        completion_tokens=sum(case.usage.completion_tokens for case in cases),
        cost_usd=round(budget.spent, 6),
        cost_estimated=budget.estimated,
        disagreements=sum(item.disagreements for item in summaries),
        thresholds_failed=thresholds_failed,
        budget_exceeded=budget.exceeded or skipped_cases > 0,
    )
    if summary.budget_exceeded and not thresholds_failed:
        status = "budget_exceeded"
    elif thresholds_failed:
        status = "failed"
    else:
        status = "passed"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid4().hex[:6]}"
    return RunRecord(
        run_id=run_id,
        project=config.project,
        created_at=datetime.now(UTC).isoformat(),
        harness_version=__version__,
        config_path=str(config_path),
        config_hash=file_digest(config_path),
        dataset_path=str(dataset_path),
        dataset_hash=file_digest(dataset_path),
        judge_model=config.judge.model,
        judge_provider=config.judge.provider,
        adapter_type=config.adapter.type,
        status=status,
        summary=summary,
        metrics=summaries,
        cases=cases,
    )


def exit_code(record: RunRecord) -> int:
    """1 when a threshold or agent error failed the run, 2 when the budget stopped it early."""
    if record.summary.thresholds_failed:
        return 1
    if record.summary.budget_exceeded:
        return 2
    return 0
