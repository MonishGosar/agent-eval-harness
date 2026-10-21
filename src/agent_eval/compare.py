"""Diff two run records and flag score regressions."""

from __future__ import annotations

from agent_eval.models import CaseDelta, CompareReport, MetricDelta, MetricResult, RunRecord


def compare_runs(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    regression_delta: float = 0.05,
) -> CompareReport:
    base_cases = {case.id: case for case in baseline.cases}
    cand_cases = {case.id: case for case in candidate.cases}
    missing_in_candidate = sorted(set(base_cases) - set(cand_cases))
    missing_in_baseline = sorted(set(cand_cases) - set(base_cases))
    metric_names = []
    for record in (baseline, candidate):
        for metric in record.metrics:
            if metric.name not in metric_names:
                metric_names.append(metric.name)

    regressions: list[CaseDelta] = []
    improvements: list[CaseDelta] = []
    metric_rows: list[MetricDelta] = []
    shared = sorted(set(base_cases) & set(cand_cases))
    for name in metric_names:
        base_scores: list[float] = []
        cand_scores: list[float] = []
        reg = 0
        imp = 0
        for case_id in shared:
            base_metric = _find(base_cases[case_id].metrics, name)
            cand_metric = _find(cand_cases[case_id].metrics, name)
            if base_metric is None or cand_metric is None:
                continue
            if base_metric.skipped or cand_metric.skipped:
                continue
            if base_metric.score is None or cand_metric.score is None:
                continue
            base_scores.append(base_metric.score)
            cand_scores.append(cand_metric.score)
            delta = cand_metric.score - base_metric.score
            kind = _kind(base_metric, cand_metric, delta, regression_delta)
            if kind is None:
                continue
            item = CaseDelta(
                case_id=case_id,
                metric=name,
                baseline_score=base_metric.score,
                candidate_score=cand_metric.score,
                delta=round(delta, 4),
                baseline_passed=base_metric.passed,
                candidate_passed=cand_metric.passed,
                kind=kind,
                reason=_reason(kind, base_metric, cand_metric, regression_delta),
            )
            if kind == "regression":
                regressions.append(item)
                reg += 1
            else:
                improvements.append(item)
                imp += 1
        base_avg = sum(base_scores) / len(base_scores) if base_scores else None
        cand_avg = sum(cand_scores) / len(cand_scores) if cand_scores else None
        metric_rows.append(
            MetricDelta(
                name=name,
                baseline=None if base_avg is None else round(base_avg, 4),
                candidate=None if cand_avg is None else round(cand_avg, 4),
                delta=None
                if base_avg is None or cand_avg is None
                else round(cand_avg - base_avg, 4),
                regressions=reg,
                improvements=imp,
            )
        )
    for case_id in missing_in_candidate:
        regressions.append(
            CaseDelta(
                case_id=case_id,
                metric="*",
                baseline_score=None,
                candidate_score=None,
                delta=None,
                baseline_passed=None,
                candidate_passed=None,
                kind="regression",
                reason="case is missing from the candidate run",
            )
        )
    return CompareReport(
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        regression_delta=regression_delta,
        regressions=regressions,
        improvements=improvements,
        metrics=metric_rows,
        missing_in_candidate=missing_in_candidate,
        missing_in_baseline=missing_in_baseline,
    )


def _find(metrics: list[MetricResult], name: str) -> MetricResult | None:
    for metric in metrics:
        if metric.name == name:
            return metric
    return None


def _kind(
    baseline: MetricResult,
    candidate: MetricResult,
    delta: float,
    regression_delta: float,
) -> str | None:
    if baseline.passed and not candidate.passed:
        return "regression"
    if delta <= -regression_delta:
        return "regression"
    if (not baseline.passed and candidate.passed) or delta >= regression_delta:
        return "improvement"
    return None


def _reason(
    kind: str,
    baseline: MetricResult,
    candidate: MetricResult,
    regression_delta: float,
) -> str:
    if baseline.passed and not candidate.passed:
        return (
            f"passed at {baseline.score} and now fails at {candidate.score} "
            f"(threshold {candidate.threshold})"
        )
    if not baseline.passed and candidate.passed:
        return f"failed at {baseline.score} and now passes at {candidate.score}"
    direction = "dropped" if kind == "regression" else "rose"
    return f"score {direction} by {abs((candidate.score or 0) - (baseline.score or 0)):.4f} (flag at {regression_delta})"


def render_compare_text(report: CompareReport) -> str:
    lines = [
        f"baseline  {report.baseline_run}",
        f"candidate {report.candidate_run}",
        f"regression delta {report.regression_delta}",
        "",
        f"{'metric':<24} {'baseline':>10} {'candidate':>10} {'delta':>10} {'reg':>5} {'imp':>5}",
    ]
    for metric in report.metrics:
        lines.append(
            f"{metric.name:<24} {_fmt(metric.baseline):>10} {_fmt(metric.candidate):>10} "
            f"{_fmt(metric.delta):>10} {metric.regressions:>5} {metric.improvements:>5}"
        )
    lines.append("")
    if report.regressions:
        lines.append(f"regressions ({len(report.regressions)})")
        for item in report.regressions:
            lines.append(
                f"  {item.case_id}  {item.metric}  "
                f"{_fmt(item.baseline_score)} -> {_fmt(item.candidate_score)}  {item.reason}"
            )
    else:
        lines.append("regressions: none")
    if report.improvements:
        lines.append(f"improvements ({len(report.improvements)})")
        for item in report.improvements:
            lines.append(
                f"  {item.case_id}  {item.metric}  "
                f"{_fmt(item.baseline_score)} -> {_fmt(item.candidate_score)}  {item.reason}"
            )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
