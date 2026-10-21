"""Persist a run as results.json plus a self-contained HTML report."""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.models import RunRecord
from agent_eval.reporting.html import render_report


def write_run(runs_dir: Path, record: RunRecord, config_path: Path) -> Path:
    destination = runs_dir / record.run_id
    destination.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json")
    (destination / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (destination / "report.html").write_text(render_report(record), encoding="utf-8")
    snapshot = destination / "evals.yaml"
    if config_path.is_file():
        snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def load_run(path: Path) -> RunRecord:
    target = path
    if target.is_dir():
        target = target / "results.json"
    if not target.is_file():
        from agent_eval.errors import ConfigError

        raise ConfigError(f"run results not found: {path}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return RunRecord.model_validate(data)


def latest_run(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        from agent_eval.errors import ConfigError

        raise ConfigError(f"runs directory not found: {runs_dir}")
    candidates = [item for item in runs_dir.iterdir() if (item / "results.json").is_file()]
    if not candidates:
        from agent_eval.errors import ConfigError

        raise ConfigError(f"no runs in {runs_dir}")
    return max(candidates, key=lambda item: item.name)


def render_summary(record: RunRecord) -> str:
    summary = record.summary
    cost = f"${summary.cost_usd:.4f}"
    if summary.cost_estimated:
        cost += " (estimated)"
    lines = [
        f"{record.project}  {record.run_id}  {record.status}",
        (
            f"cases {summary.passed}/{summary.cases} passed"
            f"  failed {summary.failed}  errors {summary.errors}  skipped {summary.skipped}"
        ),
        (
            f"agent runs {summary.agent_runs}  cache hits {summary.cache_hits}"
            f"  tokens {summary.prompt_tokens}+{summary.completion_tokens}"
            f"  cost {cost}"
        ),
        f"judge disagreements {summary.disagreements}",
        "",
        f"{'metric':<24} {'avg':>8} {'pass':>8} {'threshold':>10} {'disagree':>10}",
    ]
    for metric in record.metrics:
        average = "-" if metric.average is None else f"{metric.average:.3f}"
        rate = "-" if metric.pass_rate is None else f"{metric.pass_rate:.0%}"
        lines.append(
            f"{metric.name:<24} {average:>8} {rate:>8} {metric.threshold:>10.2f} "
            f"{metric.disagreements:>10}"
        )
    lines.append("")
    for case in record.cases:
        if case.status == "passed":
            continue
        detail = case.error or case.skip_reason or _failed_metrics(case)
        lines.append(f"{case.status.upper():<8} {case.id}  {detail}")
    return "\n".join(lines)


def _failed_metrics(case: object) -> str:
    from agent_eval.models import CaseResult

    if not isinstance(case, CaseResult):
        return ""
    parts = []
    for metric in case.metrics:
        if metric.skipped or metric.passed:
            if metric.disagreement:
                parts.append(f"{metric.name} disagree {metric.spread:.2f}")
            continue
        score = "-" if metric.score is None else f"{metric.score:.3f}"
        parts.append(f"{metric.name} {score}<{metric.threshold:.2f}")
    return ", ".join(parts)
