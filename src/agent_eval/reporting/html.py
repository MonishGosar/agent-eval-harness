"""Self-contained HTML reports. No external assets, so a run directory can be opened offline."""

from __future__ import annotations

import html
from typing import Any

from agent_eval.models import CaseResult, CompareReport, MetricResult, RunRecord


def render_report(record: RunRecord) -> str:
    summary = record.summary
    cost = f"${summary.cost_usd:.4f}"
    if summary.cost_estimated:
        cost += " estimated"
    disagreements = _disagreement_block(record)
    plan_warning = _plan_warning(record)
    metrics = "\n".join(_metric_card(metric) for metric in record.metrics)
    cases = "\n".join(_case_block(case) for case in record.cases)
    status = html.escape(record.status)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(record.project)} · {html.escape(record.run_id)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f5f2;
  --paper: #fff;
  --ink: #1c1915;
  --muted: #5e584f;
  --line: #e4dfd6;
  --pass: #1f7a4d;
  --fail: #a33b32;
  --warn: #8a5a12;
  --warn-bg: #fbf3e4;
  --chip: #efece6;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.5 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
}}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }}
header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; }}
h1 {{ font-size: 28px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.02em; }}
h2 {{ font-size: 18px; margin: 28px 0 12px; }}
.sub {{ color: var(--muted); margin: 0; }}
.pill {{
  border-radius: 999px; padding: 4px 12px; font: 12px/1.4 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.04em; text-transform: uppercase; background: var(--chip);
}}
.pill.passed {{ color: var(--pass); }}
.pill.failed, .pill.error {{ color: var(--fail); }}
.pill.budget_exceeded, .pill.skipped {{ color: var(--warn); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 18px; }}
.stat, .card, details {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; }}
.stat {{ padding: 12px 14px; }}
.stat b {{ display: block; font-size: 20px; font-variant-numeric: tabular-nums; }}
.stat span {{ color: var(--muted); font: 12px/1.3 ui-sans-serif, system-ui, sans-serif; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
.card {{ padding: 12px 14px; }}
.card .name {{ font: 13px/1.3 ui-sans-serif, system-ui, sans-serif; color: var(--muted); }}
.card .avg {{ font-size: 22px; font-variant-numeric: tabular-nums; }}
.warn {{ background: var(--warn-bg); border: 1px solid #ead7b4; border-radius: 12px; padding: 12px 14px; }}
details {{ padding: 0 14px; margin: 8px 0; }}
summary {{ cursor: pointer; padding: 12px 0; display: flex; justify-content: space-between; gap: 12px; }}
table {{ width: 100%; border-collapse: collapse; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }}
th, td {{ text-align: left; padding: 6px 8px; border-top: 1px solid var(--line); vertical-align: top; }}
pre, code {{ font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
pre {{ white-space: pre-wrap; background: var(--chip); padding: 10px; border-radius: 8px; }}
.fail {{ color: var(--fail); }}
.pass {{ color: var(--pass); }}
.samples {{ margin: 0 0 12px; padding-left: 18px; }}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>{html.escape(record.project)}</h1>
    <p class="sub">{html.escape(record.run_id)} · {html.escape(record.created_at)} · harness {html.escape(record.harness_version)}</p>
    <p class="sub">judge {html.escape(record.judge_provider)}/{html.escape(record.judge_model)} · adapter {html.escape(record.adapter_type)}</p>
  </div>
  <div class="pill {status}">{status}</div>
</header>
<section class="grid">
  {_stat(f"{summary.passed}/{summary.cases}", "cases passed")}
  {_stat(f"{summary.pass_rate:.0%}", "case pass rate")}
  {_stat(str(summary.disagreements), "judge disagreements")}
  {_stat(str(summary.agent_runs), "agent runs")}
  {_stat(str(summary.cache_hits), "cache hits")}
  {_stat(f"{summary.prompt_tokens + summary.completion_tokens}", "tokens")}
  {_stat(cost, "cost")}
  {_stat(str(summary.errors), "agent errors")}
</section>
{plan_warning}
{disagreements}
<h2>Metrics</h2>
<section class="cards">
{metrics}
</section>
<h2>Cases</h2>
{cases}
</main>
</body>
</html>
"""


def render_compare(report: CompareReport) -> str:
    rows = []
    for metric in report.metrics:
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric.name)}</td>"
            f"<td>{_num(metric.baseline)}</td>"
            f"<td>{_num(metric.candidate)}</td>"
            f"<td>{_num(metric.delta)}</td>"
            f"<td>{metric.regressions}</td>"
            f"<td>{metric.improvements}</td>"
            "</tr>"
        )
    regressions = (
        "".join(
            f"<li><b>{html.escape(item.case_id)}</b> {html.escape(item.metric)} "
            f"{_num(item.baseline_score)} → {_num(item.candidate_score)} "
            f"· {html.escape(item.reason)}</li>"
            for item in report.regressions
        )
        or "<li>None</li>"
    )
    improvements = (
        "".join(
            f"<li><b>{html.escape(item.case_id)}</b> {html.escape(item.metric)} "
            f"{_num(item.baseline_score)} → {_num(item.candidate_score)}</li>"
            for item in report.improvements
        )
        or "<li>None</li>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>compare {html.escape(report.baseline_run)} → {html.escape(report.candidate_run)}</title>
<style>
body {{ margin: 0; background: #f6f5f2; color: #1c1915; font: 15px/1.5 Georgia, serif; }}
main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e4dfd6; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }}
h1 {{ font-size: 24px; }}
</style>
</head>
<body>
<main>
<h1>Compare</h1>
<p>{html.escape(report.baseline_run)} → {html.escape(report.candidate_run)}</p>
<p>A regression is a pass that became a fail, or a score drop of at least {report.regression_delta}.</p>
<table>
<thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Regressions</th><th>Improvements</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<h2>Regressions</h2>
<ul>{regressions}</ul>
<h2>Improvements</h2>
<ul>{improvements}</ul>
</main>
</body>
</html>
"""


def _stat(value: str, label: str) -> str:
    return f'<div class="stat"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'


def _metric_card(metric: Any) -> str:
    average = "-" if metric.average is None else f"{metric.average:.3f}"
    rate = "-" if metric.pass_rate is None else f"{metric.pass_rate:.0%}"
    return (
        '<article class="card">'
        f'<div class="name">{html.escape(metric.name)}</div>'
        f'<div class="avg">{html.escape(average)}</div>'
        f'<div class="name">pass {html.escape(rate)} · threshold {metric.threshold:.2f}'
        f" · disagree {metric.disagreements}</div>"
        "</article>"
    )


def _disagreement_block(record: RunRecord) -> str:
    items: list[str] = []
    for case in record.cases:
        for metric in case.metrics:
            if not metric.disagreement:
                continue
            samples = "; ".join(
                f"{sample.score:.3f} ({sample.reason})" for sample in metric.samples
            )
            items.append(
                f"<li><b>{html.escape(case.id)}</b> · {html.escape(metric.name)} "
                f"spread {metric.spread:.3f}. Median {metric.score}. "
                f"{html.escape(samples)}</li>"
            )
    if not items:
        return ""
    return (
        "<h2>Judge disagreements</h2>"
        '<div class="warn"><ul>' + "".join(items) + "</ul>"
        "<p>The case score is the median sample. Disagreement means the samples "
        "crossed the spread setting or did not agree on pass/fail.</p></div>"
    )


def _plan_warning(record: RunRecord) -> str:
    names = {metric.name for metric in record.metrics}
    watched = names.intersection({"plan_quality", "plan_adherence"})
    if not watched:
        return ""
    missing = [
        case.id for case in record.cases if case.status != "skipped" and not case.plan_captured
    ]
    if not missing:
        return ""
    listed = ", ".join(html.escape(case_id) for case_id in missing)
    return (
        "<h2>Plan capture</h2>"
        f'<div class="warn"><p>No plan text was captured for {listed}. '
        "Plan metrics read the trace, and an empty plan is a weak signal. "
        "Have the agent emit a line starting with <code>Plan:</code> before it calls tools.</p></div>"
    )


def _case_block(case: CaseResult) -> str:
    status = html.escape(case.status)
    metric_rows = "".join(_metric_row(metric) for metric in case.metrics)
    tools = (
        "".join(_tool_row(call) for call in case.tool_calls)
        or "<tr><td colspan='3'>No tool calls</td></tr>"
    )
    plan = (
        "<br/>".join(html.escape(step) for step in case.plan)
        or "<span class='fail'>no plan captured</span>"
    )
    extra = ""
    if case.error:
        extra += f"<p class='fail'>{html.escape(case.error)}</p>"
    if case.skip_reason:
        extra += f"<p>{html.escape(case.skip_reason)}</p>"
    return f"""
<details>
<summary>
  <span><b>{html.escape(case.id)}</b> · {status}</span>
  <span class="{status}">{status}</span>
</summary>
{extra}
<p><b>Input</b></p>
<pre>{html.escape(case.rendered_input)}</pre>
<p><b>Expected</b></p>
<pre>{html.escape(case.expected_output)}</pre>
<p><b>Actual</b></p>
<pre>{html.escape(case.actual_output)}</pre>
<p><b>Plan</b></p>
<p>{plan}</p>
<table>
<thead><tr><th>Metric</th><th>Score</th><th>Reason</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>
<p><b>Tools</b></p>
<table>
<thead><tr><th>Tool</th><th>Arguments</th><th>Output</th></tr></thead>
<tbody>{tools}</tbody>
</table>
<p class="sub">latency {case.latency_ms:.0f} ms · tokens {case.usage.prompt_tokens}+{case.usage.completion_tokens} · cached {str(case.agent_cached).lower()}</p>
</details>
"""


def _metric_row(metric: MetricResult) -> str:
    if metric.skipped:
        score = "skipped"
        klass = ""
    elif metric.score is None:
        score = "-"
        klass = "fail"
    else:
        score = f"{metric.score:.3f} / {metric.threshold:.2f}"
        klass = "pass" if metric.passed else "fail"
    samples = ""
    if len(metric.samples) > 1:
        items = "".join(
            f"<li>{sample.score:.3f} · {html.escape(sample.reason)}</li>"
            for sample in metric.samples
        )
        samples = f"<ul class='samples'>{items}</ul>"
    flag = " · disagreement" if metric.disagreement else ""
    return (
        "<tr>"
        f"<td>{html.escape(metric.name)}{flag}</td>"
        f"<td class='{klass}'>{html.escape(score)}</td>"
        f"<td>{html.escape(metric.reason)}{samples}</td>"
        "</tr>"
    )


def _tool_row(call: Any) -> str:
    import json

    arguments = json.dumps(call.arguments, indent=2, default=str)
    output = json.dumps(call.output, indent=2, default=str)
    if len(output) > 2000:
        output = output[:2000] + "…"
    return (
        "<tr>"
        f"<td>{html.escape(call.name)}</td>"
        f"<td><pre>{html.escape(arguments)}</pre></td>"
        f"<td><pre>{html.escape(output)}</pre></td>"
        "</tr>"
    )


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
