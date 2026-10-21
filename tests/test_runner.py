from pathlib import Path

from agent_eval.reporting.store import load_run
from agent_eval.runner import execute, exit_code
from tests.scripted_judge import ScriptedJudge
from tests.support import calls, reset

METRICS = """
metrics:
  - name: task_completion
    threshold: 0.7
    task: Tell the customer the order status.
  - name: step_efficiency
    threshold: 0.5
  - name: plan_quality
    threshold: 0.5
  - name: plan_adherence
    threshold: 0.5
  - name: tool_correctness
    threshold: 1.0
    exact_match: true
    input_parameters: true
  - name: argument_correctness
    threshold: 0.5
  - name: answer_correctness
    type: geval
    threshold: 0.5
    criteria: Same logical conclusion as the expected output.
    evaluation_steps:
      - Compare the order status.
"""


def _write(tmp_path: Path, *, samples: int = 1, cases: str | None = None) -> Path:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        cases
        or (
            "cases:\n"
            "  - id: order-17\n"
            "    input: What is the status of order 17?\n"
            "    expected_output: Order 17 was delivered.\n"
            "    expected_tools:\n"
            "      - name: lookup_order\n"
            "        arguments: {order_id: '17'}\n"
            "  - id: order-18\n"
            "    input: What is the status of order 18?\n"
            "    expected_output: Order 17 was delivered.\n"
            "    tags: [other]\n"
            "    expected_tools:\n"
            "      - name: lookup_order\n"
            "        arguments: {order_id: '17'}\n"
        ),
        encoding="utf-8",
    )
    config = tmp_path / "evals.yaml"
    config.write_text(
        "version: 1\n"
        "project: demo\n"
        "adapter:\n"
        "  type: python\n"
        "  target: tests.support:run_agent\n"
        "judge:\n"
        "  provider: openai\n"
        "  model: gpt-4.1-mini\n"
        f"  samples: {samples}\n"
        "  disagreement_spread: 0.25\n"
        "dataset: {path: dataset.yaml}\n"
        "run:\n"
        "  concurrency: 2\n"
        "  cache: true\n"
        "  cache_dir: .cache\n"
        "  runs_dir: runs\n"
        "  budget: {max_usd: 5, assume_judge_call_usd: 0.01}\n"
        f"{METRICS}",
        encoding="utf-8",
    )
    return config


def test_one_agent_run_scores_every_metric(tmp_path: Path) -> None:
    reset()
    config = _write(tmp_path)
    record, destination = execute(
        config,
        judge=ScriptedJudge(),
        case_ids={"order-17"},
        runs_dir=tmp_path / "runs",
    )
    assert calls["n"] == 1
    assert record.status == "passed"
    assert exit_code(record) == 0
    case = record.cases[0]
    assert case.agent_cached is False
    assert case.plan_captured is True
    names = [metric.name for metric in case.metrics]
    assert names == [
        "task_completion",
        "step_efficiency",
        "plan_quality",
        "plan_adherence",
        "tool_correctness",
        "argument_correctness",
        "answer_correctness",
    ]
    assert all(metric.passed for metric in case.metrics)
    assert (destination / "results.json").is_file()
    assert (destination / "report.html").is_file()
    html = (destination / "report.html").read_text(encoding="utf-8")
    assert "order-17" in html
    assert "lookup_order" in html
    assert "same logical conclusion" in html
    assert load_run(destination).run_id == record.run_id

    record_again, _ = execute(
        config,
        judge=ScriptedJudge(),
        case_ids={"order-17"},
        runs_dir=tmp_path / "runs",
    )
    assert calls["n"] == 1
    assert record_again.cases[0].agent_cached is True
    assert record_again.summary.cache_hits == 1


def test_low_judge_score_fails_the_run(tmp_path: Path) -> None:
    reset()
    config = _write(tmp_path)
    record, _ = execute(
        config,
        judge=ScriptedJudge(scores=[0.1]),
        case_ids={"order-17"},
        runs_dir=tmp_path / "runs",
    )
    assert record.status == "failed"
    assert exit_code(record) == 1
    answer = next(
        metric for metric in record.cases[0].metrics if metric.name == "answer_correctness"
    )
    assert answer.score is not None and answer.score < 0.5
    assert answer.passed is False


def test_judge_disagreement_is_recorded(tmp_path: Path) -> None:
    reset()
    config = _write(tmp_path, samples=2)
    record, destination = execute(
        config,
        judge=ScriptedJudge(scores=[0.2, 0.9], advance=True),
        case_ids={"order-17"},
        runs_dir=tmp_path / "runs",
    )
    answer = next(
        metric for metric in record.cases[0].metrics if metric.name == "answer_correctness"
    )
    assert answer.disagreement is True
    assert answer.spread > 0.25
    assert len(answer.samples) == 2
    assert record.summary.disagreements >= 1
    html = (destination / "report.html").read_text(encoding="utf-8")
    assert "Judge disagreements" in html
    assert "0.200" in html or "0.900" in html


def test_budget_skips_remaining_cases(tmp_path: Path) -> None:
    reset()
    config = _write(tmp_path)
    record, _ = execute(
        config,
        judge=ScriptedJudge(),
        max_cases=1,
        runs_dir=tmp_path / "runs",
    )
    statuses = {case.id: case.status for case in record.cases}
    assert statuses["order-17"] == "passed"
    assert statuses["order-18"] == "skipped"
    assert record.summary.budget_exceeded is True
    assert exit_code(record) == 2
    assert calls["n"] == 1


def test_html_escapes_case_text(tmp_path: Path) -> None:
    reset()
    config = _write(
        tmp_path,
        cases=(
            "cases:\n"
            "  - id: xss\n"
            "    input: <script>alert(1)</script>\n"
            "    expected_output: Order 17 was delivered.\n"
            "    expected_tools:\n"
            "      - {name: lookup_order, arguments: {order_id: '17'}}\n"
        ),
    )
    _, destination = execute(config, judge=ScriptedJudge(), runs_dir=tmp_path / "runs")
    html = (destination / "report.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
