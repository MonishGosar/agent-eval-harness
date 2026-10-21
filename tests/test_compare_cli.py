import json
from pathlib import Path

from agent_eval.cli import main
from agent_eval.compare import compare_runs
from agent_eval.models import CaseResult, MetricResult, MetricSummary, RunRecord, RunSummary
from agent_eval.reporting.store import load_run
from tests.scripted_judge import ScriptedJudge
from tests.support import reset


def _record(run_id: str, score: float, passed: bool) -> RunRecord:
    metric = MetricResult(
        name="answer_correctness",
        score=score,
        threshold=0.7,
        passed=passed,
        reason="because",
        judged=True,
    )
    return RunRecord(
        run_id=run_id,
        project="demo",
        created_at="2026-10-21T00:00:00+00:00",
        harness_version="0.1.0",
        config_path="evals.yaml",
        config_hash="a",
        dataset_path="dataset.yaml",
        dataset_hash="b",
        judge_model="gpt-4.1-mini",
        judge_provider="openai",
        adapter_type="python",
        status="passed" if passed else "failed",
        summary=RunSummary(
            cases=1, passed=int(passed), failed=int(not passed), pass_rate=float(passed)
        ),
        metrics=[
            MetricSummary(
                name="answer_correctness", threshold=0.7, average=score, pass_rate=float(passed)
            )
        ],
        cases=[
            CaseResult(
                id="order-17",
                status="passed" if passed else "failed",
                input="status?",
                rendered_input="status?",
                expected_output="delivered",
                actual_output="delivered",
                metrics=[metric],
            )
        ],
    )


def test_compare_flags_a_regression(tmp_path: Path) -> None:
    baseline = _record("before", 0.9, True)
    candidate = _record("after", 0.4, False)
    report = compare_runs(baseline, candidate, regression_delta=0.05)
    assert len(report.regressions) == 1
    assert report.regressions[0].case_id == "order-17"
    assert report.metrics[0].delta == -0.5

    base_dir = tmp_path / "before"
    cand_dir = tmp_path / "after"
    base_dir.mkdir()
    cand_dir.mkdir()
    (base_dir / "results.json").write_text(baseline.model_dump_json(), encoding="utf-8")
    (cand_dir / "results.json").write_text(candidate.model_dump_json(), encoding="utf-8")
    code = main(["compare", str(base_dir), str(cand_dir), "--delta", "0.05"])
    assert code == 1
    saved = json.loads((cand_dir / "compare-before.json").read_text(encoding="utf-8"))
    assert saved["regressions"]
    assert (cand_dir / "compare-before.html").is_file()
    allowed = main(
        ["compare", str(base_dir), str(cand_dir), "--delta", "0.05", "--allow-regression"]
    )
    assert allowed == 0


def test_init_and_missing_key(tmp_path: Path) -> None:
    assert main(["init", "--dir", str(tmp_path)]) == 0
    assert (tmp_path / "evals.yaml").is_file()
    assert (tmp_path / "dataset.yaml").is_file()
    assert main(["init", "--dir", str(tmp_path)]) == 0

    config = tmp_path / "live.yaml"
    dataset = tmp_path / "live.jsonl"
    dataset.write_text(
        '{"id":"a","input":"hi","expected_output":"ok","expected_tools":[]}\n',
        encoding="utf-8",
    )
    config.write_text(
        "version: 1\nproject: live\n"
        "adapter: {type: python, target: tests.support:run_agent}\n"
        "judge: {provider: openai, model: gpt-4.1-mini}\n"
        f"dataset: {{path: {dataset.name}}}\n"
        "metrics:\n"
        "  - {name: answer_correctness, type: geval, threshold: 0.5,\n"
        "     criteria: same conclusion, evaluation_steps: [compare]}\n",
        encoding="utf-8",
    )
    reset()
    code = main(["run", "--config", str(config), "--runs-dir", str(tmp_path / "runs")])
    assert code == 1
    from tests.support import calls

    assert calls["n"] == 0


def test_cli_run_with_scripted_judge(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "cases:\n"
        "  - id: order-17\n"
        "    input: What is the status of order 17?\n"
        "    expected_output: Order 17 was delivered.\n"
        "    expected_tools:\n"
        "      - {name: lookup_order, arguments: {order_id: '17'}}\n",
        encoding="utf-8",
    )
    config = tmp_path / "evals.yaml"
    config.write_text(
        "version: 1\nproject: demo\n"
        "adapter: {type: python, target: tests.support:run_agent}\n"
        "judge: {provider: openai, model: gpt-4.1-mini, samples: 1}\n"
        "dataset: {path: dataset.yaml}\n"
        "run: {cache: false, runs_dir: runs, concurrency: 1}\n"
        "metrics:\n"
        "  - {name: tool_correctness, threshold: 1, exact_match: true, input_parameters: true}\n"
        "  - {name: answer_correctness, type: geval, threshold: 0.5,\n"
        "     criteria: same conclusion, evaluation_steps: [compare the status]}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_eval.runner.build_judge", lambda config: ScriptedJudge())
    reset()
    code = main(
        ["run", "--config", str(config), "--runs-dir", str(tmp_path / "runs"), "--no-cache"]
    )
    assert code == 0
    record = load_run(next((tmp_path / "runs").iterdir()))
    assert record.status == "passed"
    report_code = main(["report", "--run", str(tmp_path / "runs" / record.run_id)])
    assert report_code == 0
