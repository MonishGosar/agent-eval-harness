from pathlib import Path

import pytest

from agent_eval.config import load_config
from agent_eval.dataset import load_dataset
from agent_eval.errors import ConfigError, DatasetError


def test_example_config_and_dataset_load() -> None:
    root = Path("examples/refund_agent")
    config = load_config(root / "evals.yaml")
    baseline = load_config(root / "evals.baseline.yaml")
    cases = load_dataset(root / "dataset.yaml")
    assert config.adapter.kwargs["variant"] == "improved"
    assert baseline.adapter.kwargs["variant"] == "baseline"
    assert config.judge.model == "gpt-4.1-mini"
    assert {case.id for case in cases} >= {
        "full_refund",
        "partial_refund",
        "policy_question",
        "unknown_user",
        "out_of_window",
        "mixed_eligibility",
        "never_purchased",
    }
    policy = next(case for case in cases if case.id == "policy_question")
    assert policy.expected_tools == []
    followup = next(case for case in cases if case.id == "followup_refund")
    assert followup.turns[0].role == "user"
    assert "u_alice" in followup.rendered_input()


def test_jsonl_dataset(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"a","input":"hi","expected_output":"ok"}\n\n'
        '{"id":"b","input":"yo","expected_output":"ok","expected_tools":[{"name":"t"}]}\n',
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert [case.id for case in cases] == ["a", "b"]
    assert cases[0].expected_tools is None
    assert cases[1].expected_tools is not None


def test_duplicate_case_id(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n"
        "  - {id: a, input: x, expected_output: y}\n"
        "  - {id: a, input: x, expected_output: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


def test_unknown_metric(tmp_path: Path) -> None:
    path = tmp_path / "evals.yaml"
    path.write_text(
        "version: 1\nproject: p\n"
        "adapter: {type: python, target: tests.support:run_agent}\n"
        "dataset: {path: dataset.yaml}\n"
        "metrics: [{name: not_a_metric, type: not_a_metric}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown metric"):
        load_config(path)
