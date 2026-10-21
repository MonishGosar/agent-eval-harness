"""agent-eval command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_eval import __version__
from agent_eval.errors import AgentEvalError

INIT_CONFIG = """\
version: 1
project: my-agent

adapter:
  type: python
  target: my_agent.agent:run
  # type: langchain
  # target: my_agent.agent:build_agent
  # kwargs: {}
  # type: openai_agents
  # target: my_agent.agent:build_agent
  # type: http
  # url: http://localhost:8000/agent
  # headers:
  #   Authorization: "Bearer ${AGENT_TOKEN}"

judge:
  provider: openai          # openai | azure
  model: gpt-4.1-mini       # cheap default
  samples: 3                # >1 surfaces judge disagreement
  disagreement_spread: 0.25
  fail_on_disagreement: false
  # provider: azure
  # deployment: gpt-4.1-mini
  # api_version: "2024-10-21"
  # api_key_env: AZURE_OPENAI_API_KEY
  # endpoint_env: AZURE_OPENAI_ENDPOINT

dataset:
  path: dataset.yaml

run:
  concurrency: 4
  cache: true
  cache_dir: .agent-eval/cache
  runs_dir: runs
  regression_delta: 0.05
  budget:
    max_usd: 2.0
    max_cases: null
    max_judge_calls: null
    assume_judge_call_usd: 0.002

metrics:
  - name: task_completion
    threshold: 0.7
    task: State what this agent must accomplish.
  - name: step_efficiency
    threshold: 0.6
  - name: plan_quality
    threshold: 0.6
  - name: plan_adherence
    threshold: 0.6
  - name: tool_correctness
    threshold: 1.0
    exact_match: true
    consider_ordering: true
    input_parameters: true
  - name: argument_correctness
    threshold: 0.7
  - name: answer_correctness
    type: geval
    threshold: 0.7
    criteria: >
      The actual output reaches the same logical conclusion as the expected output.
    evaluation_params:
      - input
      - actual_output
      - expected_output
  # - name: domain_check
  #   type: custom
  #   target: my_agent.metrics:build
  #   threshold: 1.0
"""

INIT_DATASET = """\
cases:
  - id: example
    input: What is the status of order 17?
    expected_output: Order 17 was delivered.
    tags: [smoke]
    expected_tools:
      - name: lookup_order
        arguments:
          order_id: "17"
    # turns:
    #   - role: user
    #     content: I need help with an order.
    #   - role: assistant
    #     content: What is the order id?
"""


def console() -> None:
    raise SystemExit(main())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Evaluate an AI agent once per case, then score every metric from the captured trace.",
    )
    parser.add_argument("--version", action="version", version=f"agent-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="write evals.yaml and dataset.yaml")
    init_parser.add_argument("--dir", default=".", help="directory to scaffold")

    run_parser = sub.add_parser("run", help="run the dataset and write runs/<id>/")
    run_parser.add_argument("--config", default="evals.yaml")
    run_parser.add_argument("--concurrency", type=int)
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.add_argument("--budget", type=float, help="override run.budget.max_usd")
    run_parser.add_argument("--max-cases", type=int)
    run_parser.add_argument("--case", action="append", dest="cases", help="run only this case id")
    run_parser.add_argument("--tag", action="append", dest="tags")
    run_parser.add_argument("--runs-dir")

    report_parser = sub.add_parser("report", help="print a run and refresh its HTML report")
    report_parser.add_argument("--run", help="run directory or results.json")
    report_parser.add_argument("--latest", action="store_true")
    report_parser.add_argument("--runs-dir", default="runs")
    report_parser.add_argument("--config", default="evals.yaml")

    compare_parser = sub.add_parser("compare", help="diff two runs and flag regressions")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--delta", type=float, default=None)
    compare_parser.add_argument("--config", default="evals.yaml")
    compare_parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="exit 0 even when scores regress",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _init(Path(args.dir))
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            return _report(args)
        if args.command == "compare":
            return _compare(args)
    except AgentEvalError as exc:
        print(f"agent-eval: {exc}", file=sys.stderr)
        return 1
    return 2


def _init(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "evals.yaml"
    dataset_path = directory / "dataset.yaml"
    wrote = False
    if config_path.exists():
        print(f"kept existing {config_path}")
    else:
        config_path.write_text(INIT_CONFIG, encoding="utf-8")
        print(f"wrote {config_path}")
        wrote = True
    if dataset_path.exists():
        print(f"kept existing {dataset_path}")
    else:
        dataset_path.write_text(INIT_DATASET, encoding="utf-8")
        print(f"wrote {dataset_path}")
        wrote = True
    if not wrote:
        print("nothing to scaffold")
    return 0


def _run(args: argparse.Namespace) -> int:
    from agent_eval.reporting.store import render_summary
    from agent_eval.runner import execute, exit_code

    record, destination = execute(
        Path(args.config),
        concurrency=args.concurrency,
        use_cache=False if args.no_cache else None,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        case_ids=set(args.cases) if args.cases else None,
        tags=set(args.tags) if args.tags else None,
        budget_usd=args.budget,
        max_cases=args.max_cases,
    )
    print(render_summary(record))
    print(f"\nwrote {destination / 'results.json'}")
    print(f"wrote {destination / 'report.html'}")
    return exit_code(record)


def _report(args: argparse.Namespace) -> int:
    from agent_eval.config import load_config, resolve_from
    from agent_eval.reporting.html import render_report
    from agent_eval.reporting.store import latest_run, load_run, render_summary

    if args.run:
        path = Path(args.run)
    elif args.latest:
        runs_dir = Path(args.runs_dir)
        config_path = Path(args.config)
        if config_path.is_file() and args.runs_dir == "runs":
            config = load_config(config_path)
            runs_dir = resolve_from(config_path, config.run.runs_dir)
        path = latest_run(runs_dir)
    else:
        raise AgentEvalError("report requires --run or --latest")
    record = load_run(path)
    directory = path if path.is_dir() else path.parent
    (directory / "report.html").write_text(render_report(record), encoding="utf-8")
    print(render_summary(record))
    print(f"\nwrote {directory / 'report.html'}")
    return 0 if record.status == "passed" else 1


def _compare(args: argparse.Namespace) -> int:
    from agent_eval.compare import compare_runs, render_compare_text
    from agent_eval.config import load_config
    from agent_eval.reporting.html import render_compare
    from agent_eval.reporting.store import load_run

    baseline = load_run(Path(args.baseline))
    candidate = load_run(Path(args.candidate))
    delta = args.delta
    if delta is None:
        config_path = Path(args.config)
        delta = load_config(config_path).run.regression_delta if config_path.is_file() else 0.05
    report = compare_runs(baseline, candidate, regression_delta=delta)
    print(render_compare_text(report))
    destination = Path(args.candidate)
    directory = destination if destination.is_dir() else destination.parent
    json_path = directory / f"compare-{baseline.run_id}.json"
    html_path = directory / f"compare-{baseline.run_id}.html"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_compare(report), encoding="utf-8")
    print(f"\nwrote {json_path}")
    print(f"wrote {html_path}")
    if report.regressions and not args.allow_regression:
        return 1
    return 0
