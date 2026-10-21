from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from agent_eval.models import AgentResult, ExpectedTool, Golden, ToolCallRecord
from agent_eval.scoring import to_test_case
from agent_eval.trace import build_trace
from tests.scripted_judge import ScriptedJudge


def _case() -> Golden:
    return Golden(
        id="full_refund",
        input="refund headphones",
        expected_output="refund $80",
        expected_tools=[
            ExpectedTool(name="get_user_purchase_history", arguments={"user_id": "u_alice"}),
        ],
    )


def test_trace_contains_plan_and_tools() -> None:
    result = AgentResult(
        output="refunded",
        plan=["Plan: look up history"],
        tool_calls=[
            ToolCallRecord(name="get_user_purchase_history", arguments={"user_id": "u_alice"})
        ],
    )
    trace = build_trace(_case(), result)
    assert trace["type"] == "agent"
    names = [child["name"] for child in trace["children"]]
    assert names == ["plan", "get_user_purchase_history", "final_answer"]
    test_case = to_test_case(_case(), result, "resolve the refund")
    assert test_case._trace_dict["plan"] == ["Plan: look up history"]
    assert test_case.tools_called[0].input_parameters == {"user_id": "u_alice"}


def test_exact_tool_match_rejects_extra_date_lookup() -> None:
    judge = ScriptedJudge()
    metric = ToolCorrectnessMetric(
        threshold=1,
        model=judge,
        async_mode=False,
        verbose_mode=False,
        include_reason=True,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
        should_consider_ordering=True,
    )
    expected = [ToolCall(name="get_user_purchase_history", input_parameters={"user_id": "u_alice"})]
    extra = LLMTestCase(
        input="refund",
        actual_output="ok",
        tools_called=[
            ToolCall(name="get_today_date", input_parameters={}),
            ToolCall(name="get_user_purchase_history", input_parameters={"user_id": "u_alice"}),
        ],
        expected_tools=expected,
    )
    assert metric.measure(extra, _show_indicator=False) == 0

    metric = ToolCorrectnessMetric(
        threshold=1,
        model=judge,
        async_mode=False,
        verbose_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
    )
    match = LLMTestCase(
        input="refund",
        actual_output="ok",
        tools_called=[
            ToolCall(name="get_user_purchase_history", input_parameters={"user_id": "u_alice"})
        ],
        expected_tools=expected,
    )
    assert metric.measure(match, _show_indicator=False) == 1

    wrong = LLMTestCase(
        input="refund",
        actual_output="ok",
        tools_called=[
            ToolCall(name="get_user_purchase_history", input_parameters={"user_id": "someone"})
        ],
        expected_tools=expected,
    )
    metric = ToolCorrectnessMetric(
        threshold=1,
        model=judge,
        async_mode=False,
        verbose_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
    )
    assert metric.measure(wrong, _show_indicator=False) == 0


def test_empty_tool_lists_score_one() -> None:
    metric = ToolCorrectnessMetric(
        threshold=1,
        model=ScriptedJudge(),
        async_mode=False,
        verbose_mode=False,
        should_exact_match=True,
    )
    case = LLMTestCase(
        input="What is the policy?",
        actual_output="30 days.",
        tools_called=[],
        expected_tools=[],
    )
    assert metric.measure(case, _show_indicator=False) == 1.0
