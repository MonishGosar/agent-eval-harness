"""Turn one captured agent result into the nested span tree DeepEval trajectory metrics read."""

from __future__ import annotations

from agent_eval.models import AgentResult, Golden


def build_trace(case: Golden, result: AgentResult) -> dict[str, object]:
    """Match the shape of TraceManager.create_nested_spans_dict: name, type, input, output, children.

    Published DeepEval builds still store this on LLMTestCase._trace_dict. The scorer writes both
    the private attribute and the public trace_dict field when that field exists, so one captured
    trace feeds TaskCompletion, StepEfficiency, PlanQuality, and PlanAdherence without re-running.
    """
    children: list[dict[str, object]] = []
    if result.plan:
        children.append(
            {
                "name": "plan",
                "type": "llm",
                "input": case.rendered_input(),
                "output": "\n".join(result.plan),
                "children": [],
            }
        )
    for call in result.tool_calls:
        children.append(
            {
                "name": call.name,
                "type": "tool",
                "input": call.arguments,
                "output": call.output,
                "reasoning": call.reasoning,
                "children": [],
            }
        )
    children.append(
        {
            "name": "final_answer",
            "type": "llm",
            "input": case.rendered_input(),
            "output": result.output,
            "children": [],
        }
    )
    if result.error:
        children.append(
            {
                "name": "error",
                "type": "custom",
                "input": case.input,
                "output": result.error,
                "children": [],
            }
        )
    return {
        "name": "agent",
        "type": "agent",
        "input": case.rendered_input(),
        "output": result.output,
        "plan": list(result.plan),
        "children": children,
    }
