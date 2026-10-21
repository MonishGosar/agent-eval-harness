"""LangGraph refund agent.

`variant="improved"` is the spec the goldens describe.
`variant="baseline"` reproduces the tutorial's loose habits: always look up today's date,
quote every mentioned item, and pass whatever identifier is handy.
"""

from __future__ import annotations

import os
from typing import Any

from examples.refund_agent.policy import POLICY_TEXT
from pydantic import BaseModel, Field

IMPROVED_PROMPT = f"""You are a refund agent for an online store.

Policy: {POLICY_TEXT}

How to work:
- The user message names a user id. Pass that exact id to tools.
- Call get_user_purchase_history once when a refund depends on what the user bought.
- History rows already include purchase_date, days_since_purchase, and within_window.
  Do not call get_today_date.
- Call calculate_refund(user_id, item) only for a requested item whose history row has within_window true.
  Do not call it for items that were never purchased or are outside the window.
- Call issue_refund_ticket only when at least one requested item is eligible.
  Pass exactly those items and the sum of the quoted amounts.
  Tell the user a person still has to approve the ticket. Do not say the money has moved.
- If the user is unknown, say so. Do not calculate a refund and do not open a ticket.
- If the user only asks about the policy, answer from the policy above and call no tools.
- Before the first tool call, write a one-line plan that starts with "Plan:" and then follow it.
  Do not add tool calls that the plan did not list.

The final answer states the decision, the amount if any, and whether a human must approve a ticket.
"""

BASELINE_PROMPT = f"""You are a refund agent for an online store.

Policy: {POLICY_TEXT}

How to work:
- Always call get_today_date before deciding, even when purchase records include dates.
- Call calculate_refund for every item the user mentions, including items that may not be in the history.
- Pass whatever identifier you have (a name or an id) to the tools.
- If any quote looks eligible, call issue_refund_ticket for all mentioned items.
- Before tools, write a one-line plan that starts with "Plan:".
"""


class RefundAnswer(BaseModel):
    answer: str = Field(description="Decision the customer should read")
    conversation: str = Field(description="One short follow-up or confirmation")


def prompt_for(variant: str) -> str:
    if variant == "baseline":
        return BASELINE_PROMPT
    if variant == "improved":
        return IMPROVED_PROMPT
    raise ValueError(f"unknown variant {variant!r}")


def build_agent(variant: str = "improved") -> Any:
    """Return a compiled LangGraph agent. Requires OPENAI_API_KEY when it is invoked."""
    from examples.refund_agent.tools import TOOLS
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model_name = os.environ.get("REFUND_AGENT_MODEL", "gpt-4.1-mini")
    model = ChatOpenAI(model=model_name, temperature=0)
    return create_react_agent(
        model,
        TOOLS,
        prompt=prompt_for(variant),
        response_format=RefundAnswer,
    )
