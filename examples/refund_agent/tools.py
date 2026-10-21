"""LangChain tools. They delegate to policy.py so the model cannot invent prices or dates."""

from __future__ import annotations

import json

from examples.refund_agent.policy import issue_ticket, purchase_history, quote_refund, today
from langchain_core.tools import tool


@tool
def get_user_purchase_history(user_id: str) -> str:
    """Return a user's purchases, including purchase_date, days_since_purchase, and within_window."""
    return json.dumps(purchase_history(user_id))


@tool
def calculate_refund(user_id: str, item: str) -> str:
    """Quote a refund for one purchased item. Price and eligibility come from the purchase record."""
    return json.dumps(quote_refund(user_id, item))


@tool
def issue_refund_ticket(user_id: str, items: list[str], amount: float) -> str:
    """Open a ticket for a human to approve. Refuses ineligible items. Does not move money."""
    return json.dumps(issue_ticket(user_id, items, amount))


@tool
def get_today_date() -> str:
    """Return the policy clock. Prefer purchase_date and within_window from purchase history instead."""
    return json.dumps(today())


TOOLS = [
    get_user_purchase_history,
    calculate_refund,
    issue_refund_ticket,
    get_today_date,
]
