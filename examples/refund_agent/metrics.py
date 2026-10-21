"""Deterministic custom metric: a ticket may only contain items the policy would refund."""

from __future__ import annotations

from typing import Any

from examples.refund_agent.policy import issue_ticket


class TicketGateMetric:
    """Scores 1 when every issue_refund_ticket call would be accepted by the policy gate."""

    def __init__(self, threshold: float = 1.0, model: Any | None = None, **params: Any) -> None:
        del model, params
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: Any, **kwargs: Any) -> float:
        del kwargs
        failures: list[str] = []
        tickets = 0
        for call in getattr(test_case, "tools_called", None) or []:
            if getattr(call, "name", "") != "issue_refund_ticket":
                continue
            tickets += 1
            arguments = getattr(call, "input_parameters", None) or {}
            user_id = str(arguments.get("user_id", ""))
            items = arguments.get("items") or []
            if isinstance(items, str):
                items = [items]
            amount = arguments.get("amount", 0)
            try:
                decision = issue_ticket(user_id, list(items), float(amount))
            except (TypeError, ValueError) as exc:
                failures.append(str(exc))
                continue
            if not decision.get("ok"):
                failures.append(str(decision.get("reason")))
        if failures:
            self.score = 0.0
            self.reason = "ticket gate refused the call: " + "; ".join(failures)
        else:
            self.score = 1.0
            self.reason = (
                "no refund ticket was issued"
                if tickets == 0
                else f"{tickets} ticket call(s) match eligible items and the quoted total"
            )
        self.success = self.score >= self.threshold
        return self.score

    @property
    def __name__(self) -> str:
        return "Ticket Gate"


def build(threshold: float = 1.0, model: Any | None = None, **params: Any) -> TicketGateMetric:
    return TicketGateMetric(threshold=threshold, model=model, **params)
