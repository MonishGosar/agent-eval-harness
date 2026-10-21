from examples.refund_agent.agent import BASELINE_PROMPT, IMPROVED_PROMPT
from examples.refund_agent.metrics import TicketGateMetric
from examples.refund_agent.policy import issue_ticket, purchase_history, quote_refund


def test_quotes_cover_the_seven_situations() -> None:
    assert quote_refund("u_alice", "headphones")["eligible"] is True
    assert quote_refund("u_alice", "headphones")["amount"] == 80.0
    assert quote_refund("u_bob", "mouse")["amount"] == 25.0
    assert quote_refund("u_alice", "keyboard")["eligible"] is False
    assert quote_refund("u_cara", "monitor")["eligible"] is False
    assert quote_refund("u_alice", "tablet")["reason"] == "item was never purchased"
    assert quote_refund("u_dana", "headphones")["reason"] == "unknown user"
    assert purchase_history("u_dana")["found"] is False
    history = purchase_history("u_alice")
    headphones = next(row for row in history["purchases"] if row["item"] == "headphones")
    keyboard = next(row for row in history["purchases"] if row["item"] == "keyboard")
    assert headphones["within_window"] is True
    assert keyboard["within_window"] is False


def test_ticket_gate_refuses_ineligible_items() -> None:
    refused = issue_ticket("u_alice", ["keyboard"], 120.0)
    assert refused["ok"] is False
    wrong_total = issue_ticket("u_alice", ["headphones"], 1.0)
    assert wrong_total["ok"] is False
    opened = issue_ticket("u_alice", ["headphones"], 80.0)
    assert opened["status"] == "pending_human_approval"
    assert opened["ticket_id"].startswith("TCK-")

    class _Call:
        def __init__(self, arguments: dict[str, object]) -> None:
            self.name = "issue_refund_ticket"
            self.input_parameters = arguments

    class _Case:
        def __init__(self, arguments: dict[str, object]) -> None:
            self.tools_called = [_Call(arguments)]

    metric = TicketGateMetric(threshold=1)
    assert metric.measure(_Case({"user_id": "u_alice", "items": ["headphones"], "amount": 80})) == 1
    assert metric.measure(_Case({"user_id": "u_alice", "items": ["keyboard"], "amount": 120})) == 0


def test_prompts_encode_the_before_and_after() -> None:
    assert "Do not call get_today_date" in IMPROVED_PROMPT
    assert "within_window true" in IMPROVED_PROMPT
    assert "Always call get_today_date" in BASELINE_PROMPT
    assert "every item the user mentions" in BASELINE_PROMPT
