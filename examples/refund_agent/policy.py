"""Refund policy used by the example tools. The model does not own dates or prices."""

from __future__ import annotations

import hashlib
from datetime import date

REFUND_WINDOW_DAYS = 30
POLICY_TODAY = date(2026, 10, 21)

POLICY_TEXT = (
    "Refunds are available for 30 days after purchase. "
    "Only items the customer purchased and explicitly requested can be refunded. "
    "A refund ticket waits for a person; it does not move money."
)


class Purchase:
    def __init__(self, item: str, price: float, purchased_on: date, order_id: str) -> None:
        self.item = item
        self.price = price
        self.purchased_on = purchased_on
        self.order_id = order_id


PURCHASES: dict[str, list[Purchase]] = {
    "u_alice": [
        Purchase("headphones", 80.00, date(2026, 10, 1), "A100"),
        Purchase("keyboard", 120.00, date(2026, 8, 1), "A090"),
    ],
    "u_bob": [
        Purchase("laptop sleeve", 30.00, date(2026, 10, 10), "B200"),
        Purchase("mouse", 25.00, date(2026, 10, 5), "B201"),
    ],
    "u_cara": [
        Purchase("monitor", 300.00, date(2026, 9, 1), "C300"),
    ],
}


def age_days(purchased_on: date, today: date = POLICY_TODAY) -> int:
    return (today - purchased_on).days


def within_window(purchased_on: date, today: date = POLICY_TODAY) -> bool:
    age = age_days(purchased_on, today)
    return 0 <= age <= REFUND_WINDOW_DAYS


def find_purchase(user_id: str, item: str) -> Purchase | None:
    for purchase in PURCHASES.get(user_id, []):
        if purchase.item.lower() == item.strip().lower():
            return purchase
    return None


def purchase_history(user_id: str) -> dict[str, object]:
    rows = PURCHASES.get(user_id)
    if rows is None:
        return {"found": False, "user_id": user_id, "purchases": []}
    purchases = []
    for purchase in rows:
        purchases.append(
            {
                "item": purchase.item,
                "price": purchase.price,
                "order_id": purchase.order_id,
                "purchase_date": purchase.purchased_on.isoformat(),
                "days_since_purchase": age_days(purchase.purchased_on),
                "within_window": within_window(purchase.purchased_on),
            }
        )
    return {"found": True, "user_id": user_id, "purchases": purchases}


def quote_refund(user_id: str, item: str) -> dict[str, object]:
    """Server-side quote. The caller cannot supply a price or a date."""
    if user_id not in PURCHASES:
        return {
            "user_id": user_id,
            "item": item,
            "eligible": False,
            "amount": 0.0,
            "reason": "unknown user",
        }
    purchase = find_purchase(user_id, item)
    if purchase is None:
        return {
            "user_id": user_id,
            "item": item,
            "eligible": False,
            "amount": 0.0,
            "reason": "item was never purchased",
        }
    if not within_window(purchase.purchased_on):
        return {
            "user_id": user_id,
            "item": item,
            "eligible": False,
            "amount": 0.0,
            "reason": (
                f"purchased on {purchase.purchased_on.isoformat()}, "
                f"{age_days(purchase.purchased_on)} days before {POLICY_TODAY.isoformat()}, "
                f"outside the {REFUND_WINDOW_DAYS}-day window"
            ),
        }
    return {
        "user_id": user_id,
        "item": purchase.item,
        "eligible": True,
        "amount": purchase.price,
        "reason": (
            f"purchased on {purchase.purchased_on.isoformat()}, "
            f"within {REFUND_WINDOW_DAYS} days of {POLICY_TODAY.isoformat()}"
        ),
    }


def issue_ticket(user_id: str, items: list[str], amount: float) -> dict[str, object]:
    """Open a human-approval ticket. Ineligible items and wrong totals are refused."""
    if not items:
        return {"ok": False, "reason": "no items"}
    quotes = [quote_refund(user_id, item) for item in items]
    blocked = [quote for quote in quotes if not quote["eligible"]]
    if blocked:
        reasons = "; ".join(f"{quote['item']}: {quote['reason']}" for quote in blocked)
        return {"ok": False, "reason": f"refused: {reasons}", "status": "not_issued"}
    total = round(sum(float(quote["amount"]) for quote in quotes), 2)
    if abs(total - float(amount)) > 0.01:
        return {
            "ok": False,
            "reason": f"refused: amount {amount} does not match quoted total {total}",
            "status": "not_issued",
        }
    digest = hashlib.sha256(f"{user_id}|{','.join(items)}|{total:.2f}".encode()).hexdigest()[:8]
    return {
        "ok": True,
        "ticket_id": f"TCK-{digest}",
        "user_id": user_id,
        "items": [str(quote["item"]) for quote in quotes],
        "amount": total,
        "status": "pending_human_approval",
    }


def today() -> dict[str, str]:
    return {"today": POLICY_TODAY.isoformat()}
