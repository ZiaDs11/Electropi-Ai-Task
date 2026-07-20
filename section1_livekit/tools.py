"""
tools.py
--------
Business logic for the agent's tools.

DESIGN: tools return STRUCTURED DATA (dataclasses -> dicts), never prose.

The business layer owns *facts* (status, eta_minutes, courier, currency amounts).
The LLM owns *natural language* (how to say it, in which language, at what
register). Mixing the two — e.g. returning "Order A1001 is out for delivery,
ETA 12 minutes" — pushes presentation into the domain layer and causes real
problems:

  * Localisation: the agent can't answer in Arabic without rewriting the tool.
    (For an Egypt/Gulf deployment this alone is decisive.)
  * Testability: assertions become brittle string matching instead of
    `assert result.eta_minutes == 12`.
  * Reuse: the same tool can't back a mobile UI, a webhook, or an SMS template.
  * Model freedom: the LLM can't reasonably summarise, compare, or combine two
    tool results if each arrives as a finished sentence.

Errors follow the same rule: a failure is *structured data* (`ok: False` plus a
machine-readable `error_code`), not an English apology. The LLM decides how to
apologise; the domain layer only states what went wrong.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OrderStatus(str, Enum):
    """Machine-readable states. `str` mixin so it serialises cleanly to JSON."""
    PREPARING = "preparing"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    ORDER_NOT_FOUND = "order_not_found"
    NOT_CANCELLABLE = "not_cancellable"


# --- Result envelope -------------------------------------------------------
# Every tool returns the same shape, so the agent layer never has to guess
# whether it got a success or a failure.

@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] | None = None
    error_code: str | None = None
    # `detail` is a terse machine-facing hint, NOT a user-facing sentence.
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Compact dict for the LLM: drop null keys to save tokens."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def success(cls, **data: Any) -> "ToolResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: ErrorCode, detail: str) -> "ToolResult":
        return cls(ok=False, error_code=code.value, detail=detail)


@dataclass
class Order:
    order_id: str
    status: OrderStatus
    eta_minutes: int | None = None
    courier: str | None = None
    total_egp: float | None = None
    items: list[str] = field(default_factory=list)


# ---- Mocked "database" -----------------------------------------------------
# In a real system this is a call to the orders microservice.
_ORDERS: dict[str, Order] = {
    "A1001": Order("A1001", OrderStatus.OUT_FOR_DELIVERY, eta_minutes=12,
                   courier="Kareem", total_egp=185.0, items=["Koshari", "Cola"]),
    "A1002": Order("A1002", OrderStatus.PREPARING, eta_minutes=30,
                   total_egp=240.5, items=["Shawarma wrap", "Fries"]),
    "A1003": Order("A1003", OrderStatus.DELIVERED, eta_minutes=0,
                   courier="Mona", total_egp=95.0, items=["Feteer"]),
}

# Business rule lives here, not in the prompt: only these states can be cancelled.
_CANCELLABLE = {OrderStatus.PREPARING}


def _normalise(order_id: str) -> str:
    return order_id.strip().upper()


async def get_order_status(order_id: str) -> ToolResult:
    """Look up an order. Returns structured facts, not a sentence.

    Args:
        order_id: The order reference, e.g. "A1001".

    Returns:
        ToolResult with data {order_id, status, eta_minutes, courier, ...}
        or ok=False with error_code="order_not_found".
    """
    await asyncio.sleep(0.2)  # simulate network/DB latency

    order = _ORDERS.get(_normalise(order_id))
    if order is None:
        return ToolResult.failure(
            ErrorCode.ORDER_NOT_FOUND, f"no order with id {_normalise(order_id)}"
        )

    return ToolResult.success(
        order_id=order.order_id,
        status=order.status.value,
        eta_minutes=order.eta_minutes,
        courier=order.courier,
        total_egp=order.total_egp,
        items=order.items,
    )


async def cancel_order(order_id: str) -> ToolResult:
    """Cancel an order if its state permits it.

    Returns:
        ToolResult with data {order_id, status, refund_egp} on success, or
        ok=False with error_code in {order_not_found, not_cancellable}.
        On NOT_CANCELLABLE we still return the current status so the LLM can
        explain *why* without making a second tool call.
    """
    await asyncio.sleep(0.2)

    oid = _normalise(order_id)
    order = _ORDERS.get(oid)
    if order is None:
        return ToolResult.failure(
            ErrorCode.ORDER_NOT_FOUND, f"no order with id {oid}"
        )

    if order.status not in _CANCELLABLE:
        return ToolResult(
            ok=False,
            error_code=ErrorCode.NOT_CANCELLABLE.value,
            detail=f"state is {order.status.value}",
            data={"order_id": oid, "status": order.status.value,
                  "cancellable_states": [s.value for s in _CANCELLABLE]},
        )

    order.status = OrderStatus.CANCELLED
    return ToolResult.success(
        order_id=oid,
        status=order.status.value,
        refund_egp=order.total_egp,
    )
