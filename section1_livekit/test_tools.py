"""
test_tools.py
-------------
Unit tests for the tool layer.

These are the payoff of returning structured data: we assert on FACTS
(`eta_minutes == 12`, `error_code == "order_not_found"`) instead of doing
brittle substring matching on English sentences. If the wording of the agent's
reply changes, these tests keep passing — as they should, because the wording
is the LLM's job, not the domain layer's.

Run:  python test_tools.py     (or: pytest test_tools.py)
"""

import asyncio

import tools as biz


async def test_get_order_status_success():
    r = await biz.get_order_status("A1001")
    assert r.ok is True
    assert r.data["order_id"] == "A1001"
    assert r.data["status"] == biz.OrderStatus.OUT_FOR_DELIVERY.value
    assert r.data["eta_minutes"] == 12
    assert r.data["courier"] == "Kareem"
    # No prose anywhere in the payload:
    assert "minutes to go" not in str(r.data)


async def test_order_id_is_normalised():
    r = await biz.get_order_status("  a1001 ")
    assert r.ok and r.data["order_id"] == "A1001"


async def test_unknown_order_returns_structured_error():
    r = await biz.get_order_status("A9999")
    assert r.ok is False
    assert r.error_code == biz.ErrorCode.ORDER_NOT_FOUND.value
    assert r.data is None


async def test_cancel_allowed_when_preparing():
    r = await biz.cancel_order("A1002")
    assert r.ok is True
    assert r.data["status"] == biz.OrderStatus.CANCELLED.value
    assert r.data["refund_egp"] == 240.5


async def test_cancel_rejected_when_out_for_delivery():
    r = await biz.cancel_order("A1001")
    assert r.ok is False
    assert r.error_code == biz.ErrorCode.NOT_CANCELLABLE.value
    # Error still carries context so the LLM can explain why without a 2nd call:
    assert r.data["status"] == biz.OrderStatus.OUT_FOR_DELIVERY.value
    assert "preparing" in r.data["cancellable_states"]


async def test_result_dict_is_json_safe_and_compact():
    import json

    r = await biz.get_order_status("A1003")
    d = r.to_dict()
    json.dumps(d)               # must not raise
    assert "error_code" not in d  # null keys dropped to save tokens


# --- Tool-call recovery ----------------------------------------------------
# Small/local models sometimes emit a tool call as TEXT instead of using the
# API's structured `tool_calls` field. Unhandled, the customer sees raw JSON and
# the tool never runs. These lock in the repair logic.

async def test_recovers_malformed_text_tool_call():
    from conversation import _extract_text_tool_call as ex
    # Observed in a real Ollama run: stray ')' after the name, '$' before a key.
    bad = '{"name":"cancel_order)","parameters":{$"order_id":"A1002"}}}'
    assert ex(bad) == ("cancel_order", {"order_id": "A1002"})


async def test_recovers_fenced_and_variant_key_names():
    from conversation import _extract_text_tool_call as ex
    assert ex('```json\n{"name":"cancel_order","arguments":{"order_id":"A1002"}}\n```') \
        == ("cancel_order", {"order_id": "A1002"})
    assert ex('Sure! {"tool":"get_order_status","args":{"order_id":"A1003"}}') \
        == ("get_order_status", {"order_id": "A1003"})


async def test_normal_reply_is_not_treated_as_tool_call():
    from conversation import _extract_text_tool_call as ex
    assert ex("Your order A1001 is out for delivery, arriving in 12 minutes.") is None
    assert ex("I can help you cancel_order if you tell me the number.") is None
    assert ex("") is None


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        await t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    asyncio.run(main())
