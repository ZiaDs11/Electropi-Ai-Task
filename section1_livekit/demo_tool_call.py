"""
demo_tool_call.py
-----------------
Proves the LLM actually invokes the tool, using text I/O in place of STT/TTS.
This is the "mock STT/TTS, real LLM + tool-calling" path from the test.

It talks to any OpenAI-compatible endpoint via the OPENAI_* env vars:
  - Real OpenAI:   export OPENAI_API_KEY=sk-...
  - Local (free):  run Ollama/vLLM, then
        export OPENAI_BASE_URL=http://localhost:11434/v1
        export OPENAI_API_KEY=ollama
        export LLM_MODEL=llama3.2:3b

If NO endpoint is configured, it falls back to a tiny offline "router" so the
tool-call flow is still demonstrable end-to-end (clearly labelled STUB). The
tool functions themselves are always the real ones from tools.py.

Usage:
    python demo_tool_call.py "where is order A1001?"
    python demo_tool_call.py            # runs 3 scripted turns
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import tools as biz
from env_loader import load_env

load_env()  # read repo-root .env before any os.getenv() below

# ---- Tool schemas exposed to the LLM (JSON-schema, vendor-neutral) ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "e.g. A1001"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order if it is still being prepared.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

# Map tool name -> real coroutine. The LLM decides; we execute the real code.
DISPATCH = {
    "get_order_status": biz.get_order_status,
    "cancel_order": biz.cancel_order,
}

SYSTEM = (
    "You are a concise support assistant for a food-delivery app. "
    "Call a tool to look things up instead of guessing. "
    "Tools return structured JSON facts, not sentences — turn them into a "
    "short natural reply in the customer's language. Never read JSON or error "
    "codes aloud. If ok=false, explain kindly and suggest a next step. "
    "State amounts and times exactly as returned; invent nothing."
)


async def _run_tool(name: str, args: dict) -> str:
    """Execute the tool and serialise its STRUCTURED result to JSON.

    The tool returns facts (a dict); we hand that JSON to the LLM, which is
    what turns it into a sentence. Nothing here formats prose.
    """
    result = await DISPATCH[name](**args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


async def chat_once_openai(user_text: str) -> None:
    """Real path: OpenAI-compatible function calling."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]

    # Round 1: let the model decide to call a tool.
    r1 = await client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SCHEMAS
    )
    msg = r1.choices[0].message
    if not msg.tool_calls:
        print(f"[assistant] {msg.content}")
        return

    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        print(f"  >> LLM invoked {tc.function.name}({args})")
        result = await _run_tool(tc.function.name, args)
        print(f"  << tool returned: {result}")
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )

    # Round 2: model turns the tool result into a natural reply.
    r2 = await client.chat.completions.create(model=model, messages=messages)
    print(f"[assistant] {r2.choices[0].message.content}")


def _stub_render(res: dict) -> str:
    """Stand-in for the LLM's natural-language generation (offline mode only).

    In production this function does not exist — the LLM does this job. It
    lives in the DEMO layer, never in tools.py, which is exactly the
    separation of concerns we want.
    """
    if not res.get("ok"):
        code = res.get("error_code")
        if code == "order_not_found":
            return "I couldn't find that order. Could you double-check the number?"
        if code == "not_cancellable":
            st = res.get("data", {}).get("status", "").replace("_", " ")
            return f"Sorry, that order is already {st}, so it can't be cancelled."
        return "Sorry, something went wrong with that request."

    d = res.get("data", {})
    if "refund_egp" in d:
        return (f"Order {d['order_id']} is cancelled. "
                f"You'll be refunded {d['refund_egp']:.2f} EGP.")
    status = d.get("status")
    if status == "delivered":
        return f"Order {d['order_id']} was delivered by {d.get('courier')}."
    if status == "preparing":
        return (f"Order {d['order_id']} is being prepared — "
                f"about {d.get('eta_minutes')} minutes to go.")
    return (f"Order {d['order_id']} is on its way with {d.get('courier')}, "
            f"arriving in about {d.get('eta_minutes')} minutes.")


async def chat_once_stub(user_text: str) -> None:
    """Offline STUB path (no keys): naive intent+entity extraction so the
    tool-call flow still runs. NOT a real LLM — for demo continuity only."""
    import re

    print("  [STUB LLM — no OPENAI_* configured; using keyword router]")
    m = re.search(r"[Aa]\d{4}", user_text)
    order_id = m.group(0) if m else "A1001"
    name = "cancel_order" if "cancel" in user_text.lower() else "get_order_status"
    print(f"  >> LLM invoked {name}({{'order_id': '{order_id}'}})")
    result_json = await _run_tool(name, {"order_id": order_id})
    print(f"  << tool returned (structured): {result_json}")
    # A real LLM writes this sentence from the JSON. The stub does a crude
    # render just so the demo is readable offline — this templating is
    # deliberately NOT in the tool layer.
    print(f"[assistant] {_stub_render(json.loads(result_json))}")


async def handle(user_text: str) -> None:
    print(f"\n[user] {user_text}")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        try:
            await chat_once_openai(user_text)
            return
        except Exception as e:  # pragma: no cover - network/dep issues
            print(f"  (openai path failed: {e}; falling back to stub)")
    await chat_once_stub(user_text)


async def main() -> None:
    if len(sys.argv) > 1:
        await handle(" ".join(sys.argv[1:]))
        return
    for turn in [
        "Hi, where is my order A1001?",
        "Can you cancel order A1002?",
        "What about A1003?",
    ]:
        await handle(turn)


if __name__ == "__main__":
    asyncio.run(main())
