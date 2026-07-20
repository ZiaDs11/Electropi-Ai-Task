"""
conversation.py
---------------
Multi-turn conversational demo of the same SupportAgent tools, with:

  1. CONVERSATION MEMORY — a single `messages` list is carried across turns, so
     follow-ups resolve from context ("What about A1003?" after an order
     question). Each turn appends: user -> assistant(tool_call) -> tool ->
     assistant(final). That full history is what makes the sequence coherent.

  2. INTERACTIVE MODE — `--chat` lets you type turns yourself.

  3. TEXT-TOOL-CALL RECOVERY — some models (esp. small local ones via Ollama)
     don't return a structured `tool_calls` field; they emit the call as TEXT in
     the content, e.g.  {"name":"cancel_order","parameters":{"order_id":"A1002"}}
     Left unhandled, the user literally sees raw JSON as the reply and the tool
     never runs. `_extract_text_tool_call()` detects that, repairs common
     malformations, executes the real tool, and feeds the result back — so the
     conversation recovers instead of leaking JSON to the customer.

Run:
    python conversation.py            # scripted 4-turn conversation
    python conversation.py --chat     # interactive REPL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any

import tools as biz
from env_loader import load_env

load_env()

# --------------------------------------------------------------------------- #
# Tool registry: schema (for the LLM) + dispatch (for us). One source of truth.
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status, ETA, courier, total and "
                           "items of a customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string",
                                 "description": "Order reference, e.g. A1001"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order. Only succeeds while the order is "
                           "still being prepared.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

DISPATCH = {
    "get_order_status": biz.get_order_status,
    "cancel_order": biz.cancel_order,
}

# The persona drives the *style* of the reply: mention the concrete facts, then
# offer a relevant next step. Facts come from tools; wording comes from here.
SYSTEM = (
    "You are a warm, concise voice support assistant for a food-delivery app "
    "operating in Cairo.\n"
    "- Always call a tool to look up order facts; never guess an order detail.\n"
    "- Tools return structured JSON. Turn it into natural speech: mention the "
    "status, ETA, courier, total (as EGP) and items when they are present.\n"
    "- Never read JSON, field names, or error codes aloud.\n"
    "- End with a short, relevant follow-up offer (e.g. track, modify, cancel, "
    "rate, request a refund) that fits the order's current state.\n"
    "- If a result has ok=false, explain kindly why and suggest what to do next.\n"
    "- Reply in the customer's language. State amounts and times exactly as "
    "returned; invent nothing.\n"
    "- Keep it to 2-3 sentences."
)


# --------------------------------------------------------------------------- #
# Tool execution — always returns structured JSON (never prose)
# --------------------------------------------------------------------------- #
async def run_tool(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"ok": False, "error_code": "unknown_tool",
                           "detail": name})
    result = await fn(**args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Recovery for models that emit tool calls as TEXT instead of using the API field
# --------------------------------------------------------------------------- #
_TOOL_NAMES = "|".join(DISPATCH)


def _extract_text_tool_call(text: str) -> tuple[str, dict] | None:
    """Detect a tool call embedded in assistant *content* and repair it.

    Handles the real-world malformations small models produce, e.g.
        {"name":"cancel_order)","parameters":{$"order_id":"A1002"}}}
    (stray ')' after the name, a '$' before a key, unbalanced braces).

    Returns (tool_name, args) or None if the text is a normal reply.
    """
    if not text:
        return None
    # Cheap gate: must look like JSON *and* name one of our tools.
    if not re.search(_TOOL_NAMES, text) or "{" not in text:
        return None

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r"[$]", "", cleaned)          # drop stray '$'
    cleaned = re.sub(r'([A-Za-z_]+)\)"', r'\1"', cleaned)  # `cancel_order)"` -> `cancel_order"`

    # Trim trailing garbage braces, then try to parse progressively.
    for candidate in (cleaned, cleaned.rstrip("}") + "}}", cleaned.rstrip("}") + "}"):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
        if isinstance(name, str):
            name = name.strip(" )\"'")
            if name in DISPATCH and isinstance(args, dict):
                return name, args

    # Last resort: regex the tool name + an order id out of the text.
    m_name = re.search(_TOOL_NAMES, cleaned)
    m_id = re.search(r"[A-Za-z]\d{4}", cleaned)
    if m_name and m_id:
        return m_name.group(0), {"order_id": m_id.group(0)}
    return None


# --------------------------------------------------------------------------- #
# One conversational turn, against a real OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #
async def turn_openai(messages: list[dict], user_text: str) -> None:
    """Append one full turn to `messages`, executing tools as needed."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    messages.append({"role": "user", "content": user_text})

    # Round 1 — the model may choose a tool. Full history is sent every time,
    # which is what lets "What about A1003?" resolve.
    r1 = await client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SCHEMAS, temperature=0.3
    )
    msg = r1.choices[0].message
    tool_calls = msg.tool_calls or []

    # --- Path A: proper structured tool call -------------------------------
    if tool_calls:
        messages.append(msg.model_dump(exclude_none=True))
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  >> LLM invoked {name}({args})")
            out = await run_tool(name, args)
            print(f"  << tool returned: {out}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": name, "content": out})

    # --- Path B: model emitted the call as TEXT (the bug we're fixing) ------
    else:
        recovered = _extract_text_tool_call(msg.content or "")
        if recovered is None:
            # Genuine plain reply, no tool needed.
            messages.append({"role": "assistant", "content": msg.content})
            print(f"[assistant] {msg.content}")
            return

        name, args = recovered
        print(f"  !! model emitted a tool call as text; recovered it")
        print(f"  >> LLM invoked {name}({args})")
        out = await run_tool(name, args)
        print(f"  << tool returned: {out}")
        # Feed the result back as a normal user-visible observation so models
        # without tool-role support can still use it.
        messages.append({"role": "assistant",
                         "content": f"(calling {name})"})
        messages.append({"role": "user",
                         "content": f"Tool {name} returned: {out}\n"
                                    f"Reply to the customer in natural language "
                                    f"using these facts. Do not output JSON."})

    # Round 2 — natural-language reply built from the structured facts.
    r2 = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.3
    )
    final = r2.choices[0].message.content

    # Safety net: if the model *still* leaked JSON, don't show it to the user.
    if _extract_text_tool_call(final or "") is not None or (final or "").strip().startswith("{"):
        final = ("Sorry — I had trouble reading that back. Could you repeat "
                 "your request?")
    messages.append({"role": "assistant", "content": final})
    print(f"[assistant] {final}")


# --------------------------------------------------------------------------- #
# Offline stub path (no LLM configured) — keeps the same turn structure
# --------------------------------------------------------------------------- #
def _stub_render(res: dict) -> str:
    """Stand-in for the LLM's wording when no endpoint is configured.
    Lives in the demo layer, never in tools.py."""
    if not res.get("ok"):
        code = res.get("error_code")
        d = res.get("data", {}) or {}
        if code == "order_not_found":
            return ("I couldn't find that order number. Could you double-check "
                    "it for me?")
        if code == "not_cancellable":
            st = str(d.get("status", "")).replace("_", " ")
            return (f"Sorry, order {d.get('order_id')} is already {st}, so it "
                    f"can't be cancelled. Would you like to contact the courier "
                    f"instead?")
        return "Sorry, something went wrong with that request."

    d = res.get("data", {})
    if "refund_egp" in d:
        return (f"Order {d['order_id']} has been cancelled and you'll be "
                f"refunded {d['refund_egp']:.2f} EGP. Anything else I can help "
                f"with?")

    items = ", ".join(d.get("items") or [])
    status = d.get("status")
    if status == "delivered":
        return (f"Your order {d['order_id']} has been delivered by "
                f"{d.get('courier')}. The total was {d.get('total_egp'):.2f} EGP "
                f"for {items}. Would you like to rate your experience or request "
                f"a refund?")
    if status == "preparing":
        return (f"Your order {d['order_id']} is still being prepared — about "
                f"{d.get('eta_minutes')} minutes to go. It's {items}, totalling "
                f"{d.get('total_egp'):.2f} EGP. Would you like to modify or "
                f"cancel it?")
    return (f"Your order {d['order_id']} is out for delivery and should arrive "
            f"in about {d.get('eta_minutes')} minutes with {d.get('courier')}. "
            f"It's {items}, totalling {d.get('total_egp'):.2f} EGP. Would you "
            f"like to track it or modify the order?")


async def turn_stub(messages: list[dict], user_text: str) -> None:
    """Keyword router standing in for the LLM. Uses `messages` for context so
    a bare follow-up ('what about A1003?') still works."""
    messages.append({"role": "user", "content": user_text})

    m = re.search(r"[A-Za-z]\d{4}", user_text)
    if m:
        order_id = m.group(0)
    else:  # fall back to the last order id mentioned in the conversation
        prior = re.findall(r"[A-Za-z]\d{4}",
                           " ".join(str(x.get("content", "")) for x in messages))
        order_id = prior[-1] if prior else "A1001"

    name = "cancel_order" if "cancel" in user_text.lower() else "get_order_status"
    print(f"  >> LLM invoked {name}({{'order_id': '{order_id}'}})")
    out = await run_tool(name, {"order_id": order_id})
    print(f"  << tool returned: {out}")

    reply = _stub_render(json.loads(out))
    messages.append({"role": "assistant", "content": reply})
    print(f"[assistant] {reply}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _has_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"))


async def handle(messages: list[dict], user_text: str) -> None:
    print(f"\n[user] {user_text}")
    if _has_llm():
        try:
            await turn_openai(messages, user_text)
            return
        except Exception as e:
            print(f"  (LLM path failed: {e}; falling back to stub)")
    await turn_stub(messages, user_text)


SCRIPT = [
    "Hi, where is my order A1001?",
    "Can you cancel order A1002?",
    "What about A1003?",          # relies on conversation context
    "And can you cancel that one too?",   # should hit the not_cancellable guard
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true", help="interactive mode")
    args = ap.parse_args()

    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    mode = "real LLM" if _has_llm() else "STUB (no OPENAI_* configured)"
    print(f"=== Food delivery support agent — {mode} ===")

    if args.chat:
        print("Type a message, or 'quit' to exit. Try: A1001, A1002, A1003\n")
        loop = asyncio.get_event_loop()
        while True:
            try:
                text = await loop.run_in_executor(None, input, "you > ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().lower() in {"quit", "exit", "q"}:
                break
            if text.strip():
                await handle(messages, text.strip())
        print("\nbye.")
    else:
        for t in SCRIPT:
            await handle(messages, t)
        print(f"\n--- conversation carried {len(messages)} messages ---")


if __name__ == "__main__":
    asyncio.run(main())
