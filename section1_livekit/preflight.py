"""
preflight.py
------------
Verifies the environment before you run the voice agent, so failures surface
here with a clear message instead of mid-call.

Checks, in order:
  1. .env is present and loaded
  2. Ollama is reachable at OPENAI_BASE_URL
  3. LLM_MODEL is actually pulled
  4. The model really does native tool-calling (the failure that bit us with
     llama3.2:3b — it emitted tool calls as plain text instead)
  5. Deepgram / Cartesia keys are present and look well-formed
  6. LiveKit server settings are present

Run:  python preflight.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

from env_loader import load_env

load_env()

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
_failures: list[str] = []
_warnings: list[str] = []


def _fail(msg: str, fix: str) -> None:
    print(f"{FAIL} {msg}\n         fix: {fix}")
    _failures.append(msg)


def _warn(msg: str, fix: str) -> None:
    print(f"{WARN} {msg}\n         fix: {fix}")
    _warnings.append(msg)


def _ok(msg: str) -> None:
    print(f"{OK} {msg}")


# --------------------------------------------------------------------------- #
def check_env_file() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if (root / ".env").exists():
        _ok(f".env found at {root / '.env'}")
    else:
        _fail(".env not found",
              f"cp {root / '.env.example'} {root / '.env'} and fill it in")


def check_ollama() -> str | None:
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    tags_url = base.replace("/v1", "") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        _fail(f"Ollama not reachable at {base} ({e})",
              "start it with `ollama serve` (default port 11434)")
        return None

    _ok(f"Ollama reachable at {base}")

    want = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
    if any(m == want or m.startswith(want.split(":")[0]) for m in models):
        _ok(f"model available: {want}")
        return want

    _fail(f"model {want!r} not pulled (have: {', '.join(models) or 'none'})",
          f"ollama pull {want}")
    return None


async def check_tool_calling(model: str) -> None:
    """The check that actually matters: does this model use the structured
    tool-calling API, or does it emit tool calls as text?"""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        _warn("openai package not installed; skipping tool-call check",
              "pip install -r requirements.txt")
        return

    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    )
    schema = [{
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }]
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Where is my order A1001?"}],
            tools=schema,
            temperature=0,
        )
    except Exception as e:
        _fail(f"tool-calling probe failed: {e}",
              "check the model name and that Ollama is healthy")
        return

    msg = r.choices[0].message
    if msg.tool_calls:
        call = msg.tool_calls[0]
        _ok(f"native tool-calling works ({call.function.name} "
            f"{call.function.arguments})")
    else:
        _warn(
            "model did NOT use the structured tool_calls field; it replied "
            f"with text: {(msg.content or '')[:80]!r}",
            "conversation.py can recover text-emitted tool calls, but for the "
            "voice agent prefer a model with native support "
            "(qwen2.5:3b-instruct or larger)",
        )


def check_provider_keys() -> None:
    dg = os.getenv("DEEPGRAM_API_KEY", "")
    ct = os.getenv("CARTESIA_API_KEY", "")

    stt = os.getenv("STT_PROVIDER", "deepgram").lower()
    tts = os.getenv("TTS_PROVIDER", "cartesia").lower()

    if stt == "deepgram":
        if not dg:
            _fail("DEEPGRAM_API_KEY not set (STT_PROVIDER=deepgram)",
                  "add it to .env")
        elif len(dg) < 32:
            _warn("DEEPGRAM_API_KEY looks too short", "double-check the value")
        else:
            _ok(f"Deepgram key present ({dg[:6]}…{dg[-4:]})")

    if tts == "cartesia":
        if not ct:
            _fail("CARTESIA_API_KEY not set (TTS_PROVIDER=cartesia)",
                  "add it to .env")
        elif not ct.startswith("sk_car_"):
            _warn("CARTESIA_API_KEY does not start with 'sk_car_'",
                  "double-check the value")
        else:
            _ok(f"Cartesia key present ({ct[:10]}…{ct[-4:]})")


def check_livekit() -> None:
    missing = [v for v in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
               if not os.getenv(v)]
    if missing:
        _warn(f"LiveKit settings missing: {', '.join(missing)}",
              "needed only for `python agent.py dev`; run "
              "`docker run --rm -p 7880:7880 livekit/livekit-server --dev` "
              "and use the devkey/secret defaults")
    else:
        _ok(f"LiveKit configured ({os.getenv('LIVEKIT_URL')})")


async def main() -> None:
    print("\n=== preflight: Section 1 voice agent ===\n")
    check_env_file()
    model = check_ollama()
    if model:
        await check_tool_calling(model)
    check_provider_keys()
    check_livekit()

    print()
    if _failures:
        print(f"{len(_failures)} blocking issue(s). Fix the [FAIL] lines above.")
        sys.exit(1)
    if _warnings:
        print(f"Ready, with {len(_warnings)} warning(s). "
              "You can run: python agent.py dev")
    else:
        print("All checks passed. Run: python agent.py dev")


if __name__ == "__main__":
    asyncio.run(main())
