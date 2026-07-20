"""
agent.py
--------
LiveKit voice agent for a food-delivery support line.

CONCRETE STACK (this is what the project is built and documented against):

    STT   Deepgram  nova-2
    LLM   Qwen2.5-3B-Instruct, served locally by Ollama on :11434
          (OpenAI-compatible API, so the standard openai plugin is used)
    TTS   Cartesia  sonic
    VAD   Silero    (drives barge-in / interruption handling)

Running a local LLM through Ollama is a deliberate choice, not a fallback: it
keeps per-call cost at zero, keeps order data on-premise (relevant for
enterprise customers), and ties this section to the same class of
self-hosted model that Section 3 quantizes and Section 4 containerises.
Qwen2.5-3B-Instruct specifically because it has reliable native tool-calling at
this size — smaller Llama builds tend to emit tool calls as plain text.

Run:
    ollama serve && ollama pull qwen2.5:3b-instruct
    python preflight.py       # verifies Ollama, the model, and the API keys
    python agent.py dev       # then connect via the LiveKit Agents Playground

All configuration comes from the repo-root .env — see .env.example.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)

# Plugins MUST be imported at module level: LiveKit registers each plugin on
# import and requires that to happen on the main thread. Importing them lazily
# inside the factories crashes the job with
# "RuntimeError: Plugins must be registered on the main thread".
from livekit.plugins import cartesia, deepgram, openai, silero

from env_loader import load_env
import tools as biz  # vendor-free business logic (returns structured data)

load_env()

logger = logging.getLogger("food-agent")

SYSTEM_PERSONA = (
    "You are a friendly, concise support assistant for a food-delivery app "
    "operating in Cairo. You help customers check and manage their orders.\n"
    "When a customer asks about an order, call the appropriate tool instead of "
    "guessing.\n"
    "Tools return STRUCTURED JSON facts, not sentences. Your job is to turn "
    "those facts into a short, natural spoken reply in the customer's own "
    "language. Never read raw JSON, field names, or error codes aloud.\n"
    "Mention status, ETA, courier, total (in EGP) and items when present, then "
    "offer a short relevant next step (track, modify, cancel, rate, refund) "
    "that fits the order's state. Keep it to 2-3 sentences.\n"
    "If a result has ok=false, explain the situation kindly and suggest what to "
    "do next. Never invent an order detail; state amounts and times exactly as "
    "returned.\n"
    "Always use the provided tool-calling interface; never write a tool call as "
    "text in your reply."
)


class SupportAgent(Agent):
    """Persona + the tools the LLM is allowed to call."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PERSONA)

    # Both tools return a dict of structured facts. LiveKit serialises it to
    # JSON for the LLM, which owns the wording. The business layer never
    # produces user-facing prose.

    @function_tool
    async def get_order_status(
        self, context: RunContext, order_id: str
    ) -> dict[str, Any]:
        """Get the current status, ETA, courier, total and items of an order.

        Args:
            order_id: The order reference like "A1001".
        """
        logger.info("TOOL CALL get_order_status(order_id=%s)", order_id)
        result = await biz.get_order_status(order_id)
        logger.info("TOOL RESULT %s", result.to_dict())
        return result.to_dict()

    @function_tool
    async def cancel_order(
        self, context: RunContext, order_id: str
    ) -> dict[str, Any]:
        """Cancel an order if its current state still permits cancellation.

        Args:
            order_id: The order reference like "A1002".
        """
        logger.info("TOOL CALL cancel_order(order_id=%s)", order_id)
        result = await biz.cancel_order(order_id)
        logger.info("TOOL RESULT %s", result.to_dict())
        return result.to_dict()


# --------------------------------------------------------------------------- #
# Pipeline construction
#
# Each stage is built by its own factory reading a single env var. That is what
# makes the Task 1.2 swap a one-line .env change with zero agent-code change —
# `SupportAgent` above never imports a vendor.
# --------------------------------------------------------------------------- #
def _make_llm():
    """Qwen2.5-3B-Instruct via Ollama's OpenAI-compatible endpoint."""
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
    logger.info("LLM: %s @ %s", model, base_url)
    return openai.LLM(
        model=model,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # Ollama ignores the value
        temperature=0.3,
    )


def _make_stt(provider: str | None = None):
    """STT_PROVIDER=deepgram (primary) | openai (swap target for Task 1.2)."""
    provider = (provider or os.getenv("STT_PROVIDER", "deepgram")).lower()
    language = os.getenv("STT_LANGUAGE", "en")
    logger.info("STT: %s (language=%s)", provider, language)

    if provider == "deepgram":
        # nova-2 is the accuracy/latency sweet spot for streaming telephony.
        return deepgram.STT(
            model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
            language=language,
            interim_results=True,   # partials keep perceived latency low
            punctuate=True,
            smart_format=True,
        )
    if provider == "openai":
        return openai.STT(language=language)  # Whisper
    raise ValueError(f"Unknown STT_PROVIDER: {provider!r} (use deepgram|openai)")


def _make_tts(provider: str | None = None):
    """TTS_PROVIDER=cartesia (primary) | openai (swap target for Task 1.2)."""
    provider = (provider or os.getenv("TTS_PROVIDER", "cartesia")).lower()
    logger.info("TTS: %s", provider)

    if provider == "cartesia":
        # sonic-2 is Cartesia's low-latency streaming model.
        # Only pass `voice` when explicitly configured — passing voice=None
        # overrides the plugin's default voice and can yield empty synthesis
        # ("no audio frames were pushed").
        kwargs: dict = {"model": os.getenv("CARTESIA_MODEL", "sonic-2")}
        voice_id = os.getenv("CARTESIA_VOICE_ID")
        if voice_id:
            kwargs["voice"] = voice_id
        return cartesia.TTS(**kwargs)
    if provider == "openai":
        return openai.TTS(voice=os.getenv("OPENAI_TTS_VOICE", "alloy"))
    raise ValueError(f"Unknown TTS_PROVIDER: {provider!r} (use cartesia|openai)")


def _build_session() -> AgentSession:
    return AgentSession(
        stt=_make_stt(),
        llm=_make_llm(),
        tts=_make_tts(),
        # Silero VAD detects the caller speaking over the agent, which is what
        # enables barge-in: TTS playback stops and a new turn starts.
        vad=silero.VAD.load(),
        allow_interruptions=True,
        # Ignore very short bursts ("mhm", a cough) so back-channels don't
        # cancel the agent mid-sentence.
        min_interruption_duration=float(os.getenv("MIN_INTERRUPTION_SEC", "0.4")),
    )


async def entrypoint(ctx: JobContext) -> None:
    logging.basicConfig(level=logging.INFO)
    await ctx.connect()

    session = _build_session()
    await session.start(agent=SupportAgent(), room=ctx.room)
    await session.generate_reply(
        instructions=(
            "Greet the caller warmly in one sentence and ask for their order "
            "number."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))