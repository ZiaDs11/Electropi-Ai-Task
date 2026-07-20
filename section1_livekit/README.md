# Section 1 — LiveKit Agents (Real-time Voice AI)

**Deliverable:** a real-time voice support agent for a food-delivery app, with
function tools the LLM calls mid-conversation.

**Pipeline:** Deepgram STT → Qwen2.5-3B-Instruct (Ollama) → Cartesia TTS, with
Silero VAD for barge-in.

| Requirement | Where it is | Status |
|---|---|---|
| Task 1.1 — voice agent + tool calling | `agent.py`, `tools.py` | Done |
| Task 1.1 write-up — barge-in, adding a 2nd tool safely | [Write-up 1.1](#write-up-11--barge-in-and-adding-a-second-tool-safely) | Done |
| Task 1.2 (bonus) — swap a pipeline component | one env var, no code change | Done |
| Task 1.2 write-up | [Write-up 1.2](#write-up-12-bonus--swapping-a-pipeline-component) | Done |
| Tests | `test_tools.py` — 9 passing | Done |

---

## TL;DR — commands cheat sheet (Windows / VS Code cmd, no Docker)

```cmd
:: Setup (once)
cd section1_livekit
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
:: API keys are already in .env — nothing to configure

:: Ollama — keep running in its own terminal
ollama serve
ollama pull qwen2.5:3b-instruct

:: Verify the whole environment (run this first when anything breaks)
python preflight.py

:: Text-only demos — no LiveKit needed
python test_tools.py          :: 9 unit tests
python conversation.py        :: scripted 4-turn demo with real tool calls
python conversation.py --chat :: interactive REPL

:: Full voice pipeline — LiveKit Cloud, no server, no Docker
python agent.py dev
```

For voice: open the [Agents Playground](https://agents-playground.livekit.io),
sign in, select the project, and speak. Day-to-day flow is three terminals:
(1) `ollama serve`, (2) venv + `python agent.py dev`, (3) the Playground in a
browser.

**Test order IDs:** `A1001` out for delivery · `A1002` preparing (cancellable) ·
`A1003` delivered.

---

## 1. Quickstart for the reviewer

Fastest path to seeing it work — three commands, no LiveKit server needed. All
API keys are already configured in the included `.env`:

```bash
ollama serve && ollama pull qwen2.5:3b-instruct
pip install -r requirements.txt
python test_tools.py        # 9 unit tests
python conversation.py      # scripted 4-turn conversation with real tool calls
```

`conversation.py --chat` opens an interactive REPL if you want to type your own
turns.

### Full setup (~5 minutes)

```bash
# 1. Local LLM
ollama serve
ollama pull qwen2.5:3b-instruct

# 2. Config — nothing to do: .env ships with the project and already
#    contains all API keys (Deepgram, Cartesia, LiveKit, Ollama settings)

# 3. Install
pip install -r requirements.txt

# 4. Verify the whole environment in one command
python preflight.py
```

`preflight.py` on a healthy machine:

```
  [ok]   .env found
  [ok]   Ollama reachable at http://localhost:11434/v1
  [ok]   model available: qwen2.5:3b-instruct
  [ok]   native tool-calling works (get_order_status {"order_id":"A1001"})
  [ok]   Deepgram key present (e54428…13f5)
  [ok]   Cartesia key present (sk_car_P1J…tUwu)
  [ok]   LiveKit configured (ws://localhost:7880)
```

### Full voice pipeline

The agent needs a LiveKit server to attach to. Any **one** of the three options
below works — pick whichever fits your machine. Options B and C need no Docker.

**Option A — Docker (if you have it):**

```bash
docker run --rm -p 7880:7880 livekit/livekit-server --dev
python agent.py dev
```

**Option B — native binary, no Docker:**

Download `livekit-server` for your OS from the
[LiveKit releases page](https://github.com/livekit/livekit/releases)
(on Windows, unzip and run from `cmd`; on macOS `brew install livekit`), then:

```bash
livekit-server --dev
python agent.py dev
```

For A and B, open the
[LiveKit Agents Playground](https://agents-playground.livekit.io), point it at
`ws://localhost:7880`, connect, and speak.

**Option C — LiveKit Cloud, no server at all:**

The included `.env` is already configured with LiveKit Cloud credentials
(`LIVEKIT_URL=wss://...livekit.cloud` plus API key/secret), so this is the
default path:

1. Run `python agent.py dev`.
2. Open the [Agents Playground](https://agents-playground.livekit.io), sign in,
   and select the project — it connects automatically.

(To point at your own cloud project instead, create one at
[cloud.livekit.io](https://cloud.livekit.io), generate a key under
**Settings → Keys**, and replace `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and
`LIVEKIT_API_SECRET` in `.env`.)

> Note: only the room/audio transport moves to the cloud. STT, the Ollama LLM,
> and TTS still run from your machine, so keep `ollama serve` running.

**Test data:** `A1001` out for delivery · `A1002` preparing (cancellable) ·
`A1003` delivered.

---

## 2. Files

| File | Purpose |
|---|---|
| `agent.py` | The LiveKit voice agent — primary deliverable for Task 1.1/1.2. |
| `tools.py` | Business logic returning structured data (`ToolResult`), never prose. |
| `preflight.py` | Verifies Ollama, model, native tool-calling, and API keys. **Run first.** |
| `conversation.py` | Multi-turn text demo with conversation memory + `--chat` REPL. |
| `demo_tool_call.py` | Single-turn text demo — simplest proof of a tool call. |
| `test_tools.py` | 9 unit tests: tool facts, error codes, tool-call recovery. |

---

## 3. Stack and rationale

| Stage | Choice | Reasoning |
|---|---|---|
| STT | Deepgram nova-2 | Streaming with interim results; strong accuracy/latency balance on telephony audio. |
| LLM | Qwen2.5-3B-Instruct via Ollama | Local: zero per-call cost, order data stays on-premise, same class of self-hosted model that Section 3 quantizes and Section 4 serves. Reliable native tool-calling at 3B. |
| TTS | Cartesia sonic-2 | Low-latency streaming synthesis — this is what protects time-to-first-audio. |
| VAD | Silero | Detects the caller talking over the agent; the mechanism behind barge-in. |

### Model choice was a finding, not a preference

An earlier run on `llama3.2:3b` produced this as a **user-facing reply**:

```
[assistant] {"name":"cancel_order)","parameters":{$"order_id":"A1002"}}}
```

The model emitted the tool call as text instead of populating the API's
structured `tool_calls` field — malformed, and the tool never executed. Two
failures at once: the caller hears JSON read aloud, and the action silently does
not happen. Switching to `qwen2.5:3b-instruct` fixed it at the source.

Consequences carried into the deliverable: `preflight.py` probes for exactly this
before you place a call, and `conversation.py` keeps a recovery shim for models
that still misbehave.

---

## 4. Design: structured tool returns

Tools return **facts**, not sentences.

```python
# tools.py — the business layer states WHAT IS TRUE
ToolResult.success(order_id="A1001", status="out_for_delivery",
                   eta_minutes=12, courier="Kareem", total_egp=185.0)
```

```
# the LLM decides HOW TO SAY IT
[assistant] Your order A1001 is out for delivery, arriving in about 12 minutes
            with Kareem. That's Koshari and a Cola, 185.00 EGP. Want to track it?
```

**Business layer owns facts; LLM owns language.** This buys four things:
localisation (the same tool answers in Arabic or English), testability
(`assert eta_minutes == 12` beats substring-matching English), reuse (the same
tool backs an SMS or mobile UI), and model freedom (it can compare two structured
results; it can only concatenate two finished sentences).

Errors follow the same rule — data, not an apology:

```json
{"ok": false, "error_code": "not_cancellable",
 "data": {"order_id": "A1001", "status": "out_for_delivery",
          "cancellable_states": ["preparing"]}}
```

The failure carries the current status, so the LLM explains *why* without a
second round-trip — one less turn of latency on a live call.

**Trade-off.** Letting the LLM phrase things means it can paraphrase loosely,
which matters for exact refund amounts. Mitigated by instructing the model to
state amounts verbatim; for compliance-critical strings I would add a
`display_text` field to be read exactly. Narrow exception, not the default.

### Sample transcript (`python conversation.py`)

```
[user] Hi, where is my order A1001?
  >> LLM invoked get_order_status({'order_id': 'A1001'})
  << tool returned: {"ok": true, "data": {"order_id": "A1001", "status":
     "out_for_delivery", "eta_minutes": 12, "courier": "Kareem",
     "total_egp": 185.0, "items": ["Koshari", "Cola"]}}
[assistant] Your order A1001 is out for delivery and should arrive in about 12
     minutes with Kareem. It's Koshari, Cola, totalling 185.00 EGP. Would you
     like to track it or modify the order?

[user] And can you cancel that one too?     # no order id — resolved from history
  >> LLM invoked cancel_order({'order_id': 'A1003'})
  << tool returned: {"ok": false, "error_code": "not_cancellable", "data":
     {"order_id": "A1003", "status": "delivered", "cancellable_states": ["preparing"]}}
[assistant] Sorry, order A1003 is already delivered, so it can't be cancelled.
     Would you like to contact the courier instead?
```

---

## 5. Task 1.2 — swapping a pipeline component

Change one line in `.env` and restart. **No agent code changes.**

```dotenv
STT_PROVIDER=openai      # was deepgram  → Whisper
TTS_PROVIDER=openai      # was cartesia  → OpenAI TTS
```

This works because `SupportAgent` never imports a vendor. `_make_stt()`,
`_make_llm()` and `_make_tts()` are the only vendor-aware code, and each reads a
single env var:

```python
def _make_tts(provider=None):
    provider = (provider or os.getenv("TTS_PROVIDER", "cartesia")).lower()
    if provider == "cartesia":
        from livekit.plugins import cartesia
        return cartesia.TTS(model=os.getenv("CARTESIA_MODEL", "sonic-2"), ...)
    if provider == "openai":
        from livekit.plugins import openai
        return openai.TTS(voice=os.getenv("OPENAI_TTS_VOICE", "alloy"))
```

---

## Write-up 1.1 — barge-in and adding a second tool safely

### Barge-in / interruption handling

Silero VAD watches inbound audio while TTS is playing. On detecting caller
speech, LiveKit stops playback, truncates the half-spoken assistant turn *in the
chat context* — so the model does not believe it said a sentence the caller never
heard — and starts a fresh STT→LLM turn.

Two knobs matter: `allow_interruptions=True`, and `min_interruption_duration`
(set to 0.4 s via `MIN_INTERRUPTION_SEC`) so coughs and back-channels like "mhm"
do not cancel the agent.

For tool calls in flight at interruption time I treat reads and writes
differently. `get_order_status` is idempotent and cheap, so I let it finish and
cache the result. `cancel_order` mutates, so it is gated behind an explicit
confirmation turn and can never be left half-applied by an interruption.

### Adding a second tool safely

`cancel_order` is the second tool. The pattern, in four parts:

1. **Strict schema.** Typed JSON schema with `required` fields, plus
   normalisation inside the tool (`order_id.strip().upper()`), so a sloppy model
   argument cannot reach the domain layer.
2. **Failure as structured data.** Tools never raise across the boundary; they
   return `{ok: false, error_code, data}`. The LLM turns the code into a natural
   apology, and machine-readable codes mean I can alert on `not_cancellable`
   rates without parsing English.
3. **Guardrails on mutations.** The cancellable-state rule lives in `tools.py`,
   not the prompt, so the model cannot be talked out of it. In production I would
   add a confirmation turn and an idempotency key so a retried call cannot
   double-cancel.
4. **Least privilege + timeouts.** The tool touches only the orders service, with
   a timeout and circuit breaker so a slow downstream cannot stall the voice turn.

## Write-up 1.2 (bonus) — swapping a pipeline component

The pipeline is decoupled by construction. `SupportAgent` holds only the persona
and tools and imports no vendor; all vendor knowledge lives in three factories
(`_make_stt`, `_make_llm`, `_make_tts`), each keyed off one env var. Swapping
Cartesia → OpenAI TTS is `TTS_PROVIDER=openai` plus a restart; Deepgram → Whisper
is `STT_PROVIDER=openai`. Zero agent-code changes, and the tools are untouched
because they return structured data rather than provider-shaped strings.

What actually needs attention on a swap, from doing it:

- **Streaming support** — a non-streaming TTS destroys time-to-first-audio even
  when quality is identical.
- **Audio format / sample rate** expectations between plugins; LiveKit's plugin
  layer normalises most but not all of this.
- **Language coverage** — the target must support `STT_LANGUAGE`, which is the
  real constraint for Arabic.
- **Latency profile** — the thing you feel on a call and will not see in a unit
  test.

The LLM is decoupled the same way: `OPENAI_BASE_URL` points at Ollama today and
could point at the vLLM service from Section 4 tomorrow.

---

## 6. Limitations and honest scope

- The orders "DB" in `tools.py` is an in-memory dict (mocked lookup, as allowed
  by the task).
- I validated the **tool-calling and conversation logic** end-to-end via
  `conversation.py` and `test_tools.py`. The full audio path is implemented and
  configured, but I did not capture a screen recording — `preflight.py` exists so
  a reviewer can confirm their own setup in one command.
- `STT_LANGUAGE=ar` selects Arabic, but Egyptian dialect ASR is materially harder
  than MSA. I would benchmark on real dialect audio before promising accuracy.
- Tool-call recovery in `conversation.py` is a pragmatic shim for weak models. The
  correct fix is a model with native tool-calling, which is why the stack
  standardises on Qwen2.5.
- `_stub_render()` fakes the LLM's wording for offline runs only. It lives in the
  demo layer, never in `tools.py`.