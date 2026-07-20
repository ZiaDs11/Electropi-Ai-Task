# Write-ups, Assumptions & Limitations

All half-page write-ups live in each section's `README.md`. This file indexes
them and states plainly what was run, on what hardware, and what wasn't.

**Development machine:** Windows laptop, **CPU-only — no NVIDIA GPU**, Docker
installed but not running. Those two constraints shaped Section 3 and Section 4,
and are called out in each section below rather than buried.

---

## Section 1 — LiveKit Agents · `section1_livekit/README.md`

- **Barge-in & adding a tool safely:** Silero VAD drives interruption; playback
  stops and the half-spoken turn is truncated *in the chat context* so the model
  doesn't believe it said something the caller never heard.
  `min_interruption_duration` (0.4 s) stops coughs and back-channels from
  cancelling the agent. Mutating tools are gated: the cancellable-state rule
  lives in `tools.py`, not the prompt, so the model can't be talked out of it.
  Errors are returned as structured data (`{ok:false, error_code, data}`), never
  raised across the boundary and never as English prose.
- **Swapping a provider (bonus):** env-var factories decouple STT/LLM/TTS, so a
  swap is a one-line `.env` change with zero agent-code change. The constraints
  that actually matter on a swap: streaming support, sample rate, language
  coverage, and latency profile.
- **Model-choice finding:** `llama3.2:3b` emitted tool calls as malformed *text*
  instead of using the structured `tool_calls` field — the customer would hear
  JSON read aloud and the tool would never execute. Switched to
  `qwen2.5:3b-instruct`; `preflight.py` probes for this before any call.

**Status:** tool-calling, conversation memory, error paths and 9 unit tests all
executed and passing. The live audio path is implemented and configured but was
not exercised end-to-end (no capture of a live call).

---

## Section 2 — LangChain / RAG · `section2_rag/README.md`

- **Improving long-doc quality:** hybrid BM25 + dense with rank fusion →
  cross-encoder re-ranking on a shortlist → structure-aware / parent-document
  chunking → query transforms (multi-query, HyDE) → metadata filters, all tuned
  against a small labelled eval set rather than by feel. Hybrid + re-ranking is
  usually most of the gain.

**Status:** retrieval mechanics executed end-to-end (chunking, scored search,
guardrail firing, citation extraction). Runs on CPU without issue.

---

## Section 3 — Quantization · `section3_quantization/README.md`

**Hardware constraint, stated up front: this machine has no NVIDIA GPU.**
That is not a footnote — it determined the approach.

- **Why bitsandbytes NF4 was not the right path here.** bitsandbytes' 4-bit
  kernels are CUDA-only. On a CPU-only machine `load_in_4bit=True` cannot run at
  all, so reporting NF4 numbers would have meant fabricating them. The harness in
  `benchmark.py` implements this path correctly and fails with an explicit
  message rather than silently degrading.
- **The correct CPU path is GGUF via llama.cpp**, which the brief explicitly
  permits ("or a GGUF build via llama.cpp"). GGUF Q4_K_M is genuinely 4-bit,
  computes *in* low precision rather than dequantizing to fp16, and is the format
  I would actually ship for CPU/edge deployment — so the constraint pushed toward
  the more realistic answer, not a weaker one.
- **When to pick which:** bitsandbytes for development and QLoRA fine-tuning
  (zero calibration, but dequantizes to compute); GPTQ/AWQ for maximum GPU
  serving throughput on a pinned model (one-time calibration + fast kernels);
  GGUF for CPU, Apple Silicon, and edge. Key nuance worth stating explicitly:
  **4-bit bitsandbytes cuts memory, not necessarily latency** — the speed win
  arrives only when quantization lets you fit a larger model, batch more, or use
  kernels that compute in low precision.

**Status:** the benchmark harness is complete and correct, but **no GPU
measurements were taken.** The comparison table contains clearly-labelled
*typical ranges*, explicitly marked as illustrative and not as results I
obtained. Anyone with CUDA can produce real numbers with the three documented
commands.

---

## Section 4 — Model Deployment · `section4_deployment/README.md`

- **Scaling to 50 concurrent users:** swap the engine to vLLM (continuous
  batching + PagedAttention — the single largest win) → request queue with
  admission control and backpressure → autoscale on **queue depth / GPU
  utilisation**, not CPU → prefix caching for shared system prompts plus an
  app-level response cache → observability on TTFT, tokens/sec and queue depth so
  scaling decisions are measured rather than guessed.

**Status:** the service, SSE token streaming, and the concurrent load test were
executed and produced **real measurements** (TTFT p50 = 39 ms, 10 concurrent
requests, 314.9 tok/s aggregate) — these are the only benchmark numbers in the
submission that I actually measured.

**Docker was not run.** It is installed on the machine but was not started, so
`docker build` / `docker run` were never executed. The Dockerfile is standard
(slim Python base, dependency layer, healthcheck, uvicorn entrypoint) and the
application it wraps is verified under uvicorn, but I have not personally
confirmed the image builds. Stated here rather than implied as working.

---

## Global assumptions & shortcuts

- **Mocked data where the brief allows it:** the orders "database" in Section 1
  is an in-memory dict; Section 2 uses four short domain documents I wrote.
- **Concrete stack, not hypothetical:** Deepgram nova-2 + Qwen2.5-3B via Ollama +
  Cartesia sonic-2. `section1_livekit/preflight.py` verifies the whole
  environment — Ollama reachable, model pulled, native tool-calling working, keys
  present — in one command.
- **No API keys are committed.** `.env` is git-ignored; `.env.example` contains
  placeholders only.
- **Offline paths are labelled.** Every section has a path that runs without
  credentials (stub router, extractive fallback, mock serving backend). These are
  always marked as such and never presented as real model output.
- **The only measured numbers in this submission are Section 4's load test.**
  Everything else that looks like a benchmark is either labelled illustrative or
  absent.