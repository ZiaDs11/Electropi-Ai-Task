# Write-ups & Assumptions (index)

All half-page write-ups live in each section's `README.md`. Quick links + the
one-line "so what" of each:

- **Section 1 (LiveKit)** — `section1_livekit/README.md`
  - *Barge-in & adding a tool safely:* VAD-driven interruption with min-duration
    tuning + context truncation; gate mutating tools behind confirmation;
    strict JSON schemas; errors returned as data, not exceptions.
  - *Swapping a provider (bonus):* env-var factories decouple STT/LLM/TTS; a swap
    is a one-line env change, zero agent-code change.
- **Section 2 (RAG)** — `section2_rag/README.md`
  - *Improving long-doc quality:* hybrid BM25+dense → cross-encoder re-ranking →
    structure-aware / parent-document chunking → query transforms → metadata
    filters, tuned against a small eval set.
- **Section 3 (Quantization)** — `section3_quantization/README.md`
  - *When to pick which:* bitsandbytes for dev/QLoRA; GPTQ/AWQ for max GPU-serving
    speed on a pinned model; GGUF for CPU/Mac/edge. Key nuance: 4-bit
    bitsandbytes cuts *memory*, not necessarily latency.
- **Section 4 (Deployment)** — `section4_deployment/README.md`
  - *Scaling to 50 concurrent:* swap engine to vLLM (continuous batching) →
    queue + admission control → autoscale on queue depth → prefix/response
    caching → observability on TTFT/tokens-per-sec.

## Global assumptions & shortcuts
- Mocked data where the test allows it (orders "DB" in Section 1; own short docs
  in Section 2).
- Offline/mock paths exist in every section so flows are demonstrable without
  credentials or a GPU; they are always clearly labelled and are not presented as
  real model output.
- Section 3 numbers are placeholders to be replaced with real measurements from
  your hardware. See the root `README.md` "honesty" section for exactly what was
  and wasn't executed.
