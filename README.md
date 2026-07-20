# Electro Pi — AI Engineer Technical Test

Four sections, one folder each, every folder self-contained (its own
`requirements.txt`, `README.md`, and write-up). You should be able to run any
section within ~10 minutes of cloning.

| Section | Folder | What it does | Runs without keys/GPU? |
|---|---|---|---|
| 1. LiveKit Agents | `section1_livekit/` | Voice support agent (STT→LLM→TTS) with a function tool the LLM calls | ✅ tool-call demo runs offline; full voice needs keys |
| 2. LangChain RAG | `section2_rag/` | RAG over 4 docs with citations + no-context guardrail | ✅ (downloads a small embedding model once) |
| 3. Quantization | `section3_quantization/` | bf16 vs 4-bit NF4: memory / throughput / quality | ⚠️ needs a GPU + model download to produce numbers |
| 4. Deployment | `section4_deployment/` | FastAPI + streaming + Docker + load test | ✅ mock backend runs fully; real model optional |

## Quick start per section
```bash
cd section1_livekit    && pip install -r requirements.txt && python demo_tool_call.py
cd section2_rag        && pip install -r requirements.txt && python rag_pipeline.py --build --demo
cd section3_quantization && pip install -r requirements.txt && python benchmark.py --precision fp16 --out results_fp16.json
cd section4_deployment && pip install -r requirements.txt && uvicorn app:app --port 8000   # then: python loadtest.py --n 10
```

## Cross-cutting design choices
- **Vendor decoupling.** Sections 1 and 4 both read providers/endpoints from env
  vars (`STT_PROVIDER`, `TTS_PROVIDER`, `OPENAI_BASE_URL`, `MODEL_BACKEND`) so you
  can point the LLM at real OpenAI *or* a local server — and the local server can
  be the very model you containerise in Section 4.
- **Runnable-without-credentials paths.** Each section has an offline/mock path
  (stub router, extractive fallback, mock backend) so a reviewer can see the flow
  work immediately, then swap in real keys/models. Mock paths are always labelled.
- **Guardrails.** RAG refuses when retrieval is weak; tools return errors as data
  so a failed tool call never crashes a turn.

## What I actually executed vs. what needs your hardware (honesty)
Built and verified in a sandbox **without a GPU, Docker, or model-download
access**, so:
- ✅ **Ran:** Section 1 tool-call demo (transcript in its README); Section 2 RAG
  *mechanics* (chunking / FAISS retrieval-with-score / guardrail / citations,
  via a stubbed embedding); Section 4 API + streaming + 10-way load test (real
  numbers in its README).
- ⚠️ **You must run on your machine for real numbers:** Section 2 with the real
  MiniLM embedding + an LLM; Section 3 entirely (needs a GPU); Section 4 with
  `MODEL_BACKEND=hf` and `docker build/run` (Docker wasn't available in my
  sandbox — the app is verified under uvicorn and the Dockerfile is standard).

Placeholders in Section 3's table are marked as *typical ranges, replace with
your measurements* — they are not presented as real results.

## Write-ups
Each section's half-page write-up is in that section's `README.md`. `NOTES.md`
at the repo root collects all four in one place.
