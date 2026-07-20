# Section 4 — Model Deployment

Serve an LLM behind a REST API with **token streaming**, containerised, plus a
concurrent **load/latency test** reporting time-to-first-token (TTFT).

## Why FastAPI here
FastAPI is the smallest thing that fully meets the requirements (REST + real SSE
streaming + trivial Docker + easy load test) and stays readable for review. It's
the right call for a single-model service or a gateway. For high-QPS multi-tenant
serving I'd run **vLLM** as the engine (continuous batching, PagedAttention) —
covered in the write-up. FastAPI can then act as the thin auth/routing layer in
front of vLLM.

## Two backends
- `MODEL_BACKEND=mock` (default): streams canned tokens with a per-token delay.
  **The whole stack — Docker, streaming, load test — runs with no GPU/model**,
  so you can verify it in seconds, then switch to a real model.
- `MODEL_BACKEND=hf`: loads a real HF model (`MODEL_ID`, default
  `Qwen/Qwen2.5-1.5B-Instruct`) and streams real tokens via `TextIteratorStreamer`.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# in another shell:
curl -s -N -X POST localhost:8000/stream -H 'Content-Type: application/json' \
     -d '{"prompt":"hello"}'
python loadtest.py --n 10
```

## Run in Docker (end-to-end, no model needed)
```bash
docker build -t llm-api .
docker run -p 8000:8000 llm-api
# then: curl / loadtest as above
```
For a real model:
```bash
docker run -p 8000:8000 -e MODEL_BACKEND=hf -e MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct llm-api
# (uncomment the ML deps in requirements.txt / use a GPU base image first)
```

## Load test — actual output (mock backend, this machine)
```
10 concurrent requests -> http://localhost:8000/stream
 req   ttft_s   total_s  tokens
   0    0.041     0.984      31
   ...
   9    0.039     0.981      31

Aggregate
  TTFT   p50=0.039s  p95=0.042s
  Total  p50=0.981s  p95=0.984s
  Wall clock for all 10: 0.984s
  Aggregate throughput: 314.9 tokens/s across all requests
```
All 10 ran concurrently (wall clock ≈ a single request's time), and TTFT was
~39 ms — the number that governs perceived responsiveness. These are mock-backend
numbers; with a real model TTFT/total scale with model size, prompt length, and
GPU, but the harness and metrics are identical.

## Write-up — scaling to 50 concurrent users in production

The single-process FastAPI loop above is fine for a demo but would serialise
GPU work under real load. For ~50 concurrent users I'd change:

1. **Swap the engine for vLLM (biggest win).** Continuous/in-flight batching +
   PagedAttention pack many requests into each GPU step, so throughput scales far
   better than looping `generate()`. Keep the OpenAI-compatible API so clients
   don't change. This alone usually takes you from a handful to dozens of
   concurrent streams on one GPU.

2. **A request queue with admission control.** Bound in-flight requests, queue
   the rest, and return backpressure (429 / retry-after) past capacity so tail
   latency stays sane instead of everything degrading together. Set sensible
   max-tokens and timeouts per request.

3. **Autoscaling on the right signal.** Run N replicas behind a load balancer
   and scale on GPU utilisation / queue depth (not CPU), e.g. KEDA on Kubernetes.
   Streaming responses need sticky-enough connections and generous LB timeouts.

4. **Caching.** (a) KV-cache reuse / prefix caching for shared system prompts
   (vLLM does this), and (b) an app-level exact/semantic cache for repeated
   prompts so identical questions skip the model entirely.

5. **Batching knobs + observability.** Tune max batch size / max wait, and export
   TTFT, tokens/sec, queue depth, and GPU mem to Prometheus/Grafana so scaling
   decisions are data-driven. Load test with a ramp (locust) to find the real
   concurrency ceiling before it hits users.

Rough target shape: **FastAPI gateway (auth, rate-limit, routing) → queue →
vLLM replicas (continuous batching) → autoscaler on queue depth**, with prefix
caching on and per-request token/latency budgets.

## Notes / limitations
- Streaming uses SSE (`text/event-stream`), one JSON token per `data:` line,
  terminated by `data: [DONE]`.
- The mock backend proves the transport; real throughput depends on model/GPU.
- Default Docker image is CPU/mock to stay small and buildable anywhere; the real
  model path needs the ML deps (and ideally a CUDA base image) uncommented.
