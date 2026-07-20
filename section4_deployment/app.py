"""
app.py
------
FastAPI service that serves an LLM with token streaming.

Why FastAPI (not vLLM/TGI) for this task: it's the smallest thing that fully
satisfies the requirements (REST + real token streaming + trivial to
containerise + easy to load-test) and keeps the code readable for review. For
production at scale I'd put vLLM behind it — see the write-up in README.

Two backends, selected by MODEL_BACKEND:
  * "mock" (default): streams canned tokens with a small delay. Lets you run the
    container, streaming, and the load test end-to-end with NO GPU/model — so a
    reviewer can `docker run` and see it work in seconds.
  * "hf": loads a real Hugging Face model (e.g. Qwen2.5-1.5B-Instruct) and
    streams real tokens via TextIteratorStreamer.

Endpoints:
  GET  /health                      -> {"status":"ok"}
  POST /generate  {prompt}          -> full JSON response (non-streaming)
  POST /stream    {prompt}          -> text/event-stream, token by token
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "mock")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

app = FastAPI(title="LLM Serving API", version="1.0")


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None


# --------------------------------------------------------------------------- #
# Backend abstraction: both backends expose `stream_tokens(prompt, n)` yielding
# strings. This keeps the HTTP layer identical regardless of backend.
# --------------------------------------------------------------------------- #
class MockBackend:
    """Streams a canned response so the whole stack runs without a model."""

    async def stream_tokens(self, prompt: str, n: int) -> AsyncGenerator[str, None]:
        reply = (
            "This is a mock streaming response. Set MODEL_BACKEND=hf to serve a "
            "real model. Your prompt was received and would be processed by the "
            "language model token by token exactly like this."
        ).split()
        for word in reply[:n]:
            await asyncio.sleep(0.03)  # simulate per-token compute
            yield word + " "


class HFBackend:
    """Real Hugging Face model with true token streaming."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    async def stream_tokens(self, prompt: str, n: int) -> AsyncGenerator[str, None]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(inputs=inputs, max_new_tokens=n, do_sample=False,
                      streamer=streamer, pad_token_id=self.tok.eos_token_id)
        # generate() blocks, so run it in a thread and drain the streamer async.
        Thread(target=self.model.generate, kwargs=kwargs, daemon=True).start()

        loop = asyncio.get_event_loop()
        it = iter(streamer)
        while True:
            chunk = await loop.run_in_executor(None, lambda: next(it, None))
            if chunk is None:
                break
            yield chunk


BACKEND = HFBackend() if MODEL_BACKEND == "hf" else MockBackend()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": MODEL_BACKEND, "model": MODEL_ID}


@app.post("/generate")
async def generate(req: GenRequest) -> dict:
    n = req.max_new_tokens or MAX_NEW_TOKENS
    t0 = time.perf_counter()
    text = "".join([tok async for tok in BACKEND.stream_tokens(req.prompt, n)])
    return {"response": text.strip(), "latency_s": round(time.perf_counter() - t0, 3)}


@app.post("/stream")
async def stream(req: GenRequest) -> StreamingResponse:
    n = req.max_new_tokens or MAX_NEW_TOKENS

    async def event_gen() -> AsyncGenerator[bytes, None]:
        async for token in BACKEND.stream_tokens(req.prompt, n):
            # Server-Sent Events framing: one `data:` line per token.
            yield f"data: {json.dumps({'token': token})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
