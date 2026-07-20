"""
loadtest.py
-----------
Fire N concurrent streaming requests at the service and report, per request:
  * TTFT  — time to first token (the latency users actually feel)
  * total — time until the stream completes
Then print aggregate p50/p95 and throughput.

Usage:
  python loadtest.py --n 10 --url http://localhost:8000/stream
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import aiohttp

PROMPT = "Explain what token streaming is in one short paragraph."


async def one_request(session: aiohttp.ClientSession, url: str, i: int) -> dict:
    start = time.perf_counter()
    ttft = None
    tokens = 0
    async with session.post(url, json={"prompt": PROMPT}) as resp:
        async for raw in resp.content:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - start  # first token arrived
            if "[DONE]" in line:
                break
            tokens += 1
    total = time.perf_counter() - start
    return {"i": i, "ttft": ttft or total, "total": total, "tokens": tokens}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--url", default="http://localhost:8000/stream")
    args = ap.parse_args()

    wall0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[one_request(session, args.url, i) for i in range(args.n)]
        )
    wall = time.perf_counter() - wall0

    ttfts = [r["ttft"] for r in results]
    totals = [r["total"] for r in results]
    toks = sum(r["tokens"] for r in results)

    def pct(xs, p):
        return round(statistics.quantiles(xs, n=100)[p - 1], 3) if len(xs) > 1 else round(xs[0], 3)

    print(f"\n{args.n} concurrent requests -> {args.url}")
    print(f"{'req':>4} {'ttft_s':>8} {'total_s':>9} {'tokens':>7}")
    for r in sorted(results, key=lambda x: x["i"]):
        print(f"{r['i']:>4} {r['ttft']:>8.3f} {r['total']:>9.3f} {r['tokens']:>7}")

    print("\nAggregate")
    print(f"  TTFT   p50={pct(ttfts,50)}s  p95={pct(ttfts,95)}s")
    print(f"  Total  p50={pct(totals,50)}s  p95={pct(totals,95)}s")
    print(f"  Wall clock for all {args.n}: {wall:.3f}s")
    print(f"  Aggregate throughput: {toks/wall:.1f} tokens/s across all requests")


if __name__ == "__main__":
    asyncio.run(main())
