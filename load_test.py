import argparse
import asyncio
import json
import statistics
import time

import aiohttp

from openai_server import auth_headers

PROMPTS = [
    "Explain how a hash map works.",
    "Write a haiku about GPUs.",
    "Summarize the causes of World War I in three sentences.",
    "What is the difference between TCP and UDP?",
]


async def one_request(session, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    tokens = 0
    async with session.post("/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            return {"ok": False, "status": resp.status}
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            if ttft is None and chunk.get("choices") and chunk["choices"][0]["delta"]:
                ttft = time.perf_counter() - start
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
    return {
        "ok": True,
        "latency": time.perf_counter() - start,
        "ttft": ttft or 0.0,
        "tokens": tokens,
    }


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100)[int(p) - 1] if len(values) > 1 else values[0]


async def main(url: str, model: str, total: int, concurrency: int, max_tokens: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> dict:
        async with semaphore:
            return await one_request(session, model, PROMPTS[i % len(PROMPTS)], max_tokens)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(
        base_url=url, headers=auth_headers(), timeout=timeout
    ) as session:
        start = time.perf_counter()
        results = await asyncio.gather(*(bounded(i) for i in range(total)))
        elapsed = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    failed = [r["status"] for r in results if not r["ok"]]
    print(f"Requests: {len(ok)}/{total} ok, concurrency {concurrency}, {elapsed:.1f}s total")
    if failed:
        print(f"Failed status codes: {sorted(set(failed))}")
    if not ok:
        return
    latencies = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok]
    tokens = sum(r["tokens"] for r in ok)
    print(f"Latency  p50 {percentile(latencies, 50):.2f}s  p95 {percentile(latencies, 95):.2f}s")
    print(f"TTFT     p50 {percentile(ttfts, 50):.2f}s  p95 {percentile(ttfts, 95):.2f}s")
    print(f"Throughput {tokens / elapsed:.0f} output tokens/s, {len(ok) / elapsed:.2f} requests/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test an OpenAI-compatible endpoint.")
    parser.add_argument("url", help="Server base URL, without /v1")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.model, args.requests, args.concurrency, args.max_tokens))
