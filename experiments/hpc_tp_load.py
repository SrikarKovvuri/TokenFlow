"""Closed-loop streaming workload and engine measurements for the managed Slurm sweep."""
import asyncio
import gzip
import json
import random
import time
from pathlib import Path

import httpx

from vllm_metrics import parse_gauges


def prompt(index):
    rng = random.Random(431 + index)
    records = [f"Record {i}: queue={rng.randrange(1000)}, latency={rng.randrange(500)} ms." for i in range(16)]
    return "\n".join(records + ["Explain the patterns in these records in detail."])


async def generate(client, url, model, index):
    payload = dict(model=model, messages=[dict(role="user", content=prompt(index % 128))],
                   stream=True, stream_options={"include_usage": True}, seed=431,
                   temperature=0, max_tokens=256, ignore_eos=True)
    start = time.monotonic()
    row = dict(index=index, started_epoch=time.time(), content_seconds=[], done=False)
    try:
        async with client.stream("POST", url + "/v1/chat/completions", json=payload) as reply:
            row["http"] = reply.status_code
            reply.raise_for_status()
            async for line in reply.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    row["done"] = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        row["finish"] = choice["finish_reason"]
                    if choice["delta"].get("content"):
                        row["content_seconds"].append(time.monotonic() - start)
        assert row["done"] and row["content_seconds"] and row["finish"] == "length"
        assert row["usage"]["completion_tokens"] == 256
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.monotonic() - start
    return row


async def monitor(client, engine, output, stop):
    with gzip.open(output, "xt", compresslevel=1) as handle:
        while not stop.is_set():
            sample = {"start_epoch": time.time()}
            try:
                response = await client.get(engine + "/metrics")
                response.raise_for_status()
                sample.update(gauges=parse_gauges(response.text), raw=response.text)
            except httpx.HTTPError as exc:
                sample["error"] = str(exc)
            sample["end_epoch"] = time.time()
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            try:
                await asyncio.wait_for(stop.wait(), 1)
            except TimeoutError:
                pass


def percentile(values, fraction):
    ordered = sorted(values)
    x = (len(ordered) - 1) * fraction
    lo = int(x)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)] - ordered[lo]) * (x-lo)


async def cell(client, url, model, trial, concurrency):
    folder = Path(f"trial-{trial}/c-{concurrency}")
    folder.mkdir(parents=True)
    rows = []
    issued = 0
    epoch, start = time.time(), time.monotonic()
    with (folder / "requests.jsonl").open("x") as handle:
        async def consume():
            nonlocal issued
            while time.monotonic()-start < 30 or issued < max(12, 2*concurrency):
                i = issued
                issued += 1
                row = await generate(client, url, model, i)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
                if "error" in row:
                    raise RuntimeError(row["error"])
        async with asyncio.TaskGroup() as group:
            for _ in range(concurrency):
                group.create_task(consume())
    elapsed = time.monotonic()-start
    result = dict(trial=trial, concurrency=concurrency, start_epoch=epoch, end_epoch=time.time(),
                  requests=len(rows), output_tokens=sum(r["usage"]["completion_tokens"] for r in rows),
                  elapsed_seconds=elapsed, requests_per_second=len(rows)/elapsed,
                  output_tokens_per_second=256*len(rows)/elapsed,
                  p95_latency_s=percentile([r["seconds"] for r in rows], .95),
                  p50_ttft_s=percentile([r["content_seconds"][0] for r in rows], .5),
                  p50_tpot_s=percentile([(r["content_seconds"][-1]-r["content_seconds"][0])/255 for r in rows], .5))
    (folder / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


async def benchmark(url, engine, model, student, token, cap):
    private = {"Authorization": "Bearer " + token}
    async with (httpx.AsyncClient(headers=private, trust_env=False, timeout=10) as direct,
                httpx.AsyncClient(headers={"Authorization":"Bearer "+student}, trust_env=False, timeout=180,
                                  limits=httpx.Limits(max_connections=128)) as client):
        stop = asyncio.Event()
        async with asyncio.TaskGroup() as group:
            group.create_task(monitor(direct, engine, "engine.jsonl.gz", stop))
            try:
                for i in range(2):
                    row = await generate(client, url, model, i)
                    if "error" in row:
                        raise RuntimeError(row["error"])
                await asyncio.sleep(5)
                for trial in range(3):
                    levels = [c for c in (1,2,4,8,16,32,64) if c <= cap]
                    random.Random(431+trial).shuffle(levels)
                    for concurrency in levels:
                        await cell(client, url, model, trial, concurrency)
                        await asyncio.sleep(2)
            finally:
                stop.set()
