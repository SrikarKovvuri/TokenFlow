"""Repeatable prompt/length/output/concurrency sweeps against a chat endpoint."""
import argparse
import asyncio
import hashlib
import itertools
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx


def utc_now():
    return datetime.now(UTC).isoformat()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def prompt_for(task, words, seed):
    rng = random.Random(seed)
    subjects = ["router", "worker", "queue", "request", "cache", "batch", "replica"]
    # Varied records reduce identical-prefix reuse; no model tokenizer is assumed.
    context = []
    while len(context) < words:
        context.extend(f"Record {rng.randrange(1000000)} {rng.choice(subjects)} latency {rng.randrange(1, 500)} ms.".split())
    context = " ".join(context[:words])
    return f"Reference records:\n{context}\n\nTask: {task}"


def success(row):
    return 200 <= row.get("status_code", 0) < 300 and "error" not in row


async def request_once(client, target, payload, index, timeout):
    started = perf_counter()
    row = {"index": index, "started_at": utc_now(),
           "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
           "prompt_characters": len(payload["messages"][0]["content"]), "max_tokens": payload["max_tokens"]}
    chunks = []
    try:
        # Whole-request deadline, in addition to HTTPX's per-I/O timeouts.
        async with asyncio.timeout(timeout):
            async with client.stream("POST", target.rstrip("/") + "/v1/chat/completions", json=payload) as response:
                row.update(status_code=response.status_code,
                           request_id=response.headers.get("X-Request-ID"))
                if payload["stream"] and response.is_success:
                    event_lines = []
                    done = False
                    async def event(data):
                        nonlocal done
                        if data == "[DONE]":
                            done = True
                            return
                        item = json.loads(data)
                        if not isinstance(item, dict) or not isinstance(item.get("choices", []), list):
                            raise ValueError("Invalid SSE response object")
                        if "error" in item:
                            row["error"] = str(item["error"])
                        if item.get("usage"):
                            row["usage"] = item["usage"]
                        for choice in item.get("choices", []):
                            if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                                raise ValueError("Invalid SSE choice")
                            text = choice.get("delta", {}).get("content")
                            if text is not None and not isinstance(text, str):
                                raise ValueError("Invalid SSE content")
                            if text:
                                row.setdefault("time_to_first_content_seconds", perf_counter() - started)
                                chunks.append(text)
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            event_lines.append(line[5:].lstrip())
                        elif not line and event_lines:
                            await event("\n".join(event_lines))
                            event_lines.clear()
                    if event_lines:
                        await event("\n".join(event_lines))
                    if not done:
                        row.setdefault("error", "SSE ended without [DONE]")
                    row["output"] = "".join(chunks)
                else:
                    body = await response.aread()
                    try:
                        item = json.loads(body)
                        row["response"] = item
                        if response.is_success and (not isinstance(item, dict) or not isinstance(item.get("choices"), list)):
                            row["error"] = "Invalid chat completion response"
                        if isinstance(item, dict) and item.get("usage"):
                            row["usage"] = item["usage"]
                    except ValueError:
                        row["response"] = body.decode(errors="replace")
                        if response.is_success:
                            row["error"] = "Invalid JSON chat completion response"
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(finished_at=utc_now(), latency_seconds=perf_counter() - started)
    return row


def summarize(rows, elapsed):
    good = [r for r in rows if success(r)]
    latencies = [r["latency_seconds"] for r in good]
    ttfts = [r["time_to_first_content_seconds"] for r in good if "time_to_first_content_seconds" in r]
    tokens = [r["usage"]["completion_tokens"] for r in good
              if isinstance(r.get("usage"), dict) and isinstance(r["usage"].get("completion_tokens"), int)]
    return {"requests": len(rows), "successes": len(good), "failures": len(rows) - len(good),
            "elapsed_seconds": elapsed, "successful_requests_per_second": len(good) / elapsed,
            "latency_p50_seconds": percentile(latencies, .5), "latency_p95_seconds": percentile(latencies, .95),
            "latency_p99_seconds": percentile(latencies, .99),
            "first_content_p50_seconds": percentile(ttfts, .5), "first_content_p95_seconds": percentile(ttfts, .95),
            "usage_coverage_requests": len(tokens),
            "completion_tokens_mean": sum(tokens) / len(tokens) if tokens else None,
            "completion_tokens_per_second": sum(tokens) / elapsed if tokens and len(tokens) == len(good) else None}


def validate(config):
    for key in ("prompt_words", "max_tokens", "concurrency"):
        values = config[key]
        if not values or any(type(n) is not int or n <= 0 for n in values):
            raise ValueError(f"{key} must contain positive integers")
    for key in ("requests_per_cell", "trials"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["warmup_requests"]) is not int or config["warmup_requests"] < 0:
        raise ValueError("warmup_requests must be a nonnegative integer")
    if not config["stream_modes"] or any(type(v) is not bool for v in config["stream_modes"]):
        raise ValueError("stream_modes must contain booleans")
    if not config["tasks"] or any(not isinstance(t.get("instruction"), str) or not t.get("id") for t in config["tasks"]):
        raise ValueError("tasks require id and instruction")
    for key in ("timeout_seconds", "cooldown_seconds"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == "timeout_seconds" and value == 0):
            raise ValueError(f"invalid {key}")


async def run_suite(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate(config)
    if args.smoke:
        config.update(tasks=config["tasks"][:1], prompt_words=config["prompt_words"][:1],
                      max_tokens=config["max_tokens"][:1], concurrency=[1, 2],
                      stream_modes=[False, True], trials=1, requests_per_cell=2, warmup_requests=1)
    dimensions = itertools.product(config["tasks"], config["prompt_words"], config["max_tokens"],
                                   config["concurrency"], config["stream_modes"], range(config["trials"]))
    cells = [dict(task=task, prompt_words=words, max_tokens=limit, concurrency=concurrency,
                  stream=stream, trial=trial) for task, words, limit, concurrency, stream, trial in dimensions]
    random.Random(config["seed"]).shuffle(cells)
    print(f"Plan: {len(cells)} cells, {len(cells) * config['requests_per_cell']} measured requests, "
          f"{len(cells) * config['warmup_requests']} warmup requests", flush=True)
    if args.dry_run:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "suite-source.txt").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"started_at": utc_now(), "model": args.model,
        "target": args.target, "config": config, "cells": cells,
        "load_model": "closed-loop fixed concurrency", "prompt_unit": "context words, not tokens"}, indent=2))
    limit = max(config["concurrency"])
    async with httpx.AsyncClient(timeout=config["timeout_seconds"],
            limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit)) as client:
        with (output / "cells.jsonl").open("w", encoding="utf-8") as summaries:
            for cell_id, cell in enumerate(cells):
                def payload(index):
                    # Same request corpus across concurrency/stream/trial cells for a given task/length.
                    seed = f"{config['seed']}:{cell['task']['id']}:{cell['prompt_words']}:{index}"
                    item = {"model": args.model, "messages": [{"role": "user", "content": prompt_for(
                        cell["task"]["instruction"], cell["prompt_words"], seed)}],
                        "max_tokens": cell["max_tokens"], "temperature": 0, "stream": cell["stream"]}
                    if cell["stream"]:
                        item["stream_options"] = {"include_usage": True}
                    return item
                warmup = [await request_once(client, args.target, payload(-i-1), i,
                                             config["timeout_seconds"]) for i in range(config["warmup_requests"])]
                (output / f"warmup-{cell_id:04d}.json").write_text(json.dumps(warmup))
                rows = []
                indices = iter(range(config["requests_per_cell"]))
                started_at = utc_now()
                started = perf_counter()
                with (output / f"requests-{cell_id:04d}.jsonl").open("w", encoding="utf-8") as requests:
                    async def consumer():
                        for index in indices:
                            row = await request_once(client, args.target, payload(index), index, config["timeout_seconds"])
                            requests.write(json.dumps(row) + "\n")
                            requests.flush()
                            rows.append(row)
                    await asyncio.gather(*(consumer() for _ in range(min(cell["concurrency"], config["requests_per_cell"]))))
                elapsed = perf_counter() - started
                result = {"cell_id": cell_id, **cell, "started_at": started_at, "finished_at": utc_now(),
                          "warmup_failures": sum(not success(r) for r in warmup), **summarize(rows, elapsed)}
                summaries.write(json.dumps(result) + "\n")
                summaries.flush()
                print(f"Cell {cell_id+1}/{len(cells)}: {result['successes']}/{len(rows)} succeeded; "
                      f"p95={result['latency_p95_seconds']}", flush=True)
                await asyncio.sleep(config["cooldown_seconds"])
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/suite.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", default="http://localhost:8000")
    parser.add_argument("--output", default="experiments/results/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(run_suite(parser.parse_args()))
