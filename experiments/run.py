"""Run repeatable, non-streaming chat workloads against a running gateway."""
import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx


async def run(args: argparse.Namespace) -> None:
    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    payloads = workload["requests"]
    if not payloads or any(not isinstance(p, dict) or p.get("stream") for p in payloads):
        raise ValueError("requests must contain non-streaming chat request objects")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def send(index: int, payload: dict) -> dict:
            async with semaphore:
                started = perf_counter()
                result = {"index": index, "request": payload}
                try:
                    response = await client.post(
                        args.gateway.rstrip("/") + "/v1/chat/completions", json=payload
                    )
                    result.update(status_code=response.status_code,
                                  request_id=response.headers.get("X-Request-ID"))
                    try:
                        result["response"] = response.json()
                    except ValueError:
                        result["response"] = response.text
                except httpx.RequestError as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                result["latency_seconds"] = perf_counter() - started
                return result

        started = perf_counter()
        results = await asyncio.gather(*(
            send(i, payload) for i, payload in enumerate(payloads * args.repeat)
        ))
        elapsed = perf_counter() - started
    successes = sum(200 <= r.get("status_code", 0) < 300 for r in results)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "gateway": args.gateway,
        "workload": str(args.workload),
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "summary": {"requests": len(results), "successes": successes,
                    "failures": len(results) - successes, "elapsed_seconds": elapsed,
                    "requests_per_second": len(results) / elapsed},
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {output}")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--workload", default="experiments/workload.example.json")
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", default="experiments/results/latest.json")
    asyncio.run(run(parser.parse_args()))
