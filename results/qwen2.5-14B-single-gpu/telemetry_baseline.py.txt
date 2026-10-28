"""Rerun the fixed baseline with timestamped vLLM scrapes; HPC monitoring stays separate."""
import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from smoke import (SetupError, base_url, discover_model, gateway_running,
                   register_worker, show_results, start_gateway, wait_gateway, wait_worker)
from suite import prompt_for, run_suite
from summarize import export
from vllm_metrics import MetricsRecorder

HERE = Path(__file__).resolve().parent


async def preflight(client, worker, model, config):
    counts = []
    for task in config["tasks"]:
        for words in config["prompt_words"]:
            for index in range(-config["warmup_requests"], config["requests_per_cell"]):
                seed = f"{config['seed']}:{task['id']}:{words}:{index}"
                response = await client.post(worker + "/tokenize", json={"model": model,
                    "messages": [{"role": "user", "content": prompt_for(task["instruction"], words, seed)}],
                    "add_generation_prompt": True})
                response.raise_for_status()
                item = response.json()
                if item["count"] + max(config["max_tokens"]) > item["max_model_len"]:
                    raise SetupError("Baseline exceeds the worker context window")
                counts.append({"task": task["id"], "words": words, "index": index,
                               "input_tokens": item["count"], "max_model_len": item["max_model_len"]})
    return counts


async def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = HERE / "baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    process = None
    log = (output / "gateway.log").open("x", encoding="utf-8")
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            model = await discover_model(client, args.worker, args.model)
            metadata = {"created_at": datetime.now(UTC).isoformat(), "worker": args.worker,
                        "gateway": args.gateway, "models": (await client.get(args.worker + "/v1/models")).json(),
                        "gpu_telemetry": "Separate HPC monitor; join its file after retrieval", "config": config}
            version = await client.get(args.worker + "/version")
            metadata["version"] = version.json() if version.is_success else "unavailable"
            (output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            counts = await preflight(client, args.worker, model, config)
            (output / "token-preflight.json").write_text(json.dumps(counts), encoding="utf-8")
            if not await gateway_running(client, args.gateway):
                process = start_gateway(args.gateway, log)
                await wait_gateway(client, args.gateway, process, 45)
            worker_id = await register_worker(client, args.gateway, args.worker)
            await wait_worker(client, args.gateway, worker_id, 45)
        # Preserve the measurement code in text files to avoid Python reload watchers.
        for name in ("vllm_metrics.py", "telemetry_baseline.py", "baseline.json"):
            (output / (name + ".txt")).write_text((HERE / name).read_text(encoding="utf-8"), encoding="utf-8")
        async with MetricsRecorder(args.worker, output / "vllm-metrics.jsonl"):
            print("Collecting 10 seconds of idle vLLM metrics; HPC monitor should already be running.", flush=True)
            await asyncio.sleep(10)
            print("Starting the concurrency sweep now.", flush=True)
            await run_suite(argparse.Namespace(config=str(config_path), model=model,
                target=args.gateway, output=str(output / "serving"), smoke=False, dry_run=False))
            print("Sweep finished; collecting 10 seconds of idle vLLM metrics.", flush=True)
            await asyncio.sleep(10)
        rows = export(output / "serving", [str(output / "vllm-metrics.jsonl")])
        return show_results(rows, output / "serving", "Telemetry baseline")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5)
        log.close()


if __name__ == "__main__":
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=base_url, required=True)
    parser.add_argument("--gateway", type=base_url, default="http://127.0.0.1:8001")
    parser.add_argument("--model")
    parser.add_argument("--output", default=str(HERE / "results" /
        ("qwen-telemetry-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))))
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(run(args)))
    except (SetupError, httpx.HTTPError, OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as exc:
        print(f"Telemetry baseline failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
