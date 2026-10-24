"""One-command local smoke test for an already reachable inference worker."""
import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

# Script imports also work when this file is loaded directly by tests.
EXPERIMENTS = Path(__file__).resolve().parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
from suite import run_suite
from summarize import export


class SetupError(RuntimeError):
    pass


def base_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise argparse.ArgumentTypeError("use an HTTP(S) base URL without a path, credentials, or query")
    try:
        parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value.rstrip("/")


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


async def discover_model(client, worker, requested):
    try:
        health = await client.get(worker + "/health")
        health.raise_for_status()
        response = await client.get(worker + "/v1/models")
        response.raise_for_status()
        body = response.json()
        models = [item["id"] for item in body["data"] if isinstance(item.get("id"), str)]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SetupError(f"Worker check failed at {worker}. Start the worker and SSH tunnel first. {exc}") from exc
    if requested:
        if requested not in models:
            raise SetupError(f"Model {requested!r} is not served. Available models: {models}")
        return requested
    if len(models) != 1:
        raise SetupError(f"Expected one served model, found {models}. Select one with --model.")
    return models[0]


async def gateway_running(client, gateway):
    try:
        response = await client.get(gateway + "/healthz")
    except httpx.ConnectError:
        return False
    except httpx.RequestError as exc:
        raise SetupError(f"Gateway probe failed at {gateway}: {exc}") from exc
    try:
        response.raise_for_status()
        body = response.json()
        if body.get("service") != "TokenFlow" or body.get("status") != "ok":
            raise ValueError("unexpected service")
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise SetupError(f"The service at {gateway} is not a healthy TokenFlow gateway; choose --gateway.") from exc
    return True


def start_gateway(gateway, log):
    parsed = urlsplit(gateway)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.scheme != "http":
        raise SetupError("Can only start a local HTTP gateway. Start the remote gateway yourself.")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Align newly started gateway I/O timeout with the smoke runner's 300-second deadline.
    env.setdefault("TOKENFLOW_WORKER_REQUEST_TIMEOUT_SECONDS", "300")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "tokenflow.main:app",
        "--host", parsed.hostname, "--port", str(parsed.port or 80)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


async def wait_gateway(client, gateway, process, timeout):
    async with asyncio.timeout(timeout):
        while True:
            if process.poll() is not None:
                raise SetupError("Gateway exited during startup. See the gateway log.")
            if await gateway_running(client, gateway):
                # Ensure our child did not lose a bind race to a different process.
                if process.poll() is not None:
                    raise SetupError("Gateway process exited. See the gateway log.")
                return
            await asyncio.sleep(.25)


async def register_worker(client, gateway, worker):
    response = await client.get(gateway + "/v1/workers")
    response.raise_for_status()
    workers = response.json()
    matching = next((w for w in workers if w["base_url"].rstrip("/") == worker), None)
    others = [w for w in workers if w["enabled"] and (matching is None or w["id"] != matching["id"])]
    if others:
        raise SetupError("Other enabled workers could receive the smoke requests: "
                         + ", ".join(w["name"] for w in others)
                         + ". Use an isolated gateway port with --gateway http://127.0.0.1:8001.")
    if matching is None:
        response = await client.post(gateway + "/v1/workers", json={
            "name": "smoke-worker", "base_url": worker})
        response.raise_for_status()
        matching = response.json()
    elif not matching["enabled"]:
        response = await client.patch(gateway + "/v1/workers/" + matching["id"], json={"enabled": True})
        response.raise_for_status()
        matching = response.json()
    return matching["id"]


async def wait_worker(client, gateway, worker_id, timeout):
    async with asyncio.timeout(timeout):
        while True:
            response = await client.get(gateway + "/v1/workers/" + worker_id)
            response.raise_for_status()
            worker = response.json()
            if worker["enabled"] and worker["health_status"] == "healthy":
                return
            await asyncio.sleep(.5)


def show_results(rows, output, label="Smoke"):
    failures = sum(row["failures"] for row in rows)
    warmup_failures = sum(row["warmup_failures"] for row in rows)
    total = sum(row["requests"] for row in rows)
    print(f"\n{label} result: {total - failures}/{total} measured requests passed; "
          f"{warmup_failures} warmup failures.")
    def timing(value):
        return f"{value:.3f}s" if value is not None else "unavailable"
    for row in sorted(rows, key=lambda r: (r["concurrency"], r["stream"])):
        print(f"  concurrency={row['concurrency']} {'SSE' if row['stream'] else 'JSON'}: "
              f"{row['successes']}/{row['requests']} passed, "
              f"p95 latency={timing(row['latency_p95_seconds'])}, "
              f"first content={timing(row['first_content_p50_seconds'])}, "
              f"{row['successful_requests_per_second']:.2f} successful requests/s")
    if failures or warmup_failures:
        errors = []
        for path in sorted(output.glob("requests-*.jsonl")):
            errors.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        for path in sorted(output.glob("warmup-*.json")):
            errors.extend(json.loads(path.read_text(encoding="utf-8")))
        for row in [r for r in errors if "error" in r or not 200 <= r.get("status_code", 0) < 300][:3]:
            print("  Failure:", str(row.get("error", row.get("response", row.get("status_code"))))[:500])
    print(f"CSV: {output / 'summary.csv'}")
    print(f"Raw results: {output}")
    print("HPC resource data is collected separately with monitor.py.")
    return 1 if failures or warmup_failures else 0


async def smoke(args):
    model = None
    process = log = None
    output = Path(args.output).resolve()
    if output.exists():
        raise SetupError(f"Output already exists: {output}. Choose a new --output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_name(output.name + "-gateway.log")
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            print(f"Checking worker at {args.worker} ...", flush=True)
            model = await discover_model(client, args.worker, args.model)
            print(f"Using model: {model}", flush=True)
            if await gateway_running(client, args.gateway):
                print(f"Reusing TokenFlow at {args.gateway}", flush=True)
            else:
                log = log_path.open("x", encoding="utf-8")
                process = start_gateway(args.gateway, log)
                print(f"Starting temporary TokenFlow at {args.gateway}; log: {log_path}", flush=True)
                try:
                    await wait_gateway(client, args.gateway, process, args.setup_timeout)
                except TimeoutError as exc:
                    raise SetupError(f"Gateway startup timed out. See {log_path}") from exc
            worker_id = await register_worker(client, args.gateway, args.worker)
            print("Waiting for worker health polling ...", flush=True)
            try:
                await wait_worker(client, args.gateway, worker_id, args.setup_timeout)
            except TimeoutError as exc:
                raise SetupError("Worker did not become healthy. Check that its URL is reachable from the gateway.") from exc
        baseline = getattr(args, "baseline", False)
        await run_suite(argparse.Namespace(
            config=str(EXPERIMENTS / ("baseline.json" if baseline else "suite.json")),
            smoke=not baseline, dry_run=False, output=str(output), model=model, target=args.gateway))
        rows = export(output, args.telemetry)
        return show_results(rows, output, "Baseline" if baseline else "Smoke")
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait, timeout=5)
            print("Stopped the temporary gateway started by this command.", flush=True)
        if log is not None:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=base_url,
                        default=os.environ.get("TOKENFLOW_SMOKE_WORKER_URL", "http://127.0.0.1:1810"))
    parser.add_argument("--gateway", type=base_url, default="http://127.0.0.1:8000")
    parser.add_argument("--baseline", action="store_true", help="Run the reusable 1,024-request model baseline")
    parser.add_argument("--model", help="Optional when the worker serves exactly one model")
    parser.add_argument("--setup-timeout", type=positive_seconds, default=45)
    parser.add_argument("--output")
    parser.add_argument("--telemetry", nargs="*", default=[], help="Optional already copied monitor files")
    args = parser.parse_args()
    if args.output is None:
        prefix = "baseline-" if args.baseline else "smoke-"
        args.output = str(EXPERIMENTS / "results" / (prefix + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")))
    try:
        return asyncio.run(smoke(args))
    except (SetupError, httpx.HTTPError, OSError, ValueError) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Smoke test interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
