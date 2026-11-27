"""Exercise an existing, private SGLang GPU worker on loopback over real HTTP.

Set TOKENFLOW_TEST_SGLANG_PORT, TOKENFLOW_TEST_SGLANG_MODEL and
TOKENFLOW_TEST_SGLANG_MASTER_FILE. The worker and adjacent GPU exporter must
use worker_token(master, "a" * 32). Set TOKENFLOW_TEST_SGLANG_EVIDENCE to a
new JSON file under experiments/results to keep a timestamped record.
"""

import asyncio
import json
import os
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.provisioner.base import Resource
from tokenflow.provisioner.slurm import worker_token
from tokenflow.worker_manager.telemetry import read_telemetry


class ExistingAllocation:
    """Only provider inventory is substituted: the worker is owned by its caller."""

    def __init__(self, endpoint):
        self.resource = Resource("a" * 32, "local-gpu", "running", endpoint)

    async def discover(self, pool):
        return [self.resource]

    async def launch(self, pool, key):
        raise AssertionError("Test cannot launch workers")

    async def terminate(self, pool, resource):
        raise AssertionError("Test cannot terminate workers")


@pytest.mark.skipif(not os.getenv("TOKENFLOW_TEST_SGLANG_PORT"), reason="Requires a running private SGLang worker")
def test_real_sglang_worker(tmp_path):
    from test_openai_client import loopback_server, openai
    from test_slurm_worker_live import check_completions, now

    port = int(os.environ["TOKENFLOW_TEST_SGLANG_PORT"])
    assert 1024 <= port <= 65534
    model = os.environ["TOKENFLOW_TEST_SGLANG_MODEL"]
    engine, exporter = (f"http://127.0.0.1:{p}" for p in (port, port + 1))
    allocated = ExistingAllocation(engine)
    master = Path(os.environ["TOKENFLOW_TEST_SGLANG_MASTER_FILE"]).read_text().strip()
    assert master, "Gateway master file must contain a key"
    credential = {"Authorization": "Bearer " + worker_token(master, allocated.resource.key)}
    operator, student = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    config = tmp_path / "platform.json"
    config.write_text(json.dumps({
        "namespace": "local-sglang", "state_path": "state.sqlite3", "interval_seconds": 0.25,
        "pools": [{"name": "local-gpu", "provider": "slurm", "backend": "sglang", "model": model,
                   "min_replicas": 1, "max_replicas": 1, "capacity": 2,
                   "checkpoint": "/preloaded", "image": "/prestarted.sif",
                   "gpu_type": os.getenv("TOKENFLOW_TEST_SGLANG_GPU", "4060"),
                   "options": {"state_dir": "/unused-state", "account": "local", "partition": "local"}}],
    }))
    app = create_app(Settings(_env_file=None, platform_config=config, worker_secret=master,
                              admin_key=operator, api_keys=[student], worker_health_interval_seconds=.25,
                              worker_health_timeout_seconds=5), providers={"slurm": allocated})
    workload = {"model": model, "messages": [{"role": "user", "content": "List integers from 1 to 100, one per line."}],
                "temperature": 0, "max_tokens": 64}
    report = {
        "started_at_utc": now(), "backend": "sglang", "model": model, "sdk_version": openai.__version__,
        "scope": "Real existing worker, GPU exporter, gateway and SDK over loopback TCP. Slurm inventory is emulated.",
        "workload": workload, "worker_samples": [], "gpu_samples": [], "stream_chunks": [],
    }
    output_name = os.getenv("TOKENFLOW_TEST_SGLANG_EVIDENCE")
    if output_name:
        output = Path(output_name).resolve()
        root = Path(__file__).resolve().parents[1] / "experiments/results"
        assert root.resolve() in output.parents and output.parent.is_dir() and not output.exists()

    async def exercise():
        async with (
            loopback_server(app) as url,
            httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {operator}"}, trust_env=False) as admin,
            httpx.AsyncClient(trust_env=False, timeout=10) as direct,
            openai.AsyncOpenAI(base_url=url + "/v1", api_key=student, max_retries=0, timeout=90,
                               http_client=httpx.AsyncClient(trust_env=False)) as sdk,
        ):
            auth = {}
            for name, target, expected in [
                ("gateway_models", url + "/v1/models", 401), ("worker_models", engine + "/v1/models", 401),
                ("engine_health", engine + "/health", 200), ("engine_metrics", engine + "/metrics", 200),
                ("physical_gpu_metrics", exporter + "/metrics", 401),
            ]:
                auth[name] = (await direct.get(target)).status_code
                assert auth[name] == expected, name
            invalid = {"Authorization": f"Bearer {student}"}
            for label, headers in [("no_key_chat", {}), ("student_key_direct_chat", invalid)]:
                auth[label] = (await direct.post(engine + "/v1/chat/completions", json=workload, headers=headers)).status_code
                assert auth[label] == 401
            report["auth_checks"] = {"checked_at_utc": now(), **auth}
            catalog = (await direct.get(engine + "/v1/models", headers=credential)).raise_for_status().json()["data"]
            assert model in [m["id"] for m in catalog]
            # Native server_info can contain the resolved API key. Keep only its version.
            version = (await direct.get(engine + "/server_info", headers=credential)).raise_for_status().json().get("version")
            assert isinstance(version, str) and re.fullmatch(r"[0-9][a-zA-Z0-9.+_-]{0,79}", version)
            report["engine_version"] = version
            assert model in [m.id for m in (await sdk.models.list()).data]

            async def worker_sample():
                workers = (await admin.get("/v1/workers")).raise_for_status().json()
                if not workers:
                    return None
                [worker] = workers
                assert worker["metadata"]["backend"] == "sglang" and worker["base_url"].rstrip("/") == engine
                assert worker["metadata"]["gpu_metrics_port"] == port + 1
                row = {k: worker[k] for k in ("id", "active_requests", "health_status", "telemetry")}
                row["received_at_utc"] = now()
                report["worker_samples"].append(row)
                return row

            async def sample_gpu():
                metrics = (await direct.get(exporter + "/metrics", headers=credential)).raise_for_status().text
                rows = re.findall(r'^DCGM_FI_DEV_GPU_UTIL\{UUID="(GPU-[0-9a-fA-F-]+)"\} (\d+)$', metrics, re.M)
                assert len(rows) == 1 and 0 <= int(rows[0][1]) <= 100
                report["gpu_samples"].append({"received_at_utc": now(), "uuid": rows[0][0], "utilization_percent": int(rows[0][1])})

            async with asyncio.timeout(60):
                while True:
                    worker = await worker_sample()
                    if worker and worker["health_status"] == "healthy":
                        break
                    await asyncio.sleep(.1)

            async def first_content():
                # This observation separates actual incremental delivery from a
                # complete response replayed to the SDK as a series of chunks.
                row = await worker_sample()
                assert row["active_requests"] == 1, "First content was not observed while generation was still in flight"
                report["first_content_while_reserved_at_utc"] = row["received_at_utc"]

            done = asyncio.Event()

            async def sample():
                while not done.is_set():
                    await worker_sample()
                    await sample_gpu()
                    await asyncio.sleep(.2)

            async with asyncio.timeout(240), asyncio.TaskGroup() as tasks:
                tasks.create_task(sample())
                try:
                    await check_completions(sdk, workload, report, on_first_content=first_content)
                    report["stream_ended_at_utc"] = now()
                    async with asyncio.timeout(60):
                        while True:
                            row = await worker_sample()
                            gauges = row["telemetry"] or {}
                            if row["active_requests"] == 0 and all(gauges.get(k) is not None for k in (
                                "running", "waiting", "kv_cache_usage", "gpu_utilization",
                            )):
                                break
                            await asyncio.sleep(.25)
                finally:
                    done.set()
            chunks = report["stream_chunks"]
            assert len([c for r in chunks for c in r["choices"] if c["content"]]) >= 2
            assert any(c["finish_reason"] in {"length", "stop"} for r in chunks for c in r["choices"])
            for usage in [report["json_reply"]["usage"], chunks[-1]["usage"]]:
                assert usage and usage["completion_tokens"] > 0 and usage["prompt_tokens"] > 0
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert report["gpu_samples"]
            native = (await direct.get(engine + "/metrics")).raise_for_status().text
            engine_gauges = read_telemetry("\n".join(s for s in native.splitlines() if s.startswith("sglang:")))
            assert engine_gauges and all(getattr(engine_gauges, field) is not None for field in (
                "running", "waiting", "kv_cache_usage",
            ))
            report["native_engine_sample"] = engine_gauges.model_dump(mode="json", include={
                "observed_at", "running", "waiting", "kv_cache_usage",
            })

    try:
        asyncio.run(exercise())
        report["passed_at_utc"] = now()
    finally:
        if output_name:
            if "passed_at_utc" not in report:
                report["failed_at_utc"] = now()
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                output = output.with_name(f"{output.stem}-failed-{stamp}.json")
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
