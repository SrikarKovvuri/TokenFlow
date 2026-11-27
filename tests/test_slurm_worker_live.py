"""Cold-start the shipped Slurm worker on one local GPU behind the real gateway."""

import asyncio
import errno
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.provisioner import slurm


# Execute the submitted script's real, embedded worker and its arguments on the
# host. Only Apptainer's three bind mounts and container interpreter are mapped.
HOST_APPTAINER = r"""
apptainer() {
    local -A binds
    local src dst mode arg
    while (( $# )) && [[ "$1" != python3 ]]; do
        if [[ "$1" == --bind ]]; then
            IFS=: read -r src dst mode <<< "$2"
            binds["$dst"]="$src"
            shift
        fi
        shift
    done
    [[ "$1" == python3 && "$2" == /work/worker.py ]] || return 2
    shift
    local -a args=()
    for arg in "$@"; do
        case "$arg" in
            /work/*) arg="${binds[/work]}/${arg#/work/}" ;;
            /state/*) arg="${binds[/state]}/${arg#/state/}" ;;
            /checkpoint) arg="${binds[/checkpoint]}" ;;
            /checkpoint/*) arg="${binds[/checkpoint]}/${arg#/checkpoint/}" ;;
        esac
        args+=("$arg")
    done
    cd "${binds[/work]}"
    PYTHONPATH="$PWD" "$TEST_ENGINE_PYTHON" "${args[@]}" --loopback-only
}
"""


class LocalSlurm:
    """One local process group stands in for a single Slurm allocation."""

    job_id = "1"

    def __init__(self, root, interpreter):
        self.root = root
        self.name = None
        self.process = None
        self._owned_group = None
        self.cancelled = False
        shim = root / "host-apptainer.sh"
        shim.write_text(HOST_APPTAINER)
        self.env = {k: os.environ[k] for k in ("PATH", "LANG", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_DEVICE_ORDER")
                    if k in os.environ}
        self.env.update(TEST_ENGINE_PYTHON=str(interpreter), BASH_ENV=str(shim), SLURM_JOB_ID=self.job_id,
                        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                        DO_NOT_TRACK="1", PYTHONDONTWRITEBYTECODE="1")
        for key in ("HOME", "TMPDIR", "HF_HOME", "VLLM_CONFIG_ROOT", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH",
                    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            folder = root / key.lower()
            folder.mkdir()
            self.env[key] = str(folder)

    def command(self, *args, script=None):
        if args[0] == "sbatch":
            assert self.process is None and script, "Only one submission is permitted"
            self.name = next(arg.removeprefix("--job-name=") for arg in args if arg.startswith("--job-name="))
            with (self.root / "worker.log").open("xb") as log:
                self.process = subprocess.Popen(["bash", "-c", script], cwd=self.root, env=self.env,
                                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            self._owned_group = self.process.pid
            return self.job_id
        if args[0] == "squeue":
            if self.process is None:
                return ""
            phase = "CANCELLED" if self.cancelled else "FAILED" if self.process.poll() is not None else "RUNNING"
            return f"{self.job_id}|{self.name}|{phase}|127.0.0.1|0\n"
        if args == ("scancel", "--name", self.name, self.job_id):
            self.stop()
            self.cancelled = True
            return ""
        raise AssertionError(f"Unexpected Slurm command: {args[0]}")

    def stop(self):
        group, self._owned_group = self._owned_group, None
        if group is None:
            return
        try:
            os.killpg(group, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                self.process.poll()
                os.killpg(group, 0)
                time.sleep(.1)
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            self.process.wait(timeout=10)


@pytest.fixture
def local_job(monkeypatch):
    if sys.platform != "linux" or not os.environ.get("TOKENFLOW_TEST_SLURM_ENGINE_PYTHON"):
        pytest.skip("Set TOKENFLOW_TEST_SLURM_ENGINE_PYTHON on a Linux GPU host")
    interpreter = Path(os.environ["TOKENFLOW_TEST_SLURM_ENGINE_PYTHON"]).absolute()
    assert interpreter.is_file() and os.access(interpreter, os.X_OK)
    assert shutil.which("nvidia-smi"), "Put nvidia-smi on PATH"
    with tempfile.TemporaryDirectory(prefix="tokenflow-local-slurm-") as work:
        job = LocalSlurm(Path(work), interpreter)
        monkeypatch.setattr(slurm, "slurm_command", job.command)
        try:
            yield job
        finally:
            job.stop()


def now():
    return datetime.now(UTC).isoformat()


async def check_worker_auth(direct, engine, port, student, secret, key, report):
    report["unauthenticated_status"] = {}
    for label, target in [("engine_health", engine + "/health"), ("engine_metrics", engine + "/metrics"),
                          ("gpu_metrics", f"http://127.0.0.1:{port + 1}/metrics")]:
        resp = await direct.get(target)
        report["unauthenticated_status"][label] = resp.status_code
        assert resp.status_code == 401
    denied = await direct.get(engine + "/v1/models", headers={"Authorization": f"Bearer {student}"})
    report["student_key_direct_worker_status"] = denied.status_code
    assert denied.status_code == 401
    other_key = "0" * 32 if key != "0" * 32 else "1" * 32
    other_token = slurm.worker_token(secret, other_key)
    denied = await direct.get(engine + "/v1/models", headers={"Authorization": f"Bearer {other_token}"})
    report["different_job_token_status"] = denied.status_code
    assert denied.status_code == 401


async def check_completions(sdk, workload, report, *, on_first_content=None):
    model = workload["model"]
    # The warm worker has two free sequence slots. Exercise both before
    # starting the separate single-choice JSON and streaming requests.
    parallel = {**workload, "n": 2, "temperature": 0.7, "seed": 431}
    sent = now()
    reply = await sdk.chat.completions.create(**parallel)
    assert reply.model == model and sorted(c.index for c in reply.choices) == [0, 1]
    assert all(c.finish_reason in {"length", "stop"} and c.message.content and c.message.content.strip()
               for c in reply.choices)
    report["parallel_reply"] = {
        "sent_at_utc": sent, "received_at_utc": now(), "id": reply.id,
        "n": parallel["n"], "temperature": parallel["temperature"], "seed": parallel["seed"],
        "choices": [{"index": c.index, "content": c.message.content, "finish_reason": c.finish_reason}
                    for c in reply.choices],
    }

    report["json_sent_at_utc"] = now()
    reply = await sdk.chat.completions.create(**workload)
    choice = reply.choices[0]
    assert reply.model == model and choice.message.content and choice.finish_reason
    report["json_reply"] = {"received_at_utc": now(), "id": reply.id, "content": choice.message.content,
                            "finish_reason": choice.finish_reason, "usage": reply.usage.model_dump() if reply.usage else None}
    report["stream_sent_at_utc"] = now()
    async with await sdk.chat.completions.create(**workload, stream=True, stream_options={"include_usage": True}) as stream:
        async for chunk in stream:
            assert chunk.model == model
            report["stream_chunks"].append({"received_at_utc": now(), "id": chunk.id,
                "choices": [{"content": c.delta.content, "finish_reason": c.finish_reason} for c in chunk.choices],
                "usage": chunk.usage.model_dump() if chunk.usage else None})
            if on_first_content and any(c.delta.content for c in chunk.choices):
                await on_first_content()
                on_first_content = None
    choices = [c for row in report["stream_chunks"] for c in row["choices"]]
    assert any(c["finish_reason"] for c in choices) and any(c["content"] for c in choices)


def test_cold_start_authenticated_local_worker(tmp_path, local_job):
    from test_openai_client import loopback_server, openai

    namespace, pool, model = "local-acceptance", "tiny-gpu", "tokenflow-local-contract"
    root = local_job.root / "state"
    secret = secrets.token_urlsafe(32)
    checkpoint = Path(os.environ["TOKENFLOW_TEST_SLURM_CHECKPOINT"]).resolve(strict=True)
    assert checkpoint.is_dir(), "Set TOKENFLOW_TEST_SLURM_CHECKPOINT to a local checkpoint directory"
    output_name = os.environ.get("TOKENFLOW_TEST_SLURM_EVIDENCE")
    if output_name:
        output = Path(output_name)
        assert output.parent.is_dir() and not output.exists(), "Select a new JSON evidence file"
    config = tmp_path / "platform.json"
    config.write_text(json.dumps({
        "namespace": namespace, "state_path": "control.sqlite3", "interval_seconds": .25,
        "pools": [{"name": pool, "model": model, "backend": "vllm", "provider": "slurm",
                   "gpu_type": os.environ["TOKENFLOW_TEST_SLURM_GPU"], "capacity": 2, "checkpoint": str(checkpoint),
                   "image": "/worker.sif", "min_replicas": 0, "max_replicas": 1,
                   "options": {"account": "local", "partition": "local", "state_dir": str(root),
                               "dtype": "float16", "max_model_len": 512}}],
    }))
    student, operator = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    app = create_app(Settings(_env_file=None, platform_config=config, worker_secret=secret, api_keys=[student],
                              admin_key=operator, worker_health_interval_seconds=.25, worker_health_timeout_seconds=5))
    workload = {"model": model, "messages": [{"role": "user", "content": "List integers from 1 to 20."}],
                "temperature": 0, "max_tokens": 32}
    report = {"emulated": "Slurm CLI and Apptainer binds/interpreter; generated batch script executes on local Linux",
              "model": model, "job_id": local_job.job_id, "worker_samples": [], "stream_chunks": [], "workload": workload}

    async def exercise():
        async with (
            loopback_server(app) as url,
            httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {operator}"}, trust_env=False) as admin,
            httpx.AsyncClient(trust_env=False, timeout=4) as direct,
            openai.AsyncOpenAI(base_url=url + "/v1", api_key=student, max_retries=0, timeout=45,
                               http_client=httpx.AsyncClient(trust_env=False)) as sdk,
        ):
            assert model in [m.id for m in (await sdk.models.list()).data]
            assert local_job.process is None and not root.exists()
            with pytest.raises(openai.APIStatusError) as cold:
                await sdk.chat.completions.create(**workload)
            assert cold.value.status_code == 503 and cold.value.response.headers.get("x-tokenflow-admission") == "waiting-for-worker"
            report["cold_admission"] = {"received_at_utc": now(), "status": cold.value.status_code}
            ready, done = asyncio.Event(), asyncio.Event()

            async def sample():
                while not done.is_set():
                    if local_job.process and local_job.process.poll() is not None:
                        pytest.fail(f"Local worker exited during startup (exit {local_job.process.returncode})")
                    workers = (await admin.get("/v1/workers")).raise_for_status().json()
                    assert len(workers) <= 1
                    if workers:
                        worker = workers[0]
                        row = {k: worker[k] for k in ("id", "enabled", "health_status", "active_requests", "telemetry")}
                        report["worker_samples"].append({"received_at_utc": now(), **row})
                        gauges = row["telemetry"] or {}
                        if (row["health_status"] == "healthy" and gauges.get("gpu_utilization") is not None
                                and gauges.get("running") is not None):
                            ready.set()
                    await asyncio.sleep(.1)

            async with asyncio.timeout(300), asyncio.TaskGroup() as group:
                group.create_task(sample())
                try:
                    await ready.wait()
                    [worker] = (await admin.get("/v1/workers")).raise_for_status().json()
                    job_state = root / local_job.name
                    port = int((job_state / f"port-{local_job.job_id}").read_text())
                    assert 1024 <= port <= 65534
                    engine = f"http://127.0.0.1:{port}"
                    assert worker["base_url"].rstrip("/") == engine
                    assert worker["metadata"]["gpu_metrics_port"] == port + 1
                    [resource] = (await admin.get("/v1/pools")).raise_for_status().json()[0]["resources"]
                    key = resource["key"]
                    assert resource["external_id"] == (job_state / "job-id").read_text().strip() == local_job.job_id
                    assert resource["worker_id"] == worker["id"]
                    report.update(port_receipt=port, gpu_metrics_port=port+1, worker_id=worker["id"], job_phase=resource["phase"])
                    await check_worker_auth(direct, engine, port, student, secret, key, report)
                    await check_completions(sdk, workload, report)
                finally:
                    done.set()
            (await admin.put(f"/v1/pools/{pool}/limits", json={"minimum": 0, "maximum": 0})).raise_for_status()
            async with asyncio.timeout(30):
                while (await admin.get("/v1/pools")).raise_for_status().json()[0]["resources"]:
                    await asyncio.sleep(.1)
            assert not (await admin.get("/v1/workers")).raise_for_status().json()
            assert local_job.cancelled and local_job.process.poll() is not None
            for listener in (port, port + 1):
                with socket.socket() as probe:
                    probe.settimeout(2)
                    assert probe.connect_ex(("127.0.0.1", listener)) == errno.ECONNREFUSED
            report["closed_listeners"] = [port, port + 1]
            report["emulated_final_state"] = "CANCELLED"
            report["worker_process_exit"] = local_job.process.returncode

    asyncio.run(exercise())
    if output_name:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
