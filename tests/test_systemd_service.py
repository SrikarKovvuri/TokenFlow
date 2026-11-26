"""Optional process-level restart check using an unprivileged Linux systemd manager."""

import configparser
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

import httpx
import pytest


def test_gateway_recovers_its_pool_policy_after_a_process_crash():
    if sys.platform != "linux" or os.getenv("TOKENFLOW_TEST_SYSTEMD") != "1":
        pytest.skip("Set TOKENFLOW_TEST_SYSTEMD=1 on Linux with a running user systemd manager")
    cli = shutil.which("tokenflow")
    assert cli, "Install the project and put its tokenflow executable on PATH"
    evidence = os.getenv("TOKENFLOW_TEST_SYSTEMD_EVIDENCE")
    if evidence:
        assert Path(evidence).parent.is_dir() and not Path(evidence).exists()
    runtime = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env = {**os.environ, "XDG_RUNTIME_DIR": runtime}
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")
    unit = f"tokenflow-test-{uuid4().hex}.service"

    def systemctl(*args):
        return subprocess.check_output(["systemctl", "--user", *args, unit], env=env, text=True, timeout=10).strip()

    unit_config = configparser.ConfigParser(interpolation=None)
    unit_config.read(Path(__file__).resolve().parents[1] / "deploy/tokenflow.service")
    settings = unit_config["Service"]
    policy = {key: settings[key] for key in ("Type", "Restart", "RestartSec", "TimeoutStopSec", "UMask")}
    assert policy["Restart"] == "on-failure"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="tokenflow-systemd-") as temp, httpx.Client(
        base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=1,
    ) as client:
        root = Path(temp)
        commands = root / "bin"
        commands.mkdir()
        trace = root / "slurm-calls"
        for name in ("squeue", "sbatch", "scancel"):
            stub = commands / name
            stub.write_text(f"#!/bin/sh\necho '{name}' \"$PPID\" >> '{trace}'\nexit {0 if name == 'squeue' else 1}\n")
            stub.chmod(0o700)
        config = root / "catalog.json"
        config.write_text(json.dumps({
            "namespace": "restart-test", "state_path": "state.sqlite3", "interval_seconds": .2,
            "pools": [{"name": "campus", "provider": "slurm", "model": "chat", "checkpoint": "/models/test",
                       "gpu_type": "A6000", "image": "/worker.sif", "min_replicas": 0, "max_replicas": 1,
                       "options": {"partition": "test", "account": "test", "state_dir": str(root / "jobs")}}],
        }))
        student, operator = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        keys = root / "keys.env"
        keys.touch(mode=0o600)
        keys.write_text(f"PATH={commands}\nHOME={root}\nTOKENFLOW_ADMIN_KEY={operator}\n"
                        f"TOKENFLOW_API_KEYS='{json.dumps([student])}'\nTOKENFLOW_WORKER_SECRET={secrets.token_urlsafe(32)}\n")
        operator_headers = {"Authorization": f"Bearer {operator}"}

        def check_running(previous_pid=0):
            deadline = time.monotonic() + float(policy["RestartSec"]) + 20
            while time.monotonic() < deadline:
                pid = int(systemctl("show", "--value", "-p", "MainPID") or 0)
                if pid and pid != previous_pid and trace.exists() and f"squeue {pid}" in trace.read_text().splitlines():
                    try:
                        if client.get("/healthz").is_success:
                            break
                    except httpx.TransportError:
                        pass
                time.sleep(.1)
            else:
                pytest.fail("The service failed to discover its pool and become ready before the deadline")
            student_headers = {"Authorization": f"Bearer {student}"}
            anonymous = client.get("/v1/models")
            catalog = client.get("/v1/models", headers=student_headers).raise_for_status()
            student_pools = client.get("/v1/pools", headers=student_headers)
            operator_pools = client.get("/v1/pools", headers=operator_headers).raise_for_status()
            models = [row["id"] for row in catalog.json()["data"]]
            assert anonymous.status_code == student_pools.status_code == 401 and models == ["chat"]
            [pool] = operator_pools.json()
            assert pool["resources"] == [] and pool["last_error"] is None
            return {"observed_at": datetime.now(UTC).isoformat(), "pid": pid, "model": models[0],
                    "anonymous_model_status": anonymous.status_code, "student_model_status": catalog.status_code,
                    "student_pool_status": student_pools.status_code, "operator_pool_status": operator_pools.status_code,
                    "maximum": pool["max_replicas"], "resources": len(pool["resources"])}

        try:
            subprocess.run(["systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
                            *(f"--property={key}={value}" for key, value in policy.items()),
                            f"--property=WorkingDirectory={root}", f"--property=EnvironmentFile={keys}",
                            cli, "serve", "--config", str(config), "--port", str(port)],
                           env=env, check=True, timeout=10)
            before = check_running()
            assert before["maximum"] == 1
            client.put("/v1/pools/campus/limits", json={"minimum": 0, "maximum": 0},
                       headers=operator_headers).raise_for_status()
            database = root / "state.sqlite3"
            inode = database.stat().st_ino
            started = time.monotonic()
            systemctl("kill", "--kill-who=main", "--signal=SIGKILL")
            after = check_running(previous_pid=before["pid"])
            elapsed = round(time.monotonic() - started, 3)
            assert after["maximum"] == 0 and database.stat().st_ino == inode
            restarts = int(systemctl("show", "--value", "-p", "NRestarts"))
            assert restarts == 1
            calls = trace.read_text().splitlines()
            assert all(line.split()[0] == "squeue" for line in calls)
        finally:
            try:
                systemctl("stop")
            except subprocess.CalledProcessError:
                if systemctl("show", "--value", "-p", "LoadState") != "not-found":
                    raise
        assert systemctl("show", "--value", "-p", "LoadState") == "not-found"
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    if evidence:
        with Path(evidence).open("x", encoding="utf-8") as output:
            json.dump({"scope": "Transient user systemd unit and loopback gateway; Slurm inventory emulated",
                       "service_properties": policy, "before": before, "after": after,
                       "persisted_pause": True, "restart_count": restarts, "recovery_seconds": elapsed,
                       "discovery_polls": len(calls), "unit_collected": True, "listener_closed": True,
                       "temporary_state_removed": not root.exists()}, output, indent=2)
            output.write("\n")
