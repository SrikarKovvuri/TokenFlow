"""Exercise the Slurm CLI boundary and packaged worker independently of a cluster."""

import asyncio
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tokenflow.provisioner import slurm, worker
from tokenflow.provisioner.base import PoolSpec, Resource


KEY = "3" * 32


def test_slurm_lifecycle_through_command_processes(tmp_path, monkeypatch):
    # Drive the actual slurm_command subprocess protocol with short-lived Python
    # executables. The fake queue also enforces named cancellation and ownership.
    for name, value in {"SBATCH_ARRAY_INX": "0-7", "SQUEUE_PARTITION": "another-partition",
                        "SCANCEL_INTERACTIVE": "1", "SLURM_CLUSTERS": "another-cluster",
                        "SLURM_CONF": "/test/slurm.conf"}.items():
        monkeypatch.setenv(name, value)
    cli = tmp_path / "scheduler.py"
    cli.write_text("""
from pathlib import Path
import json, os, sys
command, *args = sys.argv[1:]
assert all(key not in os.environ for key in ('SBATCH_ARRAY_INX', 'SQUEUE_PARTITION', 'SCANCEL_INTERACTIVE', 'SLURM_CLUSTERS'))
assert os.environ['SLURM_CONF'] == '/test/slurm.conf'
folder = Path(__file__).parent
record = folder / 'job.json'
if command == 'sbatch':
    assert '--parsable' in args and '--gres=gpu:A6000:2' in args
    assert '--export=NIL' in args and not record.exists()
    name = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--job-name='))
    (folder / 'worker.sh').write_text(sys.stdin.read(), newline='\\n')
    record.write_text(json.dumps({'name': name, 'state': 'PENDING'}))
    print('123;local-cluster')
elif command == 'squeue':
    assert '--me' in args and '--Format=JobID:0|,Name:0|,State:0|,BatchHost:0|,RestartCnt:0' in args
    job = json.loads(record.read_text())
    print('123', job['name'], job['state'], job.get('host', 'test-node'), job.get('restarts', 0), sep='|')
    print('124|someone-elses-job|RUNNING|test-node|0')
elif command == 'scancel':
    job = json.loads(record.read_text())
    assert args == ['--name', job['name'], '123']
    job['state'] = 'COMPLETING'
    record.write_text(json.dumps(job))
else:
    raise AssertionError(command)
""")
    run_process = subprocess.run

    def invoke_stub(args, **kwargs):
        return run_process([sys.executable, str(cli), *args], **kwargs)

    monkeypatch.setattr(slurm, "subprocess", SimpleNamespace(run=invoke_stub))
    state = tmp_path / "state"
    if os.name == "nt":
        # POSIX ownership and absolute bind paths belong to the deployment host.
        monkeypatch.setattr(slurm, "Path", lambda _: state)
        monkeypatch.setattr(slurm, "private_directory", lambda path: path.mkdir(exist_ok=True))
    pool = PoolSpec(
        name="qwen", model="qwen-test", provider="slurm", checkpoint="/shared/checkpoints/Qwen Model",
        image="/shared/images/vllm.sif", gpu_type="A6000", gpu_count=2, max_replicas=1,
        options={"account": "edu", "partition": "burst", "state_dir": str(state) if os.name != "nt" else "/state"},
    )
    provider = slurm.SlurmProvisioner("testing", "test-key")
    pending = asyncio.run(provider.launch(pool, KEY))
    assert pending == Resource(KEY, "123", "pending")
    assert asyncio.run(provider.discover(pool)) == [pending]
    record = tmp_path / "job.json"
    job = json.loads(record.read_text())
    job["state"] = "RUNNING"
    record.write_text(json.dumps(job))
    port_file = state / job["name"] / "port-123"
    port_file.write_text("invalid")
    assert asyncio.run(provider.discover(pool)) == [Resource(KEY, "123", "running")]
    port_file.write_text("41402")
    live = Resource(KEY, "123", "running", "http://test-node:41402")
    assert asyncio.run(provider.launch(pool, KEY)) == live
    # A site-triggered requeue keeps the job ID and its directory. The new
    # allocation can move to another node or return to the previous one.
    job.update(state="REQUEUED")
    record.write_text(json.dumps(job))
    assert asyncio.run(provider.discover(pool)) == [pending]
    job.update(state="RUNNING", restarts=1, host="second-node")
    record.write_text(json.dumps(job))
    assert asyncio.run(provider.discover(pool)) == [Resource(KEY, "123", "running")]
    (port_file.parent / "port-123-restart-1").write_text("42002")
    assert asyncio.run(provider.discover(pool)) == [
        Resource(KEY, "123", "running", "http://second-node:42002"),
    ]
    job.update(restarts=2)
    record.write_text(json.dumps(job))
    assert asyncio.run(provider.discover(pool)) == [Resource(KEY, "123", "running")]
    asyncio.run(provider.terminate(pool, live))
    assert asyncio.run(provider.discover(pool)) == [Resource(KEY, "123", "running")]
    job["state"] = "CANCELLED"
    record.write_text(json.dumps(job))
    assert asyncio.run(provider.discover(pool)) == [Resource(KEY, "123", "terminal")]
    check_apptainer_command(tmp_path)


def check_apptainer_command(tmp_path):
    # Execute the submitted script in Bash with a recording Apptainer function.
    # No GPU engine starts. Git Bash can check the same script on Windows.
    if os.name == "nt":
        bash = Path(os.environ.get("ProgramFiles", "C:/Program Files"), "Git/bin/bash.exe")
        if not bash.is_file():
            return
    else:
        bash = shutil.which("bash")
        if not bash:
            return
    env_file = tmp_path / "bash-env"
    env_file.write_text("""
apptainer() {
    test -f "$APPTAINER_TMPDIR/../runtime/worker.py"
    test "$APPTAINERENV_CUDA_VISIBLE_DEVICES" = '1,3'
    test "$APPTAINERENV_VLLM_CACHE_ROOT" = '/work/cache/vllm'
    printf '%s\\n' "$@" > "$CAPTURE"
}
""", newline="\n")
    env = {**os.environ, "BASH_ENV": env_file.as_posix(), "CAPTURE": (tmp_path / "command.txt").as_posix(),
           "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1", "CUDA_VISIBLE_DEVICES": "1,3",
           "TMPDIR": tmp_path.as_posix()}
    subprocess.run([str(bash), "worker.sh"], cwd=tmp_path, env=env, check=True, timeout=15)
    args = (tmp_path / "command.txt").read_text().splitlines()
    assert args[:5] == ["exec", "--cleanenv", "--nv", "--pwd", "/work"]
    assert "/shared/checkpoints/Qwen Model:/checkpoint:ro" in args
    assert args[args.index("--gpus") + 1] == "2"
    assert args[args.index("--key-file") + 1] == "/state/key"
    assert args[args.index("--port-file") + 1] == "/state/port-123-restart-1"
    assert not list(tmp_path.glob("tokenflow-worker.*"))


def test_publish_only_after_correct_model_and_surface_storage_errors(tmp_path, monkeypatch):
    port_file = tmp_path / "port-123"
    responses = iter([[], {"data": [{"id": "another-model"}]}, {"data": [{"id": "qwen"}]}])
    def open_model_list(request, timeout):
        assert not port_file.exists()
        assert request.full_url == "http://127.0.0.1:41402/v1/models"
        assert request.get_header("Authorization") == "Bearer example"
        return io.BytesIO(json.dumps(next(responses)).encode())

    opener = SimpleNamespace(open=open_model_list)
    monkeypatch.setattr(worker, "build_opener", lambda *_: opener)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    child = SimpleNamespace(poll=lambda: None)
    worker.publish_when_ready(child, port_file, 41402, "example", "qwen")
    assert port_file.read_text() == "41402"
    assert not port_file.with_suffix(".tmp").exists()

    opener.open = lambda *args, **kwargs: io.BytesIO(b'{"data":[{"id":"qwen"}]}')
    monkeypatch.setattr(worker.time, "sleep", Mock(side_effect=AssertionError("Port publication failed silently")))
    with pytest.raises(FileNotFoundError):
        worker.publish_when_ready(child, tmp_path / "missing" / "port-123", 41402, "example", "qwen")


def test_device_probe_uses_the_allocated_mask(tmp_path, monkeypatch):
    # The real probe starts another interpreter. Mimic torch's CUDA device API
    # there rather than replacing allocated_gpus or the child-process call.
    (tmp_path / "torch.py").write_text("""
import os
from types import SimpleNamespace
visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
cuda = SimpleNamespace(device_count=lambda: len(visible), get_device_properties=lambda i:
    SimpleNamespace(name='NVIDIA RTX A6000', uuid='GPU-' + visible[i]))
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "DEADBEEF,ABCDE012")
    assert worker.allocated_gpus(2, "A6000") == ["GPU-DEADBEEF", "GPU-ABCDE012"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "DEADBEEF,DEADBEEF")
    with pytest.raises(RuntimeError, match="duplicate"):
        worker.allocated_gpus(2, "A6000")


def test_gpu_metrics_require_a_distinct_sample_for_each_allocated_device(monkeypatch):
    uuids = ["GPU-first", "GPU-second"]
    samples = Mock(return_value="GPU-first, 75\nGPU-second, 12\n")
    monkeypatch.setattr(worker.subprocess, "check_output", samples)
    handler = worker.metrics_handler(uuids, "test-token")
    request = Mock(path="/metrics", headers={"Authorization": "Bearer test-token"}, wfile=io.BytesIO())
    handler.do_GET(request)
    request.send_response.assert_called_once_with(200)
    assert samples.call_args.args[0][:3] == ["nvidia-smi", "-i", "GPU-first,GPU-second"]
    assert b'UUID="GPU-second"} 12' in request.wfile.getvalue()
    samples.return_value = "GPU-first, 75\nGPU-first, 75\n"
    handler.do_GET(request)
    request.send_error.assert_called_once_with(503)
