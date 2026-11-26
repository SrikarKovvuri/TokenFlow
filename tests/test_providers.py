import asyncio
from argparse import Namespace
import json
from pathlib import Path
import shlex
from unittest.mock import MagicMock

import pytest

from tokenflow.provisioner import slurm, worker
from tokenflow.provisioner.base import PoolSpec, Resource
from tokenflow.provisioner.cloud import CloudOptions, CloudProvisioner, bootstrap
from tokenflow.provisioner.slurm import SlurmOptions, SlurmProvisioner, batch_script, worker_token


KEY = "a" * 32
OTHER_KEY = "b" * 32


def hpc_pool():
    return PoolSpec(name="team", model="qwen", checkpoint="/models/Qwen 14B", provider="slurm",
                    gpu_type="A6000", gpu_count=2, image="/images/vllm.sif", max_replicas=2,
                    options={"account": "edu", "partition": "burst", "state_dir": "/project/state",
                             "qos": "burst", "apptainer_module": "apptainer/1.4.5"})


def cloud_pool():
    return PoolSpec(name="team", model="qwen", checkpoint="/models/Qwen 14B", provider="cloud",
                    backend="sglang", gpu_type="L40S", image="example/sglang:stable", max_replicas=2,
                    options={"region": "us-west-2", "ami_id": "ami-123", "instance_type": "g6e.xlarge",
                             "subnet_id": "subnet-123", "security_group_ids": ["sg-123"]})


REVISION = "abcdef12" * 5


@pytest.mark.parametrize(("directory", "revision", "cached"), [
    ("models--org--chat-1.5B", REVISION, True),
    ("models--gpt2", REVISION, True),
    ("ordinary", REVISION, False),
    ("models--", REVISION, False),
    ("datasets--org--data", REVISION, False),
    ("models--org--model", "main", False),
    ("models--org--model", REVISION + "/subdirectory", False),
])
def test_local_checkpoint_mount_in_both_provider_commands(directory, revision, cached):
    repo = f"/shared model cache/{directory}"
    path = f"{repo}/snapshots/{revision}"
    source = repo if cached else path
    checkpoint = f"/checkpoint/snapshots/{revision}" if cached else "/checkpoint"
    hpc = hpc_pool().model_copy(update={"checkpoint": path})
    cloud = cloud_pool().model_copy(update={"checkpoint": path})

    batch = batch_script(hpc, SlurmOptions.model_validate(hpc.options), Path("/state"))
    native = shlex.split(batch[batch.rindex("\napptainer exec "):].replace("\\\n", " "))
    docker = shlex.split(bootstrap(cloud, CloudOptions.model_validate(cloud.options)).splitlines()[-1])
    assert f"{source}:/checkpoint:ro" in native
    assert f"type=bind,src={source},dst=/checkpoint,readonly" in docker
    for args in (native, docker):
        assert args[args.index("--checkpoint") + 1] == checkpoint


def test_cloud_hub_model_id_and_snapshot_mount_boundaries():
    pool = cloud_pool().model_copy(update={"checkpoint": "org/model"})
    opts = CloudOptions.model_validate(pool.options)
    docker = shlex.split(bootstrap(pool, opts).splitlines()[-1])
    assert docker[docker.index("--checkpoint") + 1] == "org/model"
    assert not any("dst=/checkpoint" in arg for arg in docker)

    repo = "/data/models--org--model"
    path = f"{repo}/snapshots/{REVISION}"
    for root in (repo, "/data"):
        CloudProvisioner.validate_pool(pool.model_copy(update={
            "checkpoint": path, "options": {**pool.options, "checkpoint_mount": root},
        }))
    for root in (path, f"{repo}/snapshots"):
        with pytest.raises(ValueError, match="checkpoint_mount"):
            CloudProvisioner.validate_pool(pool.model_copy(update={
                "checkpoint": path, "options": {**pool.options, "checkpoint_mount": root},
            }))


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_slurm_submission_adoption_and_scoped_termination(tmp_path, monkeypatch, backend):
    pool = hpc_pool().model_copy(update={"backend": backend})
    provider = SlurmProvisioner("deployment", "operator-secret")
    state = tmp_path / "private-state"
    # Slurm runs on POSIX. Map its configured shared path to a local fixture.
    monkeypatch.setattr(slurm, "Path", lambda _: state)
    monkeypatch.setattr(slurm, "private_directory", lambda path: path.mkdir(exist_ok=True))
    calls = []
    queue = ""

    def command(*args, script=None):
        calls.append((args, script))
        return {"sbatch": "123;insomnia\n", "squeue": queue, "scancel": ""}[args[0]]

    monkeypatch.setattr(slurm, "slurm_command", command)
    pending = asyncio.run(provider.launch(pool, KEY))
    assert pending == Resource(key=KEY, external_id="123", phase="pending")
    args, script = calls[0]
    assert "--gres=gpu:A6000:2" in args
    assert "--qos=burst" in args
    assert "--export=NIL" in args
    assert "operator-secret" not in script
    assert worker_token("operator-secret", KEY) not in script
    assert "apptainer exec --cleanenv --nv" in script
    name = next(arg.removeprefix("--job-name=") for arg in args if arg.startswith("--job-name="))
    token = worker_token("operator-secret", KEY)
    assert (state / name / "key").read_text() == token
    if backend == "sglang":
        assert (state / name / "sglang.yaml").read_text() == f'api-key: "{token}"\n'
        assert "APPTAINERENV_SGLANG_JIT_CACHE_DIR=/work/cache/sglang/jit" in script
    queue = f"123|{name}|RUNNING|ins081|0\n124|unrelated|RUNNING|ins081|0"
    assert asyncio.run(provider.discover(pool)) == [Resource(key=KEY, external_id="123", phase="running")]
    (state / name / "port-123").write_text("44100")
    live = asyncio.run(provider.launch(pool, KEY))
    assert live == Resource(key=KEY, external_id="123", phase="running", base_url="http://ins081:44100")
    assert [args[0] for args, _ in calls].count("sbatch") == 1
    asyncio.run(provider.terminate(pool, Resource(key=KEY, external_id="124", phase="running")))
    assert calls[-1][0][0] == "squeue"
    asyncio.run(provider.terminate(pool, live))
    assert calls[-1][0] == ("scancel", "--name", name, "123")


def test_ambiguous_slurm_submission_never_submits_again(tmp_path, monkeypatch):
    pool = hpc_pool()
    provider = SlurmProvisioner("test", "secret")
    submitted = []
    monkeypatch.setattr(slurm, "Path", lambda _: tmp_path)
    monkeypatch.setattr(slurm, "private_directory", lambda path: path.mkdir(exist_ok=True))

    def command(*args, script=None):
        if args[0] == "sbatch":
            submitted.append(script)
            raise TimeoutError
        return ""

    monkeypatch.setattr(slurm, "slurm_command", command)
    with pytest.raises(TimeoutError):
        asyncio.run(provider.launch(pool, KEY))
    with pytest.raises(RuntimeError, match="Ambiguous"):
        asyncio.run(provider.launch(pool, KEY))
    # An older provider could leave an empty receipt if the filesystem filled
    # after sbatch had already accepted this job.
    (tmp_path / (provider._prefix(pool) + KEY) / "job-id").write_text("")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        asyncio.run(provider.launch(pool, KEY))
    assert len(submitted) == 1


def test_confirmed_slurm_rejection_releases_claim_for_retry(tmp_path, monkeypatch):
    pool = hpc_pool()
    provider = SlurmProvisioner("test", "secret")
    monkeypatch.setattr(slurm, "Path", lambda _: tmp_path)
    monkeypatch.setattr(slurm, "private_directory", lambda path: path.mkdir(exist_ok=True))
    responses = [
        Namespace(returncode=1, stderr="sbatch: error: Batch job submission failed: Invalid qos specification", stdout=""),
        Namespace(returncode=0, stderr="", stdout="124"),
    ]
    monkeypatch.setattr(slurm.subprocess, "run", lambda *args, **kwargs: responses.pop(0))
    with pytest.raises(slurm.SubmissionRejected, match="Invalid qos"):
        asyncio.run(provider.launch(pool, KEY))
    assert asyncio.run(provider.launch(pool, KEY)).external_id == "124"


def test_slurm_requires_readable_shell_inputs_and_protected_state():
    pool = hpc_pool()
    invalid = pool.model_copy(update={"image": "/tmp/image:two.sif"})
    with pytest.raises(ValueError, match="Mount paths"):
        SlurmProvisioner.validate_pool(invalid)
    script = batch_script(pool, SlurmOptions.model_validate(pool.options), Path("/state/test"))
    assert "'/models/Qwen 14B:/checkpoint:ro'" in script
    assert "$work/runtime:/work" in script
    assert "APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/cache" in script


def test_private_ec2_launch_is_idempotent():
    pool = cloud_pool()
    client = MagicMock()
    client.run_instances.return_value = {"Instances": [{"InstanceId": "i-123"}]}
    provider = CloudProvisioner("test")
    provider._clients["us-west-2"] = client
    first = asyncio.run(provider.launch(pool, KEY))
    second = asyncio.run(provider.launch(pool, KEY))
    assert first == second == Resource(key=KEY, external_id="i-123", phase="pending")
    kwargs = client.run_instances.call_args.kwargs
    assert kwargs == client.run_instances.call_args_list[0].kwargs
    assert kwargs["NetworkInterfaces"] == [
        {"DeviceIndex": 0, "SubnetId": "subnet-123", "Groups": ["sg-123"],
         "AssociatePublicIpAddress": False, "DeleteOnTermination": True}
    ]
    assert kwargs["MetadataOptions"]["HttpTokens"] == "required"
    assert kwargs["MinCount"] == kwargs["MaxCount"] == 1
    assert len(kwargs["ClientToken"]) == 64
    assert "TokenFlowKey" in {item["Key"] for item in kwargs["TagSpecifications"][0]["Tags"]}
    # Validate request shape against botocore when the AWS extra is installed.
    boto3 = pytest.importorskip("boto3")
    from botocore.validate import validate_parameters
    sdk = boto3.client("ec2", region_name="us-west-2", aws_access_key_id="test", aws_secret_access_key="test")
    validate_parameters(kwargs, sdk.meta.service_model.operation_model("RunInstances").input_shape)


def test_ec2_follows_pages_and_checks_ownership_before_termination():
    pool = cloud_pool()
    provider = CloudProvisioner("test")
    client = MagicMock()
    provider._clients["us-west-2"] = client

    def page(instance, key, deployment="test"):
        tags = {"TokenFlowPool": "team", "TokenFlowNamespace": deployment, "TokenFlowKey": key}
        return {"Reservations": [{"Instances": [
            {"InstanceId": instance, "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
             "State": {"Name": "running"}, "PrivateIpAddress": "10.0.0.12"}]}]}

    pages = [page("i-11", KEY), page("i-22", OTHER_KEY), page("i-33", KEY, deployment="unrelated")]
    client.get_paginator.return_value.paginate.return_value = pages
    live, other = asyncio.run(provider.discover(pool))
    assert [live.key, other.key] == [KEY, OTHER_KEY]
    assert live.base_url == "http://10.0.0.12:8000"
    stopped = page("i-11", KEY)["Reservations"][0]["Instances"][0]
    stopped["State"]["Name"] = "stopped"
    assert provider._resource(pool, stopped) == Resource(key=KEY, external_id="i-11", phase="running")
    asyncio.run(provider.terminate(pool, Resource(key=KEY, external_id="i-33", phase="running")))
    client.terminate_instances.assert_not_called()
    asyncio.run(provider.terminate(pool, live))
    client.terminate_instances.assert_called_once_with(InstanceIds=["i-11"])


def test_cloud_bootstrap_and_engine_args():
    pool = cloud_pool()
    script = bootstrap(pool, CloudOptions.model_validate(pool.options))
    assert "'type=bind,src=/models/Qwen 14B,dst=/checkpoint,readonly'" in script
    assert "--publish 8000:8000 --publish 8001:8001" in script
    assert len(script.encode()) <= 16 * 1024
    args = Namespace(model="qwen", checkpoint="/checkpoint", gpus=2, capacity=4,
                     dtype="bfloat16", max_model_len=2048, backend="sglang")
    command = worker.engine_command(args, 8000, Path("/state/key"))
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert "--enable-metrics" in command
    assert str(Path("/state/sglang.yaml")) in command
    assert command[command.index("--tp-size") + 1] == "2"
    assert command[command.index("--log-level") + 1] == "warning"
    args.backend = "vllm"
    command = worker.engine_command(args, 8010, Path("/state/key"), "127.0.0.1")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--middleware") + 1] == "worker.GatewayAuth"
    assert command[command.index("--max-model-len") + 1] == "2048"


def test_wrong_gpu_class_fails_before_launch(monkeypatch):
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: json.dumps([["NVIDIA L40S", KEY]]))
    assert worker.allocated_gpus(1, "l40s") == [f"GPU-{KEY}"]
    with pytest.raises(RuntimeError, match="Requested 1 L4"):
        worker.allocated_gpus(1, "L4")


def test_loopback_sampler_binds_adjacent_to_ephemeral_engine():
    port, server = worker.claim_ports(0, worker.metrics_handler([], None), "127.0.0.1")
    with server:
        assert server.server_address == ("127.0.0.1", port + 1)
        assert port > 0


def test_empty_worker_key_rejected_before_using_a_gpu(monkeypatch, tmp_path, capsys):
    key = tmp_path / "key"
    key.write_text(" \n")
    monkeypatch.setattr(worker.sys, "argv", [
        "worker.py", "--backend", "vllm", "--model", "test", "--checkpoint", "local",
        "--gpus", "1", "--gpu-type", "A6000", "--capacity", "2", "--dtype", "float16",
        "--port", "0", "--key-file", str(key), "--loopback-only",
    ])
    monkeypatch.setattr(worker, "allocated_gpus", lambda *_: pytest.fail("GPU probe ran"))
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 2
    assert "--key-file must contain a nonempty bearer key" in capsys.readouterr().err
