"""Exercise the regular EC2 provider against botocore's real request and pagination models."""

import asyncio
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tokenflow.provisioner.base import PoolSpec, Resource
from tokenflow.provisioner.cloud import CloudOptions, CloudProvisioner, bootstrap


boto3 = pytest.importorskip("boto3")
from botocore.exceptions import ClientError
from botocore.httpsession import URLLib3Session
from botocore.stub import ANY, Stubber


REGION = "us-east-1"
KEY = "1" * 32
INSTANCE_ID = "i-0123456789abcdef0"
FILTERS = [
    {"Name": "tag:TokenFlowNamespace", "Values": ["contract-test"]},
    {"Name": "tag:TokenFlowPool", "Values": ["qwen-t4"]},
]
KEY_FILTERS = [*FILTERS, {"Name": "tag:TokenFlowKey", "Values": [KEY]}]


@pytest.fixture
def fleet(monkeypatch):
    def reject_http(*args, **kwargs):
        pytest.fail("The EC2 contract tests must stay offline")

    monkeypatch.setattr(URLLib3Session, "send", reject_http)
    session = boto3.Session(
        aws_access_key_id="offline", aws_secret_access_key="offline", region_name=REGION,
    )
    pool = PoolSpec(
        name="qwen-t4", model="qwen", checkpoint="/models/qwen", provider="cloud", gpu_type="T4",
        image="example/vllm:test", max_replicas=1,
        options={
            "region": REGION, "instance_type": "g4dn.xlarge", "ami_id": "ami-123",
            "subnet_id": "subnet-123", "security_group_ids": ["sg-abc"],
            "iam_instance_profile": "TokenFlowWorker", "image_pull_policy": "never",
        },
    )
    provider = CloudProvisioner("contract-test", session=session)
    with Stubber(provider._client(REGION)) as stub:
        yield provider, pool, stub
        stub.assert_no_pending_responses()


def described_instance(instance_id=INSTANCE_ID, state="running"):
    return {
        "InstanceId": instance_id,
        "State": {"Name": state},
        "PrivateIpAddress": "10.0.1.37",
        "Tags": [
            {"Key": "TokenFlowNamespace", "Value": "contract-test"},
            {"Key": "TokenFlowPool", "Value": "qwen-t4"},
            {"Key": "TokenFlowKey", "Value": KEY},
        ],
    }


def test_run_instances_retry_reuses_ownership_and_idempotency_token(fleet):
    provider, pool, stub = fleet
    launch_args = {
        "MinCount": 1, "MaxCount": 1, "ImageId": "ami-123", "InstanceType": "g4dn.xlarge",
        "InstanceInitiatedShutdownBehavior": "terminate",
        "NetworkInterfaces": [{
            "DeviceIndex": 0, "SubnetId": "subnet-123", "Groups": ["sg-abc"],
            "AssociatePublicIpAddress": False, "DeleteOnTermination": True,
        }],
        "IamInstanceProfile": {"Name": "TokenFlowWorker"},
        "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [*described_instance()["Tags"], {"Key": "Name", "Value": "tokenflow-qwen-t4"}],
        }],
        "ClientToken": hashlib.sha256(f"contract-test/qwen-t4/{KEY}".encode()).hexdigest(),
        "UserData": ANY,
    }
    # An EC2 InternalError leaves acceptance ambiguous. Replaying the exact
    # on-demand launch key lets EC2 return the same instance if it was accepted.
    stub.add_client_error("run_instances", "InternalError", http_status_code=500, expected_params=launch_args)
    stub.add_response("run_instances", {"Instances": [{"InstanceId": INSTANCE_ID}]}, launch_args)
    with pytest.raises(ClientError, match="InternalError"):
        asyncio.run(provider.launch(pool, KEY))
    assert asyncio.run(provider.launch(pool, KEY)) == Resource(key=KEY, external_id=INSTANCE_ID, phase="pending")


def test_recover_through_next_token_and_repeat_termination(fleet):
    provider, pool, stub = fleet
    page = {"Reservations": [{"Instances": [described_instance()]}]}
    stub.add_response("describe_instances", {"Reservations": [], "NextToken": "page-two"}, {"Filters": FILTERS})
    stub.add_response("describe_instances", page, {"Filters": FILTERS, "NextToken": "page-two"})
    recovered, = asyncio.run(provider.discover(pool))
    assert recovered == Resource(key=KEY, external_id=INSTANCE_ID, phase="running", base_url="http://10.0.1.37:8000")

    stub.add_response("describe_instances", page, {"Filters": KEY_FILTERS})
    stub.add_response("terminate_instances", {"TerminatingInstances": [{"InstanceId": INSTANCE_ID}]},
                      {"InstanceIds": [INSTANCE_ID]})
    asyncio.run(provider.terminate(pool, recovered))

    terminal = {"Reservations": [{"Instances": [described_instance(state="shutting-down")]}]}
    stub.add_response("describe_instances", terminal, {"Filters": KEY_FILTERS})
    asyncio.run(provider.terminate(pool, recovered))


def test_partial_inventory_error_does_not_terminate(fleet):
    provider, pool, stub = fleet
    page = {"Reservations": [{"Instances": [described_instance()]}], "NextToken": "next"}
    stub.add_response("describe_instances", page, {"Filters": KEY_FILTERS})
    stub.add_client_error("describe_instances", "UnauthorizedOperation", http_status_code=403,
                          expected_params={"Filters": KEY_FILTERS, "NextToken": "next"})
    resource = Resource(key=KEY, external_id=INSTANCE_ID, phase="running")
    with pytest.raises(ClientError, match="UnauthorizedOperation"):
        asyncio.run(provider.terminate(pool, resource))


@pytest.mark.parametrize("checkpoint", ["Qwen/Qwen2.5-14B", "/models-old/qwen", "/models/../other"])
def test_checkpoint_must_be_on_the_requested_mount(fleet, checkpoint):
    provider, pool, _ = fleet
    with pytest.raises(ValueError, match="checkpoint_mount"):
        provider.validate_pool(pool.model_copy(update={
            "checkpoint": checkpoint, "options": {**pool.options, "checkpoint_mount": "/models"},
        }))


@pytest.mark.parametrize("failure", ["", "mountpoint", "docker"])
def test_worker_bootstrap_mount_and_failure_shutdown(fleet, tmp_path, failure):
    _, pool, _ = fleet
    if os.name == "nt":
        bash = Path(os.environ.get("ProgramFiles", "C:/Program Files"), "Git/bin/bash.exe")
        if not bash.is_file():
            pytest.skip("Git Bash is required on Windows to execute the cloud-init script")
    else:
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("Bash is required to execute the cloud-init script")
    opts = CloudOptions.model_validate({**pool.options, "checkpoint_mount": "/models with spaces"})
    pool = pool.model_copy(update={"checkpoint": "/models with spaces/qwen"})
    data = bootstrap(pool, opts).replace("/var/lib/tokenflow", "worker-state").replace(
        "/etc/systemd/system", "unit-config",
    )
    # All systemd, mount, Docker, and shutdown operations are shell-local fakes.
    commands = """
systemd-escape() { test "$3" = '/models with spaces' && echo models.mount; }
systemctl() { echo "systemctl $*" >> calls; }
mountpoint() {
    test "$3" = '/models with spaces' || return 2
    echo mountpoint >> calls
    test "$FAILURE" != mountpoint
}
docker() {
    echo docker >> calls
    test "$FAILURE" != docker
}
"""
    (tmp_path / "bootstrap.sh").write_text(commands + data, newline="\n")
    result = subprocess.run(
        [str(bash), "bootstrap.sh"], cwd=tmp_path,
        env={**os.environ, "FAILURE": failure}, capture_output=True, text=True, timeout=15,
    )
    assert (tmp_path / "calls").is_file(), result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    expected = ["systemctl daemon-reload", "systemctl start -- models.mount", "mountpoint"]
    if failure != "mountpoint":
        expected.extend(["systemctl enable --now docker", "docker"])
    if failure:
        expected.append("systemctl --no-block poweroff")
    assert calls == expected
    dropin = tmp_path / "unit-config/docker.service.d/tokenflow-checkpoint.conf"
    assert dropin.read_text().splitlines() == ["[Unit]", "Requires=models.mount", "After=models.mount"]
    assert result.returncode == int(bool(failure)), result.stderr
