"""Run the managed controller against a stateful, in-memory EC2 service."""

import asyncio
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from tokenflow.platform.controller import Controller
from tokenflow.provisioner.base import PoolSpec
from tokenflow.provisioner.cloud import CloudProvisioner
from tokenflow.scheduler.selection import NoAvailableWorkerError, WorkersAtCapacityError
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.registry import WorkerRegistry


moto = pytest.importorskip("moto")
import boto3
from botocore.httpsession import URLLib3Session


@pytest.fixture
def emulated_pool(monkeypatch):
    # A missing Moto handler must never fall through to a live endpoint.
    def reject_network(*args, **kwargs):
        pytest.fail("Unexpected network access in the EC2 emulator test")

    monkeypatch.setattr(URLLib3Session, "send", reject_network)
    with moto.mock_aws():
        session = boto3.Session(
            aws_access_key_id="offline", aws_secret_access_key="offline", region_name="us-east-1",
        )
        ec2 = session.client("ec2")
        vpc = ec2.create_vpc(CidrBlock="10.70.0.0/16")["Vpc"]["VpcId"]
        subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.70.1.0/24")["Subnet"]["SubnetId"]
        group = ec2.create_security_group(VpcId=vpc, GroupName="workers", Description="Private workers")["GroupId"]
        image = ec2.register_image(Name="test-gpu-image", Architecture="x86_64")["ImageId"]
        pool = PoolSpec(
            name="qwen-t4", model="qwen", provider="cloud", backend="vllm", gpu_type="T4",
            checkpoint="/models/qwen", image="vllm:test", min_replicas=0, max_replicas=2, capacity=1,
            options=dict(region="us-east-1", instance_type="g4dn.xlarge", ami_id=image,
                         subnet_id=subnet, security_group_ids=[group]),
        )

        yield session, ec2, pool


def test_two_replicas_scale_from_zero_and_drain_in_their_own_namespace(tmp_path: Path, emulated_pool):
    session, ec2, pool = emulated_pool
    registry = WorkerRegistry()
    provider = CloudProvisioner("autoscaling-test", session=session)

    def engine(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": pool.model}]})
        assert request.url.path == "/metrics"
        metric = "vllm:num_requests_running 0" if request.url.port == pool.port else "DCGM_FI_DEV_GPU_UTIL 25"
        return httpx.Response(200, text=metric)

    def instance_states():
        return {vm["InstanceId"]: vm["State"]["Name"]
                for reservation in ec2.describe_instances()["Reservations"] for vm in reservation["Instances"]}

    async def exercise(controller, client):
        health = WorkerHealthMonitor(registry, client, interval_seconds=1, timeout_seconds=2,
                                     managed_worker=controller.owns_worker)
        outsider = await CloudProvisioner("unrelated", session=session).launch(pool, "a" * 32)
        await controller.reconcile()
        assert instance_states() == {outsider.external_id: "running"}

        with pytest.raises(NoAvailableWorkerError):
            await registry.assign(pool.model)
        await controller.reconcile()  # submit
        await controller.reconcile()  # discover its private IP
        await health.check_once()
        first = await registry.assign(pool.model)
        with pytest.raises(WorkersAtCapacityError):
            await registry.assign(pool.model)
        await controller.reconcile()
        await controller.reconcile()
        await health.check_once()
        second = await registry.assign(pool.model)

        ids = {r.base_url: r.external_id for r in await provider.discover(pool)}
        first_url, second_url = (str(w.base_url).rstrip("/") for w in (first, second))
        assert first_url != second_url and set(ids) == {first_url, second_url}
        assert all(w.telemetry.gpu_utilization == 0.25 for w in await registry.list())

        await registry.release(second.id)
        controller.set_limits(pool.name, 0, 0)
        await controller.reconcile()
        assert instance_states() == {
            ids[first_url]: "running", ids[second_url]: "terminated", outsider.external_id: "running",
        }
        await registry.release(first.id)
        await controller.reconcile()
        await controller.reconcile()
        assert not await registry.list()
        assert not controller.status()[0]["resources"]
        assert instance_states() == {
            ids[first_url]: "terminated", ids[second_url]: "terminated", outsider.external_id: "running",
        }

    async def scenario():
        with closing(Controller([pool], {"cloud": provider}, registry, tmp_path / "control.sqlite3",
                                interval_seconds=1, idle_seconds=60, startup_timeout_seconds=30)) as controller:
            async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
                await exercise(controller, client)

    asyncio.run(scenario())
