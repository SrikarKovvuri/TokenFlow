"""Run the managed controller against a stateful, in-memory EC2 service."""

import asyncio
import json
import os
import re
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import pytest

from tokenflow.platform.controller import Controller
from tokenflow.provisioner.base import PoolSpec
from tokenflow.provisioner.cloud import CloudProvisioner
from tokenflow.scheduler.selection import NoAvailableWorkerError, WorkersAtCapacityError
from tokenflow.worker_manager.health_monitor import WorkerHealthMonitor
from tokenflow.worker_manager.registry import WorkerRegistry
from tokenflow.worker_manager.telemetry import read_telemetry


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


@pytest.mark.skipif(not os.getenv("TOKENFLOW_TEST_VLLM_PORT"), reason="Opt in with a local vLLM engine port")
def test_local_gpu_with_emulated_ec2(tmp_path: Path, emulated_pool):
    """Moto owns the VM lifecycle. A separately started local GPU serves real HTTP inference."""
    from test_openai_client import loopback_server, openai
    from tokenflow.config import Settings
    from tokenflow.main import create_app

    port = int(os.environ["TOKENFLOW_TEST_VLLM_PORT"])
    assert 1024 <= port <= 65534
    origin = f"http://127.0.0.1:{port}"
    model = os.environ["TOKENFLOW_TEST_VLLM_MODEL"]
    assert re.fullmatch(r"[\w.-]+(?:/[\w.-]+)*", model), "Set a served alias, such as local-qwen"
    evidence_path = os.getenv("TOKENFLOW_TEST_VLLM_EVIDENCE")
    if evidence_path:
        destination = Path(evidence_path)
        assert destination.parent.is_dir() and not destination.exists(), "Choose a new file in an existing directory"

    session, ec2, original = emulated_pool
    pool = original.model_copy(update={"name": "local-gpu", "model": model, "port": port, "max_replicas": 1})
    namespace = "local-gpu-contract"

    class LocalEndpoint(CloudProvisioner):
        async def discover(self, spec):
            items = await super().discover(spec)
            return [replace(item, base_url=origin) if item.phase == "running" and item.base_url else item
                    for item in items]

    config = tmp_path / "deployment.json"
    config.write_text(json.dumps({"namespace": namespace, "state_path": "state.sqlite3", "pools": [pool.model_dump()],
                                  "interval_seconds": .25, "idle_seconds": 60}))
    student, operator = "local-only-student-key", "local-only-operator-key"
    gateway = create_app(Settings(_env_file=None, platform_config=config, api_keys=[student], admin_key=operator,
                                  worker_health_interval_seconds=.25, worker_health_timeout_seconds=5),
                         providers={"cloud": LocalEndpoint(namespace, session=session)})
    workload = {"model": model, "messages": [{"role": "user", "content": "List the integers from 1 to 40 in order."}],
                "temperature": 0, "max_tokens": 64}
    report = {"infrastructure": "Moto EC2 lifecycle; loopback vLLM and GPU sampler", "engine_origin": origin,
              "workload": workload, "worker_samples": [], "stream_chunks": []}

    def stamp():
        return datetime.now(UTC).isoformat()

    def usage_counts(response):
        if response.usage is None:
            return None
        return response.usage.model_dump(include={"prompt_tokens", "completion_tokens", "total_tokens"})

    async def scenario():
        async with (
            loopback_server(gateway) as url,
            httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {operator}"}, trust_env=False) as admin,
            httpx.AsyncClient(base_url=origin, trust_env=False, timeout=5) as engine,
            openai.AsyncOpenAI(base_url=f"{url}/v1", api_key=student, max_retries=0, timeout=45,
                               http_client=httpx.AsyncClient(trust_env=False)) as sdk,
        ):
            version = (await engine.get("/version")).raise_for_status().json()["version"]
            models = (await engine.get("/v1/models")).raise_for_status().json()["data"]
            identity = next(row for row in models if row["id"] == model)
            sampler = (await engine.get(f"http://127.0.0.1:{port + 1}/metrics")).raise_for_status()
            physical = read_telemetry(sampler.text)
            assert physical and physical.gpu_utilization is not None, "Adjacent sampler must report GPU utilization"
            report["engine"] = {"version": version, "model": identity["id"], "max_model_len": identity.get("max_model_len"),
                                "sampler_checked_at_utc": stamp(), "sampler_gpu_utilization": physical.gpu_utilization}
            report["gateway_models"] = [entry.id for entry in (await sdk.models.list()).data]
            assert model in report["gateway_models"]
            assert (await admin.get("/v1/pools")).raise_for_status().json()[0]["resources"] == []
            ready, done = asyncio.Event(), asyncio.Event()

            async def sample():
                while not done.is_set():
                    workers = (await admin.get("/v1/workers")).raise_for_status().json()
                    rows = [{k: w[k] for k in ("id", "enabled", "health_status", "active_requests", "telemetry")}
                            for w in workers]
                    report["worker_samples"].append({"received_at_utc": stamp(), "workers": rows})
                    for worker in rows:
                        gauges = worker["telemetry"] or {}
                        if (worker["health_status"] == "healthy" and gauges.get("gpu_utilization") is not None
                                and any(gauges.get(name) is not None for name in ("running", "waiting", "kv_cache_usage"))):
                            ready.set()
                    await asyncio.sleep(.1)

            try:
                async with asyncio.timeout(120), asyncio.TaskGroup() as tasks:
                    tasks.create_task(sample())
                    try:
                        with pytest.raises(openai.APIStatusError) as cold:
                            await sdk.chat.completions.create(**workload)
                        marker = cold.value.response.headers.get("x-tokenflow-admission")
                        assert cold.value.status_code == 503 and marker == "waiting-for-worker"
                        report["cold_admission"] = {"received_at_utc": stamp(), "status": cold.value.status_code, "marker": marker}
                        await asyncio.wait_for(ready.wait(), 35)
                        [allocation] = (await admin.get("/v1/pools")).raise_for_status().json()[0]["resources"]
                        assert allocation["phase"] == "running"
                        report["adopted_instance"] = {"checked_at_utc": stamp(), "id": allocation["external_id"],
                                                      "phase": allocation["phase"], "worker_id": allocation["worker_id"]}
                        sent = stamp()
                        reply = await sdk.chat.completions.create(**workload)
                        choice = reply.choices[0]
                        assert reply.model == model and choice.message.content and choice.finish_reason
                        report["json"] = {"sent_at_utc": sent, "received_at_utc": stamp(), "model": reply.model, "id": reply.id,
                                          "content": choice.message.content, "finish_reason": choice.finish_reason,
                                          "usage": usage_counts(reply)}
                        report["stream_sent_at_utc"] = stamp()
                        start = perf_counter()
                        async with await sdk.chat.completions.create(**workload, stream=True,
                                stream_options={"include_usage": True}) as stream:
                            async for chunk in stream:
                                assert chunk.model == model
                                report["stream_chunks"].append({
                                    "received_at_utc": stamp(), "elapsed_seconds": perf_counter() - start,
                                    "id": chunk.id, "model": chunk.model,
                                    "choices": [{"index": c.index, "role": c.delta.role,
                                                 "content": c.delta.content, "finish_reason": c.finish_reason}
                                                for c in chunk.choices], "usage": usage_counts(chunk),
                                })
                        report["stream_ended_at_utc"] = stamp()
                        choices = [c for chunk in report["stream_chunks"] for c in chunk["choices"]]
                        assert any(c["content"] for c in choices) and any(c["finish_reason"] for c in choices)
                    finally:
                        done.set()
            finally:
                (await admin.put(f"/v1/pools/{pool.name}/limits", json={"minimum": 0, "maximum": 0})).raise_for_status()
                async with asyncio.timeout(15):
                    while (await admin.get("/v1/pools")).raise_for_status().json()[0]["resources"]:
                        await asyncio.sleep(.1)
            assert not (await admin.get("/v1/workers")).raise_for_status().json()
            instances = [vm for res in ec2.describe_instances()["Reservations"] for vm in res["Instances"]]
            assert len(instances) == 1 and instances[0]["State"]["Name"] == "terminated"
            assert instances[0]["InstanceId"] == report["adopted_instance"]["id"]
            report["emulated_instance"] = {"id": instances[0]["InstanceId"], "type": instances[0]["InstanceType"],
                                           "final_state": instances[0]["State"]["Name"], "checked_at_utc": stamp()}

    asyncio.run(scenario())
    if evidence_path:
        with Path(evidence_path).open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
