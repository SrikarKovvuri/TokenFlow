"""Exercise the real gateway and controller with a deterministic provider and engine."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from tokenflow.config import Settings
from tokenflow.main import create_app
from tokenflow.platform.config import PlatformConfig
from tokenflow.platform.lease import controller_lease
from tokenflow.provisioner.base import Resource
from tokenflow.provisioner.slurm import worker_token

STUDENT = "student-test-key-123"
ADMIN = "operator-test-key-123"
SECRET = "worker-test-secret-123"
DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


class AllocationProvider:
    def __init__(self, url="http://compute:8100"):
        self.resources = {}
        self.launches = 0
        self.url = url
        self.offline = False

    async def discover(self, pool):
        if self.offline:
            raise ConnectionError("provider inventory unavailable")
        return list(self.resources.values())

    async def launch(self, pool, key):
        self.launches += 1
        self.resources[key] = Resource(key, "job-123", "running", self.url)
        return self.resources[key]

    async def terminate(self, pool, resource):
        self.resources[resource.key] = Resource(resource.key, resource.external_id, "terminal")


class Events(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b"data: [DONE]\n\n"


async def wait_workers(client, count):
    async with asyncio.timeout(5):
        while True:
            response = await client.get("/v1/workers", headers={"Authorization": f"Bearer {ADMIN}"})
            assert response.status_code == 200
            workers = response.json()
            if len(workers) == count and all(w["enabled"] and w["health_status"] == "healthy" for w in workers):
                return workers
            await asyncio.sleep(.01)


@pytest.mark.parametrize("reuse_app", [False, True])
def test_managed_cold_start_stream_restart_and_drain(tmp_path, reuse_app):
    config = json.loads((DEPLOY / "insomnia.example.json").read_text())
    config.update(state_path="state.sqlite3", interval_seconds=.025, idle_seconds=300)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(config))
    settings = Settings(platform_config=path, admin_key=ADMIN, api_keys=[STUDENT],
                        worker_secret=SECRET, worker_health_interval_seconds=.025)
    provider = AllocationProvider()
    student = {"Authorization": f"Bearer {STUDENT}"}
    admin = {"Authorization": f"Bearer {ADMIN}"}
    payload = {"model": "qwen-14b", "messages": [{"role": "user", "content": "Hi"}],
               "stream": True}
    seen = set()
    allow_health = asyncio.Event()

    async def engine(req):
        assert req.headers["Authorization"] == "Bearer " + worker_token(SECRET, next(iter(provider.resources)))
        seen.add((req.url.port, req.url.path))
        if req.url.path == "/health":
            await allow_health.wait()
            return httpx.Response(200)
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen-14b"}]})
        if req.url.path == "/metrics":
            metrics = ("vllm:num_requests_running 0" if req.url.port == 8100
                       else 'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-one"} 27')
            return httpx.Response(200, text=metrics)
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(200, stream=Events(), headers={"Content-Type": "text/event-stream"})

    async def scenario():
        app = None
        worker_path = None
        allow_health.set()
        for restart in (False, True):
            provider.offline = restart
            if app is None or not reuse_app:
                app = create_app(settings, providers={"slurm": provider},
                                 worker_transport=httpx.MockTransport(engine))
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                assert (await client.get("/v1/models")).status_code == 401
                assert (await client.get("/v1/pools", headers=student)).status_code == 401
                assert (await client.get("/v1/workers", headers=student)).status_code == 401
                catalog = await client.get("/v1/models", headers=student)
                assert [m["id"] for m in catalog.json()["data"]] == ["qwen-14b"]
                if not restart:
                    cold = await client.post("/v1/chat/completions", headers=student, json=payload)
                    assert cold.status_code == 503
                else:
                    async with asyncio.timeout(5):
                        while not (await client.get("/v1/pools", headers=admin)).json()[0]["last_error"]:
                            await asyncio.sleep(.01)
                        assert (await client.get("/v1/workers", headers=admin)).json() == []
                        provider.offline = False
                        while not (await client.get("/v1/workers", headers=admin)).json():
                            await asyncio.sleep(.01)
                    adopted = (await client.get(worker_path, headers=admin)).raise_for_status().json()
                    assert adopted["health_status"] == "unknown"
                    assert adopted["telemetry"] is None and adopted["active_requests"] == 0
                    assert (await client.post("/v1/chat/completions", headers=student, json=payload)).status_code == 503
                    allow_health.set()
                workers = await wait_workers(client, 1)
                target = "/v1/workers/" + workers[0]["id"]
                assert (await client.delete(target, headers=admin)).status_code == 409
                assert (await client.patch(target, headers=admin, json={"enabled": False})).status_code == 409
                reply = await client.post("/v1/chat/completions", headers=student, json=payload)
                assert reply.status_code == 200 and reply.text.endswith("data: [DONE]\n\n")
                assert provider.launches == 1
                if not restart:
                    worker_path = target
                    # Simulate the reservation a previous process lost on exit.
                    await app.state.worker_registry.assign("qwen-14b")
                    prior = (await client.get(worker_path, headers=admin)).json()
                    assert prior["telemetry"] is not None and prior["active_requests"] == 1
                    allow_health.clear()
                else:
                    path = "/v1/pools/qwen-a6000/limits"
                    limits = {"minimum": 0, "maximum": 0}
                    assert (await client.put(path, headers=student, json=limits)).status_code == 401
                    assert (await client.put(path, headers=admin, json=limits)).status_code == 200
                    await wait_workers(client, 0)
                    pools = await client.get("/v1/pools", headers=admin)
                    assert pools.json()[0]["resources"] == []
                    assert all(r.phase == "terminal" for r in provider.resources.values())
        assert {(8100, "/metrics"), (8101, "/metrics")} <= seen

    asyncio.run(scenario())


@pytest.mark.parametrize("outage", ["cloud", "slurm"])
def test_concurrent_models_stay_in_their_pools_during_one_provider_outage(tmp_path, outage):
    config = json.loads((DEPLOY / "insomnia.example.json").read_text())
    cloud = json.loads((DEPLOY / "aws.example.json").read_text())["pools"][0]
    cloud.update(model="cloud-chat", backend="sglang", image="example/sglang:test")
    config["pools"].append(cloud)
    config.update(state_path="catalog.sqlite3", interval_seconds=.015)
    for pool in config["pools"]:
        pool.update(min_replicas=1, capacity=1)
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(config))
    settings = Settings(platform_config=path, admin_key=ADMIN, api_keys=[STUDENT], worker_secret=SECRET,
                        worker_health_interval_seconds=.015)
    models = {pool["provider"]: pool["model"] for pool in config["pools"]}
    providers = {kind: AllocationProvider(f"http://{kind}:8100") for kind in models}

    async def scenario():
        started = {kind: asyncio.Event() for kind in models}
        release = asyncio.Event()

        async def engine(request):
            kind = request.url.host
            model = models[kind]
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": model}]})
            if request.url.path == "/metrics":
                if request.url.port == 8101:
                    return httpx.Response(200, text="DCGM_FI_DEV_GPU_UTIL 12")
                gauge = "sglang:num_running_reqs" if kind == "cloud" else "vllm:num_requests_running"
                return httpx.Response(200, text=f"{gauge} 0")
            assert request.url.path == "/v1/chat/completions" and json.loads(request.content)["model"] == model
            if kind == "slurm":
                key = next(iter(providers[kind].resources))
                assert request.headers["Authorization"] == "Bearer " + worker_token(SECRET, key)
            else:
                assert "Authorization" not in request.headers
            started[kind].set()
            await release.wait()
            return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": kind}}]})

        app = create_app(settings, providers=providers, worker_transport=httpx.MockTransport(engine))
        async with asyncio.timeout(5), app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway",
            headers={"Authorization": f"Bearer {STUDENT}"},
        ) as client:
            async def chat(kind):
                return await client.post("/v1/chat/completions", json={
                    "model": models[kind], "messages": [{"role": "user", "content": "hi"}],
                })

            initial_workers = await wait_workers(client, 2)
            async with asyncio.TaskGroup() as requests:
                pending = {kind: requests.create_task(chat(kind)) for kind in models}
                await asyncio.gather(*(event.wait() for event in started.values()))
                assert [(await chat(kind)).status_code for kind in models] == [429, 429]
                providers[outage].offline = True
                while True:
                    pools = await client.get("/v1/pools", headers={"Authorization": f"Bearer {ADMIN}"})
                    if any(p["provider"] == outage and p["last_error"] for p in pools.json()):
                        break
                    await asyncio.sleep(.01)
                assert (await chat(outage)).status_code == 503
                release.set()
            for kind, task in pending.items():
                assert task.result().json()["choices"][0]["message"]["content"] == kind
            survivor = next(kind for kind in models if kind != outage)
            assert (await chat(survivor)).status_code == 200
            catalog = (await client.get("/v1/models")).raise_for_status().json()
            assert {row["id"] for row in catalog["data"]} == set(models.values())
            providers[outage].offline = False
            recovered = await wait_workers(client, 2)
            assert {w["id"] for w in recovered} == {w["id"] for w in initial_workers}
            assert (await chat(outage)).status_code == 200
            assert all(provider.launches == 1 for provider in providers.values())

    asyncio.run(scenario())


def test_managed_mode_requires_separate_keys(tmp_path):
    with pytest.raises(ValidationError):
        Settings(platform_config=tmp_path / "config")
    with pytest.raises(ValidationError):
        Settings(platform_config=tmp_path / "config", api_keys=[ADMIN], admin_key=ADMIN)


def test_only_one_controller_owns_a_database(tmp_path):
    path = tmp_path / "gateway.db"
    with controller_lease(path, "original"):
        with pytest.raises(RuntimeError, match="controller lock"), controller_lease(path, "original"):
            pytest.fail("Two gateways acquired one database")
    with pytest.raises(ValueError, match="namespace"), controller_lease(path, "other"):
        pytest.fail("Database was reassigned")


@pytest.mark.parametrize("name", ["insomnia", "aws"])
def test_deployment_examples_validate_offline(name):
    config = PlatformConfig.load(DEPLOY / f"{name}.example.json")
    assert len(config.pools) == 1
    assert config.pools[0].min_replicas == 0
