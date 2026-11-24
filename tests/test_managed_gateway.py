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
    def __init__(self):
        self.resources = {}
        self.launches = 0

    async def discover(self, pool):
        return list(self.resources.values())

    async def launch(self, pool, key):
        self.launches += 1
        self.resources[key] = Resource(key, "job-123", "running", "http://compute:8100")
        return self.resources[key]

    async def terminate(self, pool, resource):
        self.resources[resource.key] = Resource(resource.key, resource.external_id, "terminal")


class Events(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b"data: [DONE]\n\n"


def test_managed_cold_start_stream_restart_and_drain(tmp_path):
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

    async def engine(req):
        assert req.headers["Authorization"] == "Bearer " + worker_token(SECRET, next(iter(provider.resources)))
        seen.add((req.url.port, req.url.path))
        if req.url.path == "/health":
            return httpx.Response(200)
        if req.url.path == "/metrics":
            metrics = ("vllm:num_requests_running 0" if req.url.port == 8100
                       else 'DCGM_FI_DEV_GPU_UTIL{UUID="GPU-one"} 27')
            return httpx.Response(200, text=metrics)
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(200, stream=Events(), headers={"Content-Type": "text/event-stream"})

    async def wait_workers(client, count):
        async with asyncio.timeout(5):
            while True:
                response = await client.get("/v1/workers", headers=admin)
                assert response.status_code == 200
                workers = response.json()
                if len(workers) == count and all(w["health_status"] == "healthy" for w in workers):
                    return workers
                await asyncio.sleep(.01)

    async def scenario():
        for restart in (False, True):
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
                workers = await wait_workers(client, 1)
                target = "/v1/workers/" + workers[0]["id"]
                assert (await client.delete(target, headers=admin)).status_code == 409
                assert (await client.patch(target, headers=admin, json={"enabled": False})).status_code == 409
                reply = await client.post("/v1/chat/completions", headers=student, json=payload)
                assert reply.status_code == 200 and reply.text.endswith("data: [DONE]\n\n")
                assert provider.launches == 1
                if restart:
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
