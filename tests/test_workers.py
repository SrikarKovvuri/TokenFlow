from uuid import uuid4

from fastapi.testclient import TestClient


def worker_payload(
    *,
    name: str = "worker-1",
    base_url: str = "http://localhost:8101",
) -> dict[str, object]:
    return {
        "name": name,
        "base_url": base_url,
        "enabled": True,
        "metadata": {"model": "simulated"},
    }


def test_registry_starts_empty(client: TestClient) -> None:
    response = client.get("/v1/workers")

    assert response.status_code == 200
    assert response.json() == []


def test_worker_crud_lifecycle(client: TestClient) -> None:
    create_response = client.post("/v1/workers", json=worker_payload())

    assert create_response.status_code == 201
    created = create_response.json()
    worker_id = created["id"]
    assert created["name"] == "worker-1"
    assert created["base_url"] == "http://localhost:8101/"
    assert created["health_status"] == "unknown"
    assert create_response.headers["Location"] == f"/v1/workers/{worker_id}"

    list_response = client.get("/v1/workers")
    assert list_response.status_code == 200
    assert [worker["id"] for worker in list_response.json()] == [worker_id]

    get_response = client.get(f"/v1/workers/{worker_id}")
    assert get_response.status_code == 200
    assert get_response.json() == created

    update_response = client.patch(
        f"/v1/workers/{worker_id}",
        json={"name": "worker-primary", "enabled": False},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["name"] == "worker-primary"
    assert updated["enabled"] is False
    assert updated["registered_at"] == created["registered_at"]

    delete_response = client.delete(f"/v1/workers/{worker_id}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""

    assert client.get(f"/v1/workers/{worker_id}").status_code == 404


def test_duplicate_worker_url_is_rejected(client: TestClient) -> None:
    first_response = client.post("/v1/workers", json=worker_payload())
    duplicate_response = client.post(
        "/v1/workers",
        json=worker_payload(name="worker-2"),
    )

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409


def test_invalid_worker_url_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/workers",
        json=worker_payload(base_url="not-a-url"),
    )

    assert response.status_code == 422


def test_empty_or_null_update_is_rejected(client: TestClient) -> None:
    create_response = client.post("/v1/workers", json=worker_payload())
    worker_id = create_response.json()["id"]

    assert client.patch(f"/v1/workers/{worker_id}", json={}).status_code == 422
    assert (
        client.patch(
            f"/v1/workers/{worker_id}",
            json={"name": None},
        ).status_code
        == 422
    )


def test_missing_worker_returns_not_found(client: TestClient) -> None:
    worker_id = uuid4()

    assert client.get(f"/v1/workers/{worker_id}").status_code == 404
    assert (
        client.patch(
            f"/v1/workers/{worker_id}",
            json={"enabled": False},
        ).status_code
        == 404
    )
    assert client.delete(f"/v1/workers/{worker_id}").status_code == 404
