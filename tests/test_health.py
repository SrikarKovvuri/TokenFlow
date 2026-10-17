from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "TokenFlow",
        "version": "0.1.0",
    }
    assert response.headers["X-Request-ID"]


def test_request_id_is_preserved(client: TestClient) -> None:
    response = client.get(
        "/healthz",
        headers={"X-Request-ID": "test-request"},
    )

    assert response.headers["X-Request-ID"] == "test-request"
