import httpx
from fastapi import Request

from tokenflow.registry import WorkerRegistry


def get_registry(request: Request) -> WorkerRegistry:
    return request.app.state.worker_registry


def get_worker_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.worker_client
