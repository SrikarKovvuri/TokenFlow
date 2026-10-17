from fastapi import Request

from tokenflow.registry import WorkerRegistry


def get_registry(request: Request) -> WorkerRegistry:
    return request.app.state.worker_registry
