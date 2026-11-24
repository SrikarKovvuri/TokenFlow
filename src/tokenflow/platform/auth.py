import secrets
from collections.abc import Callable
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

bearer = HTTPBearer(auto_error=False)


def require_key(keys: list[SecretStr]) -> Callable:
    """Keep student inference credentials separate from operator credentials."""
    async def check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
    ) -> None:
        if not keys:
            return
        supplied = credentials.credentials.encode() if credentials else b""
        if not any(secrets.compare_digest(supplied, key.get_secret_value().encode()) for key in keys):
            raise HTTPException(401, "A valid bearer key is required",
                                headers={"WWW-Authenticate": "Bearer"})

    return check
