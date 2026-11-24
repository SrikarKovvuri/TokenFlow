from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TokenFlow"
    app_version: str = "0.1.0"
    worker_health_interval_seconds: float = Field(default=5.0, gt=0)
    worker_health_timeout_seconds: float = Field(default=2.0, gt=0)
    worker_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    worker_request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_inference_body_bytes: int = Field(default=4 * 1024 * 1024, gt=0)
    platform_config: Path | None = None
    admin_key: SecretStr | None = None
    api_keys: list[SecretStr] = Field(default_factory=list)
    worker_secret: SecretStr | None = None

    @model_validator(mode="after")
    def check_access_keys(self) -> Self:
        if self.platform_config or self.api_keys or self.admin_key:
            if not self.api_keys or self.admin_key is None:
                raise ValueError("Set TOKENFLOW_API_KEYS and TOKENFLOW_ADMIN_KEY together")
            keys = [key.get_secret_value() for key in [self.admin_key, *self.api_keys]]
            if self.worker_secret:
                keys.append(self.worker_secret.get_secret_value())
            if any(len(key) < 16 or not key.isascii() or not key.isprintable()
                   or any(char.isspace() for char in key) for key in keys):
                raise ValueError("Use access keys of at least 16 printable non-space ASCII characters")
            if len(set(keys)) != len(keys):
                raise ValueError("Each access key must be distinct")
        return self

    model_config = SettingsConfigDict(
        env_prefix="TOKENFLOW_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
