from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TokenFlow"
    app_version: str = "0.1.0"
    worker_health_interval_seconds: float = Field(default=5.0, gt=0)
    worker_health_timeout_seconds: float = Field(default=2.0, gt=0)
    worker_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    worker_request_timeout_seconds: float = Field(default=120.0, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="TOKENFLOW_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
