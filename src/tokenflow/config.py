from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TokenFlow"
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_prefix="TOKENFLOW_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
