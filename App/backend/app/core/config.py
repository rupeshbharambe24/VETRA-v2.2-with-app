
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_v1_prefix: str = "/api/v1"
    backend_cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    )
    db_path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "vetra_results.sqlite"))

    model_config = SettingsConfigDict(env_prefix="VETRA_", case_sensitive=False)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
