"""
API server configuration — loaded from env vars or .env file.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    # Master API key — auto-generated on first run if not set
    master_api_key: str = ""
    # JWT secret for OAuth tokens
    jwt_secret: str = secrets.token_hex(32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    data_dir: str = "./memai_data"
    graph_backend: str = "auto"   # auto | kuzu | networkx
    vector_backend: str = "auto"  # auto | chromadb | dict
    embedding_model: str = "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------ #
    # Memory defaults
    # ------------------------------------------------------------------ #
    default_token_budget: int = 2000
    decay_lambda: float = 0.05

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    cors_origins: list[str] = ["*"]

    model_config = {
        "env_prefix": "MEMAI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, __context):
        # Auto-generate master API key if not set
        if not self.master_api_key:
            key_file = Path(self.data_dir) / ".master_key"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            if key_file.exists():
                object.__setattr__(self, "master_api_key", key_file.read_text().strip())
            else:
                generated = f"sk-memai-{secrets.token_urlsafe(32)}"
                key_file.write_text(generated)
                object.__setattr__(self, "master_api_key", generated)
                print(f"\n[memai] Generated master API key: {generated}")
                print(f"[memai] Saved to: {key_file}\n")


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
