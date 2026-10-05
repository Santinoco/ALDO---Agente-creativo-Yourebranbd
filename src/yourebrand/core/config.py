"""Configuración central: único punto donde se leen variables de entorno."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> core -> yourebrand -> src -> raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = PROJECT_ROOT / "data"

    # LLM: obligatorios, sin ellos no funciona ningún componente.
    llm_provider: Literal["anthropic"] = "anthropic"
    anthropic_api_key: SecretStr
    model_generation: str
    model_fast: str

    # Opcionales hasta que se construya el componente que los usa.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    supabase_url: str | None = None
    supabase_key: SecretStr | None = None
    notion_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración, leída una sola vez por proceso."""
    return Settings()
