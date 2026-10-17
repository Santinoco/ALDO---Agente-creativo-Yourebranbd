"""Configuración central: único punto donde se leen variables de entorno."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> core -> yourebrand -> src -> raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Un error de validación muestra los valores recibidos, claves incluidas.
        hide_input_in_errors=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = PROJECT_ROOT / "data"

    # LLM: obligatorios, sin ellos no funciona ningún componente.
    llm_provider: Literal["anthropic"] = "anthropic"
    anthropic_api_key: SecretStr
    model_generation: str
    model_fast: str

    # Confiabilidad de las llamadas al modelo, igual para cualquier proveedor.
    llm_timeout_seconds: float = Field(default=120, gt=0)  # por intento
    llm_max_retries: int = Field(default=2, ge=0, le=5)

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
