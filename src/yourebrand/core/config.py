"""Configuración central: único punto donde se leen variables de entorno."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # LLM: sin ellos no funciona ningún componente. Solo es obligatoria la
    # clave del proveedor elegido (ver _require_provider_key).
    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
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

    @field_validator("gemini_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def _empty_key_is_missing(cls, value: object) -> object:
        """`CLAVE=` sin valor, como queda al copiar `.env.example`, cuenta como que falta."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_provider_key(self) -> Self:
        """Falla al arrancar si falta la clave del proveedor elegido, no en la primera llamada."""
        keys = {"gemini": self.gemini_api_key, "anthropic": self.anthropic_api_key}
        if keys[self.llm_provider] is None:
            raise ValueError(f"LLM_PROVIDER={self.llm_provider} requiere {self.llm_provider.upper()}_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración, leída una sola vez por proceso."""
    return Settings()
