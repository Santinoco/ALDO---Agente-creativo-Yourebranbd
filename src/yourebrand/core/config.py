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
    # Videos subidos y lo que se deriva de ellos. Local, en la máquina donde corre
    # la app; fuera de git. Ver core/storage.py.
    storage_dir: Path = PROJECT_ROOT / "storage"
    max_video_mb: int = Field(default=500, gt=0)
    # Transcripción local con faster-whisper (ver core/transcription.py).
    # "small" anda bien en español en CPU; "medium" es más preciso y más lento.
    whisper_model: str = "small"
    # Fotogramas que recibe el modelo del plan de edición: más es más caro.
    video_max_frames: int = Field(default=40, ge=1, le=100)

    # LLM: sin ellos no funciona ningún componente. Solo es obligatoria la
    # clave del proveedor elegido (ver _require_provider_key).
    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    model_generation: str
    model_fast: str
    # Edición de video: siempre con el proveedor de Claude, sin importar
    # LLM_PROVIDER. Vacío = edición deshabilitada. Ver core/llm.py (_route).
    model_video_edit: str | None = None
    # Topes de gasto de la edición de video (Opus es el modelo caro del sistema).
    # Salida: tope duro que aplica la API; incluye el razonamiento del modelo.
    video_edit_max_output_tokens: int = Field(default=16000, ge=1000, le=64000)
    # Entrada: estimada antes de llamar; si la supera, no se llama.
    video_edit_max_input_tokens: int = Field(default=60000, ge=1000)
    # Plazo propio: un plan con razonamiento tarda más que un turno de chat. Un
    # timeout no se reintenta en esta tarea (ver core/llm.py).
    video_edit_timeout_seconds: float = Field(default=600, gt=0)
    # Cuántas veces puede pedir el agente un plan en un mismo mensaje.
    video_edit_max_calls_per_turn: int = Field(default=2, ge=1, le=5)

    # Confiabilidad de las llamadas al modelo, igual para cualquier proveedor.
    llm_timeout_seconds: float = Field(default=120, gt=0)  # por intento
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # Opcionales hasta que se construya el componente que los usa.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    supabase_url: str | None = None
    supabase_key: SecretStr | None = None
    notion_api_key: SecretStr | None = None

    @field_validator("gemini_api_key", "anthropic_api_key", "model_video_edit", mode="before")
    @classmethod
    def _empty_is_missing(cls, value: object) -> object:
        """`VARIABLE=` sin valor, como queda al copiar `.env.example`, cuenta como que falta."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_provider_key(self) -> Self:
        """Falla al arrancar si falta una clave que hace falta, no en la primera llamada."""
        keys = {"gemini": self.gemini_api_key, "anthropic": self.anthropic_api_key}
        if keys[self.llm_provider] is None:
            raise ValueError(f"LLM_PROVIDER={self.llm_provider} requiere {self.llm_provider.upper()}_API_KEY")
        # La edición de video usa Claude aunque la conversación use otro proveedor.
        if self.model_video_edit is not None and self.anthropic_api_key is None:
            raise ValueError("MODEL_VIDEO_EDIT requiere ANTHROPIC_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración, leída una sola vez por proceso."""
    return Settings()
