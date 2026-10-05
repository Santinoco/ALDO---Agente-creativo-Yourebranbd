"""Proveedores de LLM: único módulo que importa SDKs de proveedores.

Para sumar un proveedor: crear una clase que cumpla `LLMProvider`,
agregarla en `build_provider` y sumar su nombre a `llm_provider` en config.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from anthropic import Anthropic

from yourebrand.core.config import Settings, get_settings


@dataclass(frozen=True)
class ProviderResult:
    """Respuesta normalizada: igual para todos los proveedores."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    truncated: bool  # la respuesta se cortó por llegar a max_tokens


class LLMProvider(Protocol):
    """Contrato que cumple todo proveedor de generación de texto."""

    name: str

    def generate(self, *, model: str, system: str, prompt: str, max_tokens: int) -> ProviderResult: ...


class AnthropicProvider:
    """Adaptador para la API de Claude."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = Anthropic(api_key=api_key)

    def generate(self, *, model: str, system: str, prompt: str, max_tokens: int) -> ProviderResult:
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return ProviderResult(
            text="".join(block.text for block in message.content if block.type == "text"),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
            truncated=message.stop_reason == "max_tokens",
        )


def build_provider(settings: Settings) -> LLMProvider:
    """Construye el proveedor elegido en la configuración (LLM_PROVIDER)."""
    match settings.llm_provider:
        case "anthropic":
            return AnthropicProvider(settings.anthropic_api_key.get_secret_value())


@lru_cache
def get_provider() -> LLMProvider:
    """Devuelve el proveedor configurado, creado una sola vez por proceso."""
    return build_provider(get_settings())
