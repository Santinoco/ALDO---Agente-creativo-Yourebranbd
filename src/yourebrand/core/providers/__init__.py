"""Proveedores de LLM: los únicos módulos que importan SDKs de proveedores.

Cada proveedor vive en su propio módulo, con su traducción de errores, porque
cambia cuando cambia su SDK. Para sumar uno: crear un módulo con una clase que
cumpla `LLMProvider`, agregarla en `build_provider` y sumar su nombre a
`llm_provider` en config.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr

from yourebrand.core.config import Settings, get_settings
from yourebrand.core.errors import ProviderConfigurationError
from yourebrand.core.providers.anthropic_provider import AnthropicProvider
from yourebrand.core.providers.base import ImageInput, LLMProvider, Message, ProviderResult, TurnResult
from yourebrand.core.providers.gemini_provider import GeminiProvider

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "ImageInput",
    "LLMProvider",
    "Message",
    "ProviderResult",
    "TurnResult",
    "ProviderName",
    "build_provider",
    "get_provider",
]

ProviderName = Literal["gemini", "anthropic"]


def _require_key(key: SecretStr | None, variable: str) -> str:
    # config.py ya lo valida; esto cubre una Settings armada a mano.
    if key is None:
        raise ProviderConfigurationError(f"falta {variable} para usar este proveedor")
    return key.get_secret_value()


def build_provider(name: ProviderName, settings: Settings) -> LLMProvider:
    """Construye el proveedor pedido con la clave de la configuración."""
    match name:
        case "anthropic":
            return AnthropicProvider(_require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"))
        case "gemini":
            return GeminiProvider(_require_key(settings.gemini_api_key, "GEMINI_API_KEY"))
    raise ProviderConfigurationError(f"proveedor desconocido: {name}")


@lru_cache
def get_provider(name: ProviderName | None = None) -> LLMProvider:
    """Devuelve un proveedor, creado una sola vez por proceso.

    Sin nombre, el de LLM_PROVIDER (el de la conversación). Con nombre, ese
    proveedor en particular: la edición de video pide "anthropic" siempre.
    """
    settings = get_settings()
    return build_provider(name or settings.llm_provider, settings)
