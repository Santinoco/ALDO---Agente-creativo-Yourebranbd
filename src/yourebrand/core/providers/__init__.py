"""Proveedores de LLM: los únicos módulos que importan SDKs de proveedores.

Cada proveedor vive en su propio módulo, con su traducción de errores, porque
cambia cuando cambia su SDK. Para sumar uno: crear un módulo con una clase que
cumpla `LLMProvider`, agregarla en `build_provider` y sumar su nombre a
`llm_provider` en config.
"""

from functools import lru_cache

from pydantic import SecretStr

from yourebrand.core.config import Settings, get_settings
from yourebrand.core.errors import ProviderConfigurationError
from yourebrand.core.providers.anthropic_provider import AnthropicProvider
from yourebrand.core.providers.base import LLMProvider, Message, ProviderResult, TurnResult
from yourebrand.core.providers.gemini_provider import GeminiProvider

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "LLMProvider",
    "Message",
    "ProviderResult",
    "TurnResult",
    "build_provider",
    "get_provider",
]


def _require_key(key: SecretStr | None, variable: str) -> str:
    # config.py ya lo valida; esto cubre una Settings armada a mano.
    if key is None:
        raise ProviderConfigurationError(f"falta {variable} para el proveedor elegido")
    return key.get_secret_value()


def build_provider(settings: Settings) -> LLMProvider:
    """Construye el proveedor elegido en la configuración (LLM_PROVIDER)."""
    match settings.llm_provider:
        case "anthropic":
            return AnthropicProvider(_require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"))
        case "gemini":
            return GeminiProvider(_require_key(settings.gemini_api_key, "GEMINI_API_KEY"))
    raise ProviderConfigurationError(f"proveedor desconocido: {settings.llm_provider}")


@lru_cache
def get_provider() -> LLMProvider:
    """Devuelve el proveedor configurado, creado una sola vez por proceso."""
    return build_provider(get_settings())
