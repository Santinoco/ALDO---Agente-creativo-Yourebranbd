"""Errores de la capa de modelos, iguales para cualquier proveedor.

Cada proveedor traduce las excepciones de su SDK a estos tipos. El resto del
sistema solo conoce estos: nunca captura un error de un SDK.
"""


class LLMError(Exception):
    """Error base de la capa de modelos."""


class ProviderConfigurationError(LLMError):
    """Clave inválida, sin permiso o modelo inexistente. No se reintenta."""


class InvalidRequestError(LLMError):
    """El proveedor rechazó el pedido por estar mal armado. No se reintenta."""


class TransientProviderError(LLMError):
    """Falla temporal del proveedor. Se puede reintentar."""


class RateLimitError(TransientProviderError):
    """El proveedor rechazó la llamada por límite de uso."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        # Espera que pide el proveedor antes de reintentar, si la informa.
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeoutError(TransientProviderError):
    """La llamada superó el tiempo máximo permitido."""


class LLMOutputError(LLMError):
    """La llamada terminó pero su salida no se puede usar."""


class TruncatedOutputError(LLMOutputError):
    """La respuesta se cortó por llegar a max_tokens."""


class RefusedOutputError(LLMOutputError):
    """El modelo se negó a responder."""


class InvalidOutputError(LLMOutputError):
    """La respuesta no cumple el schema pedido."""
