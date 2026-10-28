"""Adaptador para la API de Gemini: único módulo que importa el SDK de Google."""

import re
from typing import Any

import httpx
from google.genai import Client as GeminiClient
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from yourebrand.core.errors import (
    InvalidRequestError,
    LLMError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)
from yourebrand.core.providers.base import ProviderResult
from yourebrand.core.reliability import call_with_deadline


def _transport_errors(name: str) -> tuple[type[Exception], ...]:
    """Clase `name` de httpx y de httpx2, el fork que el SDK de Google usa si está instalado.

    Son clases distintas aunque se llamen igual, así que hay que capturar las dos.
    """
    classes: list[type[Exception]] = [getattr(httpx, name)]
    try:
        import httpx2

        classes.append(getattr(httpx2, name))
    except ImportError:
        pass
    return tuple(classes)


# El SDK de Google no envuelve las fallas de red en su APIError: llegan como
# excepciones de la librería HTTP. TimeoutException hereda de TransportError,
# así que se pregunta primero.
_GEMINI_TIMEOUT_ERRORS = _transport_errors("TimeoutException")
_GEMINI_TRANSPORT_ERRORS = _transport_errors("TransportError")

# Plazo mínimo que acepta la API de Gemini (lo informa al rechazar uno menor).
GEMINI_MIN_TIMEOUT_SECONDS = 10

# Motivos de corte que significan que el modelo se negó o lo frenó un filtro.
_GEMINI_REFUSAL_REASONS = {
    genai_types.FinishReason.SAFETY,
    genai_types.FinishReason.RECITATION,
    genai_types.FinishReason.BLOCKLIST,
    genai_types.FinishReason.PROHIBITED_CONTENT,
    genai_types.FinishReason.SPII,
}


def _gemini_error_details(exc: genai_errors.APIError) -> list[dict[str, Any]]:
    """Lista `error.details` de la respuesta de Google, o vacía si no viene."""
    body = exc.details if isinstance(exc.details, dict) else {}
    details = body.get("error", {}).get("details", [])
    return [item for item in details if isinstance(item, dict)]


def _gemini_retry_after_seconds(exc: genai_errors.APIError) -> float | None:
    """Espera pedida por Google ante un 429. La informa en el cuerpo (RetryInfo), no en un header."""
    for item in _gemini_error_details(exc):
        delay = item.get("retryDelay")
        if isinstance(delay, str) and (match := re.fullmatch(r"(\d+(?:\.\d+)?)s", delay)):
            return float(match.group(1))
    return None


def _translate_gemini_error(exc: genai_errors.APIError) -> LLMError:
    """Convierte una excepción del SDK de Google en un error de `core/errors.py`."""
    # Igual que con Anthropic: solo tipo y código. str(exc) trae el cuerpo
    # completo de la respuesta y no se copia al mensaje.
    status = exc.code
    detail = f"{type(exc).__name__}, HTTP {status}"
    if status == 429:
        return RateLimitError(
            f"gemini: límite de uso ({detail})", retry_after_seconds=_gemini_retry_after_seconds(exc)
        )
    # Google responde 400 (no 401) ante una clave inválida; se reconoce por el motivo.
    invalid_key = any(item.get("reason") == "API_KEY_INVALID" for item in _gemini_error_details(exc))
    if status in (401, 403, 404) or invalid_key:
        return ProviderConfigurationError(f"gemini: clave, permiso o modelo inválido ({detail})")
    # El SDK manda el timeout como plazo al servidor, y Google corta con 504
    # (DEADLINE_EXCEEDED): es el timeout del contrato, no una falla genérica.
    if status == 504:
        return ProviderTimeoutError(f"gemini: la llamada superó el tiempo máximo ({detail})")
    if status == 408 or (status is not None and status >= 500):
        return TransientProviderError(f"gemini: falla temporal ({detail})")
    if status is not None and 400 <= status < 500:
        return InvalidRequestError(f"gemini: pedido rechazado ({detail})")
    return LLMError(f"gemini: error inesperado ({detail})")


class GeminiProvider:
    """Adaptador para la API de Gemini (SDK `google-genai`)."""

    name = "gemini"

    def __init__(self, api_key: str) -> None:
        # attempts=1: sin reintentos del SDK. Hoy ya es su comportamiento por
        # defecto, pero se deja explícito para no depender de eso.
        self._client = GeminiClient(
            api_key=api_key,
            http_options=genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=1)),
        )

    def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        timeout_seconds: float,
        output_schema: dict[str, Any] | None = None,
    ) -> ProviderResult:
        # Google rechaza con 400 un plazo menor: se avisa antes de gastar la llamada.
        if timeout_seconds < GEMINI_MIN_TIMEOUT_SECONDS:
            raise ProviderConfigurationError(
                f"gemini: el timeout mínimo es {GEMINI_MIN_TIMEOUT_SECONDS} s, llegó {timeout_seconds} s"
            )
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            # El SDK toma el timeout en milisegundos. Va en cada llamada porque
            # en el contrato es un argumento de generate.
            http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            # Sin llamado automático de funciones: las tools las ejecuta el
            # grafo, que es donde se validan y se registran.
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )
        if output_schema is not None:
            # response_json_schema recibe JSON Schema estándar, el que genera
            # Pydantic, sin adaptarlo (a diferencia de response_schema).
            config.response_mime_type = "application/json"
            config.response_json_schema = output_schema
        try:
            # El timeout del SDK no corta la duración total (ver call_with_deadline).
            response = call_with_deadline(
                lambda: self._client.models.generate_content(model=model, contents=prompt, config=config),
                timeout_seconds,
                label="gemini",
            )
        except genai_errors.APIError as exc:
            raise _translate_gemini_error(exc) from exc
        except _GEMINI_TIMEOUT_ERRORS as exc:
            raise ProviderTimeoutError(
                f"gemini: la llamada superó el tiempo máximo ({type(exc).__name__})"
            ) from exc
        except _GEMINI_TRANSPORT_ERRORS as exc:
            raise TransientProviderError(f"gemini: no se pudo conectar ({type(exc).__name__})") from exc
        return _normalize_gemini_response(response, requested_model=model)


def _normalize_gemini_response(
    response: genai_types.GenerateContentResponse, *, requested_model: str
) -> ProviderResult:
    """Lleva la respuesta de Gemini a `ProviderResult`."""
    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason if candidate else None
    blocked = response.prompt_feedback.block_reason if response.prompt_feedback else None
    usage = response.usage_metadata
    # Los tokens de razonamiento se cobran como salida aunque no lleguen en el
    # texto: se suman para que el registro de costo sea real.
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
    if finish is not None:
        stop_reason: str | None = finish.name
    elif blocked is not None:
        stop_reason = f"PROMPT_BLOCKED_{blocked.name}"
    else:
        stop_reason = None
    return ProviderResult(
        # response.text omite el razonamiento y es None si no hubo respuesta.
        text=response.text or "",
        model=response.model_version or requested_model,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        truncated=finish == genai_types.FinishReason.MAX_TOKENS,
        refused=blocked is not None or finish in _GEMINI_REFUSAL_REASONS,
    )
