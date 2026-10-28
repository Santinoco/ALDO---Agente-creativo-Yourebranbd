"""Proveedores de LLM: único módulo que importa SDKs de proveedores.

Para sumar un proveedor: crear una clase que cumpla `LLMProvider`,
agregarla en `build_provider` y sumar su nombre a `llm_provider` en config.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx
from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    transform_schema,
)
from google.genai import Client as GeminiClient
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import SecretStr

from yourebrand.core.config import Settings, get_settings
from yourebrand.core.errors import (
    InvalidRequestError,
    LLMError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)


@dataclass(frozen=True)
class ProviderResult:
    """Respuesta normalizada: igual para todos los proveedores."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    truncated: bool  # la respuesta se cortó por llegar a max_tokens
    refused: bool = False  # el modelo se negó a responder


class LLMProvider(Protocol):
    """Contrato que cumple todo proveedor de generación de texto."""

    name: str

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
        """Genera texto. Con `output_schema` (JSON Schema estándar), el texto es un JSON con esa forma.

        Cómo se consigue esa forma es asunto de cada proveedor. Quien llama
        valida igual el resultado: el contrato no garantiza que el JSON cumpla.

        Todo proveedor debe:
        - cortar la llamada al pasar `timeout_seconds`;
        - hacer un solo intento: los reintentos son de `core/llm.py`;
        - lanzar solo errores de `core/errors.py`, nunca los de su SDK.
        """
        ...


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _translate_anthropic_error(exc: APIError) -> LLMError:
    """Convierte una excepción del SDK de Anthropic en un error de `core/errors.py`."""
    # El mensaje lleva solo el tipo y el código. El detalle queda en la excepción
    # original, encadenada como causa.
    detail = type(exc).__name__
    # APITimeoutError hereda de APIConnectionError: se pregunta primero.
    if isinstance(exc, APITimeoutError):
        return ProviderTimeoutError(f"anthropic: la llamada superó el tiempo máximo ({detail})")
    if isinstance(exc, APIConnectionError):
        return TransientProviderError(f"anthropic: no se pudo conectar ({detail})")
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        detail = f"{detail}, HTTP {status}"
        if status == 429:
            return RateLimitError(
                f"anthropic: límite de uso ({detail})", retry_after_seconds=_retry_after_seconds(exc)
            )
        if status in (401, 403, 404):
            return ProviderConfigurationError(f"anthropic: clave, permiso o modelo inválido ({detail})")
        if status == 408 or status >= 500:
            return TransientProviderError(f"anthropic: falla temporal ({detail})")
        return InvalidRequestError(f"anthropic: pedido rechazado ({detail})")
    return LLMError(f"anthropic: error inesperado ({detail})")


class AnthropicProvider:
    """Adaptador para la API de Claude."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        # max_retries=0: el SDK reintenta por su cuenta y eso se sumaría a los
        # reintentos de core/llm.py, que son los que quedan registrados.
        self._client = Anthropic(api_key=api_key, max_retries=0)

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
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout_seconds,
        }
        if output_schema is not None:
            # transform_schema adapta el JSON Schema a lo que admite la API de Claude
            # (por ejemplo, quita largos mínimos). Es un detalle de este proveedor.
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": transform_schema(output_schema)}
            }
        try:
            message = self._client.messages.create(**request)
        except APIError as exc:
            raise _translate_anthropic_error(exc) from exc
        return ProviderResult(
            text="".join(block.text for block in message.content if block.type == "text"),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
            truncated=message.stop_reason == "max_tokens",
            refused=message.stop_reason == "refusal",
        )


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


def _call_with_deadline[T](fn: Callable[[], T], timeout_seconds: float) -> T:
    """Corre `fn` y lanza ProviderTimeoutError si no termina en `timeout_seconds` de reloj.

    El timeout del SDK de Google no alcanza: en httpx el de lectura es el tiempo
    entre dos datos recibidos, no la duración total, y en una prueba real una
    llamada con 120 s de timeout terminó a los 410 s. Si se vence el plazo, el
    hilo sigue hasta que Google responda (y esa llamada se cobra), pero quien
    llamó queda libre. Es daemon para no frenar la salida del proceso.
    """
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # se relanza en el hilo de quien llamó
            outcome["error"] = exc

    worker = threading.Thread(target=run, name="gemini-call", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise ProviderTimeoutError(f"gemini: la llamada superó el tiempo máximo ({timeout_seconds} s)")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


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
            response = _call_with_deadline(
                lambda: self._client.models.generate_content(model=model, contents=prompt, config=config),
                timeout_seconds,
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
