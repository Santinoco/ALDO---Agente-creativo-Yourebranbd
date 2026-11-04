"""Adaptador para la API de Gemini: único módulo que importa el SDK de Google."""

import re
import uuid
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
from yourebrand.core.providers.base import Message, ProviderResult, TurnResult
from yourebrand.core.reliability import call_with_deadline
from yourebrand.core.tools import ToolCall, ToolResult


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

# Motivos de corte de un pedido de tool mal armado: recuperables, el grafo le
# da al modelo otra oportunidad.
_GEMINI_MALFORMED_REASONS = {
    genai_types.FinishReason.MALFORMED_FUNCTION_CALL,
    genai_types.FinishReason.UNEXPECTED_TOOL_CALL,
}

# Prefijo de los ids de ToolCall que generamos cuando Gemini no manda uno.
_LOCAL_CALL_ID_PREFIX = "local-"

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
        config = _base_config(system=system, max_tokens=max_tokens, timeout_seconds=timeout_seconds)
        if output_schema is not None:
            # response_json_schema recibe JSON Schema estándar, el que genera
            # Pydantic, sin adaptarlo (a diferencia de response_schema).
            config.response_mime_type = "application/json"
            config.response_json_schema = output_schema
        response = self._send(model=model, contents=prompt, config=config, timeout_seconds=timeout_seconds)
        usage = _usage_fields(response, requested_model=model)
        # response.text omite el razonamiento y es None si no hubo respuesta.
        return ProviderResult(text=response.text or "", **usage)

    def generate_turn(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int,
        timeout_seconds: float,
    ) -> TurnResult:
        config = _base_config(system=system, max_tokens=max_tokens, timeout_seconds=timeout_seconds)
        if tools:
            # Las declaraciones del registry ya tienen la forma que pide Gemini:
            # nombre, descripción y JSON Schema de los argumentos.
            config.tools = [
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters_json_schema=tool["parameters"],
                        )
                        for tool in tools
                    ]
                )
            ]
            # AUTO: el modelo decide si pide una tool o responde con texto. Es la
            # selección dinámica que pide la consigna.
            config.tool_config = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode=genai_types.FunctionCallingConfigMode.AUTO
                )
            )
        response = self._send(
            model=model, contents=_to_gemini_contents(messages), config=config, timeout_seconds=timeout_seconds
        )
        usage = _usage_fields(response, requested_model=model)
        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason if candidate else None
        return TurnResult(
            message=_message_from_response(response),
            malformed_tool_call=finish in _GEMINI_MALFORMED_REASONS,
            **usage,
        )

    def _send(
        self,
        *,
        model: str,
        contents: str | list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        timeout_seconds: float,
    ) -> genai_types.GenerateContentResponse:
        """Hace la llamada con plazo por reloj y traduce los errores del SDK."""
        try:
            # El timeout del SDK no corta la duración total (ver call_with_deadline).
            return call_with_deadline(
                lambda: self._client.models.generate_content(model=model, contents=contents, config=config),
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


def _base_config(*, system: str, max_tokens: int, timeout_seconds: float) -> genai_types.GenerateContentConfig:
    """Configuración común a toda llamada: instrucciones, tope de tokens, plazo y sin AFC."""
    # Google rechaza con 400 un plazo menor: se avisa antes de gastar la llamada.
    if timeout_seconds < GEMINI_MIN_TIMEOUT_SECONDS:
        raise ProviderConfigurationError(
            f"gemini: el timeout mínimo es {GEMINI_MIN_TIMEOUT_SECONDS} s, llegó {timeout_seconds} s"
        )
    return genai_types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        # El SDK toma el timeout en milisegundos. Va en cada llamada porque
        # en el contrato es un argumento de generate.
        http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        # Sin llamado automático de funciones: las tools las ejecuta el
        # grafo, que es donde se validan y se registran.
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    )


def _usage_fields(response: genai_types.GenerateContentResponse, *, requested_model: str) -> dict[str, Any]:
    """Datos de uso y de corte comunes a ProviderResult y TurnResult."""
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
    return {
        "model": response.model_version or requested_model,
        "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "truncated": finish == genai_types.FinishReason.MAX_TOKENS,
        "refused": blocked is not None or finish in _GEMINI_REFUSAL_REASONS,
    }


def _to_gemini_contents(messages: list[Message]) -> list[genai_types.Content]:
    """Convierte el historial neutral al formato de Gemini."""
    contents: list[genai_types.Content] = []
    # Gemini no tiene rol "tool": los resultados van como partes function_response
    # de un mensaje "user". Los seguidos se juntan en uno solo, como los devolvería
    # Gemini si el modelo pidió varias tools a la vez.
    pending_results: list[genai_types.Part] = []

    def flush_results() -> None:
        if pending_results:
            contents.append(genai_types.Content(role="user", parts=list(pending_results)))
            pending_results.clear()

    for message in messages:
        if message.role == "tool":
            pending_results.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        id=_api_call_id(message.tool_call_id),
                        name=message.tool_name,
                        response=_tool_response_payload(message.result),
                    )
                )
            )
            continue
        flush_results()
        if message.role == "user":
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=message.text or "")]))
        elif message.provider_raw is not None:
            # La respuesta original, idéntica: así viajan de vuelta las firmas
            # del razonamiento, estén en la parte que estén.
            contents.append(genai_types.Content.model_validate(message.provider_raw))
        else:
            # Mensaje del modelo sin respuesta original (por ejemplo, de otro
            # proveedor): se arma con los campos neutrales.
            parts = [genai_types.Part(text=message.text)] if message.text else []
            parts += [
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=_api_call_id(call.id), name=call.tool_name, args=call.arguments
                    )
                )
                for call in message.tool_calls
            ]
            contents.append(genai_types.Content(role="model", parts=parts))
    flush_results()
    return contents


def _tool_response_payload(result: ToolResult | None) -> dict[str, Any]:
    """Lo que ve el modelo de un ToolResult: el contenido, o el error con lo que hace falta para corregirlo.

    La metadata de runtime (latencia, intento) no se manda: para el modelo es
    ruido y suma tokens.
    """
    if result is None:
        return {"error": "la tool no devolvió resultado"}
    if result.ok:
        return {"result": result.content}
    payload = {
        "error": result.content,
        "error_type": result.metadata.get("error_type"),
        # Qué argumento falló y por qué, cuando el error es de validación.
        "details": result.metadata.get("details"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _message_from_response(response: genai_types.GenerateContentResponse) -> Message:
    """Convierte la respuesta de Gemini en un Message neutral, guardando la original."""
    candidate = response.candidates[0] if response.candidates else None
    content = candidate.content if candidate else None
    parts = (content.parts or []) if content else []
    # Las partes de razonamiento (thought) no son respuesta para la persona.
    texts = [part.text for part in parts if part.text and not part.thought]
    calls = [
        ToolCall(
            # Si Gemini no manda id, se genera uno local para emparejar el
            # resultado con su pedido; ese id no se le reenvía (ver _api_call_id).
            id=part.function_call.id or f"{_LOCAL_CALL_ID_PREFIX}{uuid.uuid4().hex}",
            tool_name=part.function_call.name or "",
            arguments=part.function_call.args or {},
        )
        for part in parts
        if part.function_call
    ]
    return Message(
        role="model",
        text="".join(texts) or None,
        tool_calls=calls,
        # mode="json": los bytes de las firmas quedan en base64 y el dict se puede
        # guardar como JSON (el checkpointer de la iteración 3) sin perderlas.
        provider_raw=content.model_dump(mode="json", exclude_none=True) if content else None,
    )


def _api_call_id(call_id: str | None) -> str | None:
    """El id que se le manda a Gemini: None si lo generamos nosotros y Gemini no lo conoce."""
    if call_id is None or call_id.startswith(_LOCAL_CALL_ID_PREFIX):
        return None
    return call_id
