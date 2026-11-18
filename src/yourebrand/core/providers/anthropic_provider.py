"""Adaptador para la API de Claude: único módulo que importa el SDK de Anthropic."""

import base64
import json
from collections.abc import Sequence
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    transform_schema,
)

from yourebrand.core.errors import (
    InvalidRequestError,
    LLMError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)
from yourebrand.core.providers.base import ImageInput, Message, ProviderResult, TurnResult, tool_error_payload
from yourebrand.core.tools import ToolCall


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


def _user_content(prompt: str, images: Sequence[ImageInput]) -> str | list[dict[str, Any]]:
    """Contenido del mensaje: el texto solo, o las imágenes (cada una con su rótulo) y después el texto.

    Las imágenes van primero: la documentación de Claude recomienda ponerlas
    antes de la consigna que las usa.
    """
    if not images:
        return prompt
    blocks: list[dict[str, Any]] = []
    for image in images:
        if image.label:
            blocks.append({"type": "text", "text": image.label})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64.standard_b64encode(image.data).decode("ascii"),
            },
        })
    blocks.append({"type": "text", "text": prompt})
    return blocks


def _tool_result_block(message: Message) -> dict[str, Any]:
    """Un ToolResult en la forma de Claude: un bloque tool_result, con is_error si falló.

    Qué se manda de un error lo decide el contrato (tool_error_payload); acá
    solo va como JSON en el contenido del bloque.
    """
    result = message.result
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": message.tool_call_id}
    if result is not None and result.ok:
        return block | {"content": result.content}
    payload = json.dumps(tool_error_payload(result), ensure_ascii=False, default=str)
    return block | {"content": payload, "is_error": True}


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convierte el historial neutral al formato de Claude.

    Claude no tiene rol "tool": los resultados van como bloques tool_result en
    un mensaje "user", todos los de un mismo turno juntos (separarlos le enseña
    al modelo a dejar de pedir tools en paralelo).
    """
    converted: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            converted.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if message.role == "tool":
            pending_results.append(_tool_result_block(message))
            continue
        flush_results()
        if message.role == "user":
            converted.append({"role": "user", "content": message.text or ""})
        elif message.provider_raw is not None and message.provider == "anthropic":
            # La respuesta original, idéntica: los bloques de razonamiento
            # (thinking) tienen que volver sin tocar en los turnos con tools.
            converted.append({"role": "assistant", "content": message.provider_raw["content"]})
        else:
            # Mensaje de otro proveedor (por ejemplo, de una sesión que empezó
            # con Gemini): se arma con los campos neutrales.
            blocks: list[dict[str, Any]] = [{"type": "text", "text": message.text}] if message.text else []
            blocks += [
                {"type": "tool_use", "id": call.id, "name": call.tool_name, "input": call.arguments}
                for call in message.tool_calls
            ]
            # La API rechaza un mensaje sin contenido.
            converted.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
    flush_results()
    return converted


def _message_from_response(response: Any) -> Message:
    """Convierte la respuesta de Claude en un Message neutral, guardando la original."""
    # Los bloques thinking no son respuesta para la persona: no entran al texto.
    texts = [block.text for block in response.content if block.type == "text"]
    calls = [
        ToolCall(id=block.id, tool_name=block.name, arguments=dict(block.input or {}))
        for block in response.content
        if block.type == "tool_use"
    ]
    return Message(
        role="model",
        text="".join(texts) or None,
        tool_calls=calls,
        # mode="json": queda como dict guardable (el checkpointer) sin perder
        # las firmas de los bloques thinking.
        provider_raw={"content": [block.model_dump(mode="json", exclude_none=True) for block in response.content]},
        provider="anthropic",
    )


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
        images: Sequence[ImageInput] = (),
    ) -> ProviderResult:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": _user_content(prompt, images)}],
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
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _to_anthropic_messages(messages),
            "timeout": timeout_seconds,
        }
        if tools:
            # Sin tool_choice: el default es "auto", el modelo decide si pide una
            # tool o responde con texto. Es la selección dinámica de la consigna.
            request["tools"] = [
                {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
                for tool in tools
            ]
        try:
            response = self._client.messages.create(**request)
        except APIError as exc:
            raise _translate_anthropic_error(exc) from exc
        return TurnResult(
            message=_message_from_response(response),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            truncated=response.stop_reason == "max_tokens",
            refused=response.stop_reason == "refusal",
            # Claude valida la forma del pedido de tool en el servidor: no hay
            # un equivalente al MALFORMED_FUNCTION_CALL de Gemini. Los
            # argumentos igual los valida el registry.
            malformed_tool_call=False,
        )
