"""Contrato común a todos los proveedores de LLM.

No importa ningún SDK: es lo que el resto del sistema conoce de un proveedor.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from yourebrand.core.tools import ToolCall, ToolResult


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


class Message(BaseModel):
    """Un mensaje de la conversación, en un formato que no depende del proveedor.

    Es lo que guarda el estado del grafo. Cada resultado de tool es un mensaje
    propio (rol "tool"), así el historial es una lista plana; cada proveedor lo
    convierte a su forma.
    """

    role: Literal["user", "model", "tool"]
    text: str | None = None
    # Rol "model": las tools que pidió el modelo en este mensaje.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Rol "tool": a qué pedido responde y qué devolvió la tool.
    tool_call_id: str | None = None
    tool_name: str | None = None
    result: ToolResult | None = None
    # Rol "model": la respuesta original del proveedor, sin tocar. Se reenvía
    # tal cual porque puede traer datos que el proveedor exige recibir de vuelta
    # (en Gemini, las firmas del razonamiento). Solo la entiende el proveedor
    # que la generó; los demás la ignoran y usan los campos neutrales.
    provider_raw: dict[str, Any] | None = None

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", text=text)

    @classmethod
    def tool(cls, call: ToolCall, result: ToolResult) -> "Message":
        """Resultado de una tool, emparejado con el pedido que lo originó."""
        return cls(role="tool", tool_call_id=call.id, tool_name=call.tool_name, result=result)


@dataclass(frozen=True)
class TurnResult:
    """Un turno del modelo: su mensaje (texto o pedidos de tools) y el uso de la llamada."""

    message: Message
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    truncated: bool
    refused: bool = False
    # El modelo intentó pedir una tool pero el pedido llegó mal armado. Es
    # recuperable: el grafo le avisa y le da otra oportunidad.
    malformed_tool_call: bool = False


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
        """Un turno de conversación con tools disponibles.

        `tools` son las declaraciones de `ToolRegistry.function_declarations()`.
        El modelo responde con texto, con pedidos de tools o con ambos. El
        proveedor nunca ejecuta las tools: eso lo hace quien llama.

        Mismas garantías que `generate`: plazo, un solo intento y errores propios.
        """
        ...
