"""Contrato común a todos los proveedores de LLM.

No importa ningún SDK: es lo que el resto del sistema conoce de un proveedor.
"""

from dataclasses import dataclass
from typing import Any, Protocol


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
