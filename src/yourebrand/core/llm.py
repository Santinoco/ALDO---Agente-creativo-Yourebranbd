"""Única puerta a los modelos de lenguaje para el resto del sistema.

Acá vive lo común a cualquier proveedor (elegir modelo, medir, registrar uso).
Lo específico de cada proveedor vive en `core/providers.py`.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from yourebrand.core.config import get_settings
from yourebrand.core.providers import ProviderResult, get_provider
from yourebrand.core.tenancy import TenantContext

logger = logging.getLogger(__name__)

Tier = Literal["generation", "fast"]


@dataclass(frozen=True)
class LLMResponse:
    """Respuesta del modelo más los datos de uso de la llamada."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None
    run_id: str


@dataclass(frozen=True)
class StructuredResponse[T: BaseModel]:
    """Salida ya validada contra un schema, más los datos de uso de la llamada."""

    data: T
    response: LLMResponse


class LLMOutputError(RuntimeError):
    """La llamada terminó pero su salida no se puede usar."""


class TruncatedOutputError(LLMOutputError):
    """La respuesta se cortó por llegar a max_tokens."""


class RefusedOutputError(LLMOutputError):
    """El modelo se negó a responder."""


class InvalidOutputError(LLMOutputError):
    """La respuesta no cumple el schema pedido."""


def new_run_id() -> str:
    """Identificador único para agrupar todas las llamadas de una misma ejecución."""
    return uuid.uuid4().hex


def _model_for(tier: Tier) -> str:
    settings = get_settings()
    return settings.model_generation if tier == "generation" else settings.model_fast


def _call(
    *,
    system: str,
    prompt: str,
    tenant: TenantContext,
    run_id: str,
    tier: Tier,
    max_tokens: int,
    output_schema: dict[str, Any] | None = None,
) -> tuple[LLMResponse, ProviderResult]:
    """Llama al proveedor configurado, mide la llamada y registra su uso."""
    provider = get_provider()
    request: dict[str, Any] = {
        "model": _model_for(tier),
        "system": system,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    if output_schema is not None:
        request["output_schema"] = output_schema

    started = time.perf_counter()
    result = provider.generate(**request)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    response = LLMResponse(
        text=result.text,
        provider=provider.name,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        stop_reason=result.stop_reason,
        run_id=run_id,
    )

    # Se registran tokens y metadatos, nunca el prompt ni la respuesta (datos del cliente).
    logger.info(
        "llm_call run_id=%s client_id=%s provider=%s model=%s "
        "input_tokens=%d output_tokens=%d latency_ms=%.2f stop_reason=%s",
        run_id,
        tenant.client_id,
        response.provider,
        response.model,
        response.input_tokens,
        response.output_tokens,
        response.latency_ms,
        response.stop_reason,
    )
    if result.truncated:
        logger.warning("llm_call run_id=%s respuesta cortada por max_tokens=%d", run_id, max_tokens)

    return response, result


def complete(
    *,
    system: str,
    prompt: str,
    tenant: TenantContext,
    run_id: str,
    tier: Tier = "generation",
    max_tokens: int = 2000,
) -> LLMResponse:
    """Hace una llamada al modelo y registra su uso.

    `tenant` solo sirve para atribuir el uso a un cliente: el aislamiento
    depende de qué contexto se arma en el prompt, no de este módulo.
    """
    response, _ = _call(
        system=system, prompt=prompt, tenant=tenant, run_id=run_id, tier=tier, max_tokens=max_tokens
    )
    return response


def complete_structured[T: BaseModel](
    *,
    system: str,
    prompt: str,
    schema: type[T],
    tenant: TenantContext,
    run_id: str,
    tier: Tier = "generation",
    max_tokens: int = 16000,
) -> StructuredResponse[T]:
    """Hace una llamada al modelo y devuelve su salida validada contra `schema`.

    Al proveedor le llega un JSON Schema estándar y devuelve texto. La validación
    se hace acá, igual para todos los proveedores.
    """
    response, result = _call(
        system=system,
        prompt=prompt,
        tenant=tenant,
        run_id=run_id,
        tier=tier,
        max_tokens=max_tokens,
        output_schema=schema.model_json_schema(),
    )

    if result.refused:
        raise RefusedOutputError(f"run_id={run_id}: el modelo se negó a responder")
    if result.truncated:
        raise TruncatedOutputError(f"run_id={run_id}: respuesta cortada por max_tokens={max_tokens}")
    try:
        data = schema.model_validate_json(result.text)
    except ValidationError as exc:
        # Se informa qué campos fallaron, nunca sus valores (datos del cliente).
        # `from None` evita que el error de Pydantic, que sí los trae, quede encadenado.
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors(include_input=False)})
        raise InvalidOutputError(f"run_id={run_id}: la salida no cumple el schema en {fields}") from None

    return StructuredResponse(data=data, response=response)
