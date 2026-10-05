"""Única puerta a los modelos de lenguaje para el resto del sistema.

Acá vive lo común a cualquier proveedor (elegir modelo, medir, registrar uso).
Lo específico de cada proveedor vive en `core/providers.py`.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from yourebrand.core.config import get_settings
from yourebrand.core.providers import get_provider
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


def new_run_id() -> str:
    """Identificador único para agrupar todas las llamadas de una misma ejecución."""
    return uuid.uuid4().hex


def _model_for(tier: Tier) -> str:
    settings = get_settings()
    return settings.model_generation if tier == "generation" else settings.model_fast


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
    provider = get_provider()

    started = time.perf_counter()
    result = provider.generate(
        model=_model_for(tier),
        system=system,
        prompt=prompt,
        max_tokens=max_tokens,
    )
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

    return response
