"""Única puerta a los modelos de lenguaje para el resto del sistema.

Acá vive lo común a cualquier proveedor (elegir modelo, medir, registrar uso).
Lo específico de cada proveedor vive en `core/providers/`.
"""

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from yourebrand.core.config import get_settings

# Los errores de salida se definen en core/errors.py; se importan acá también
# para quienes ya los usan desde este módulo.
from yourebrand.core.errors import (  # noqa: F401
    InvalidOutputError,
    LLMError,
    LLMOutputError,
    BudgetExceededError,
    ProviderConfigurationError,
    RefusedOutputError,
    TruncatedOutputError,
)
from yourebrand.core.providers import ImageInput, LLMProvider, Message, ProviderResult, get_provider
from yourebrand.core.reliability import RetryEvent, call_with_retry
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.trace import emit

logger = logging.getLogger(__name__)

# Tareas, no modelos: cada una se resuelve a un proveedor y un modelo en _route.
Tier = Literal["generation", "fast", "video_edit"]


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
    attempts: int  # 1 si no hubo reintentos


@dataclass(frozen=True)
class StructuredResponse[T: BaseModel]:
    """Salida ya validada contra un schema, más los datos de uso de la llamada."""

    data: T
    response: LLMResponse


@dataclass(frozen=True)
class TurnResponse:
    """Un turno del agente: el mensaje del modelo más los datos de uso de la llamada."""

    message: Message  # texto, pedidos de tools o ambos; listo para guardar en el historial
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None
    run_id: str
    attempts: int
    # El modelo intentó pedir una tool pero el pedido llegó mal armado: quien
    # llama decide si le da otra oportunidad.
    malformed_tool_call: bool = False


def new_run_id() -> str:
    """Identificador único para agrupar todas las llamadas de una misma ejecución."""
    return uuid.uuid4().hex


def _route(tier: Tier) -> tuple[LLMProvider, str]:
    """Qué proveedor y qué modelo atienden cada tarea.

    - generation y fast: el proveedor de LLM_PROVIDER, el de la conversación.
    - video_edit: siempre Claude, con MODEL_VIDEO_EDIT, aunque la conversación
      use otro proveedor. Quien llama pide la tarea, no el proveedor.
    """
    settings = get_settings()
    if tier == "video_edit":
        if not settings.model_video_edit:
            raise ProviderConfigurationError("edición de video deshabilitada: falta MODEL_VIDEO_EDIT")
        return get_provider("anthropic"), settings.model_video_edit
    model = settings.model_generation if tier == "generation" else settings.model_fast
    return get_provider(), model


@dataclass(frozen=True)
class _Limits:
    """Plazo, reintentos y topes de tokens de una tarea."""

    timeout_seconds: float
    retry_timeouts: bool
    max_output_tokens: int | None  # None: el que pida quien llama
    max_input_tokens: int | None  # None: sin tope de entrada


def _limits(tier: Tier) -> _Limits:
    """Los topes de cada tarea. La edición de video tiene los suyos: es la llamada cara."""
    settings = get_settings()
    if tier == "video_edit":
        return _Limits(
            timeout_seconds=settings.video_edit_timeout_seconds,
            retry_timeouts=False,
            max_output_tokens=settings.video_edit_max_output_tokens,
            max_input_tokens=settings.video_edit_max_input_tokens,
        )
    return _Limits(settings.llm_timeout_seconds, True, None, None)


# Estimación de tokens de entrada, sin llamar a nadie. Es deliberadamente
# pesimista: sirve para cortar antes de gastar, no para facturar.
_CHARS_PER_TOKEN = 3  # el español ronda 3,5 a 4 caracteres por token
# Claude achica toda imagen a 1568 px de lado mayor y ~1,15 megapíxeles, y
# cobra ~ancho×alto/750 tokens. Sin el tamaño, se toma el máximo.
_IMAGE_MAX_EDGE_PX = 1568
_IMAGE_MAX_PIXELS = 1_150_000
_PIXELS_PER_TOKEN = 750
_IMAGE_MAX_TOKENS = _IMAGE_MAX_PIXELS // _PIXELS_PER_TOKEN + 1


def estimate_image_tokens(image: ImageInput) -> int:
    if not image.width or not image.height:
        return _IMAGE_MAX_TOKENS
    scale = min(1.0, _IMAGE_MAX_EDGE_PX / max(image.width, image.height))
    pixels = min(image.width * image.height * scale * scale, _IMAGE_MAX_PIXELS)
    return int(pixels // _PIXELS_PER_TOKEN) + 1


def estimate_input_tokens(*, system: str, prompt: str, images: Sequence[ImageInput] = ()) -> int:
    """Tokens de entrada estimados (por arriba) de una llamada."""
    labels = sum(len(image.label or "") for image in images)
    text_tokens = (len(system) + len(prompt) + labels) // _CHARS_PER_TOKEN + 1
    return text_tokens + sum(estimate_image_tokens(image) for image in images)


def _with_retries[T](
    call: Callable[[], T],
    *,
    tenant: TenantContext,
    run_id: str,
    provider_name: str,
    retry_timeouts: bool = True,
) -> tuple[T, int, float]:
    """Corre una llamada al proveedor con reintentos. Devuelve resultado, intentos y latencia en ms.

    Es lo común a `generate` y `generate_turn`: reintentar, medir y registrar
    los reintentos y las fallas.
    """
    settings = get_settings()

    def log_retry(event: RetryEvent) -> None:
        emit(
            logger,
            "llm_retry",
            level=logging.WARNING,
            # El mensaje explica la falla en la pantalla, pero no va al log: no
            # se puede garantizar que ningún proveedor meta contenido en él.
            private={"message": event.message},
            run_id=run_id,
            client_id=tenant.client_id,
            attempt=event.attempt,
            error=event.error_type,
            delay_s=event.delay_seconds,
        )

    # La latencia es la que esperó quien llamó: incluye reintentos y esperas.
    started = time.perf_counter()
    try:
        result, attempts = call_with_retry(
            call, max_retries=settings.llm_max_retries, on_retry=log_retry, retry_timeouts=retry_timeouts
        )
    except LLMError as exc:
        # El mensaje del error (proveedor, tipo, código HTTP) explica la falla
        # en la pantalla; como en los reintentos, no va al log.
        emit(
            logger,
            "llm_call_failed",
            level=logging.ERROR,
            private={"message": str(exc)},
            run_id=run_id,
            client_id=tenant.client_id,
            provider=provider_name,
            error=type(exc).__name__,
        )
        raise
    return result, attempts, round((time.perf_counter() - started) * 1000, 2)


def _call(
    *,
    system: str,
    prompt: str,
    tenant: TenantContext,
    run_id: str,
    tier: Tier,
    max_tokens: int,
    output_schema: dict[str, Any] | None = None,
    images: Sequence[ImageInput] = (),
) -> tuple[LLMResponse, ProviderResult]:
    """Llama al proveedor de la tarea, reintenta fallas temporales, mide y registra el uso."""
    provider, model = _route(tier)
    limits = _limits(tier)
    # El tope de la tarea gana sobre lo que pida quien llama.
    if limits.max_output_tokens is not None:
        max_tokens = min(max_tokens, limits.max_output_tokens)
    if limits.max_input_tokens is not None:
        estimated = estimate_input_tokens(system=system, prompt=prompt, images=images)
        if estimated > limits.max_input_tokens:
            emit(logger, "llm_budget_exceeded", level=logging.WARNING, run_id=run_id, client_id=tenant.client_id,
                 tier=tier, estimated_input_tokens=estimated, max_input_tokens=limits.max_input_tokens)
            raise BudgetExceededError(
                f"run_id={run_id}: la entrada estimada ({estimated} tokens) supera el tope de "
                f"{limits.max_input_tokens}; no se llamó al modelo"
            )
    request: dict[str, Any] = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "timeout_seconds": limits.timeout_seconds,
    }
    if output_schema is not None:
        request["output_schema"] = output_schema
    # Solo si hay: así un proveedor sin soporte de imágenes no recibe el argumento.
    if images:
        request["images"] = list(images)

    result, attempts, latency_ms = _with_retries(
        lambda: provider.generate(**request),
        tenant=tenant,
        run_id=run_id,
        provider_name=provider.name,
        retry_timeouts=limits.retry_timeouts,
    )

    response = LLMResponse(
        text=result.text,
        provider=provider.name,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        stop_reason=result.stop_reason,
        run_id=run_id,
        attempts=attempts,
    )

    # Se registran tokens y metadatos, nunca el prompt ni la respuesta (datos del cliente).
    emit(
        logger,
        "llm_call",
        run_id=run_id,
        client_id=tenant.client_id,
        provider=response.provider,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        stop_reason=response.stop_reason,
        attempts=response.attempts,
        images=len(images),
        max_tokens=max_tokens,
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
    images: Sequence[ImageInput] = (),
) -> StructuredResponse[T]:
    """Hace una llamada al modelo y devuelve su salida validada contra `schema`.

    Al proveedor le llega un JSON Schema estándar y devuelve texto. La validación
    se hace acá, igual para todos los proveedores. `images` acompañan al prompt
    solo en esta llamada (por ejemplo, los fotogramas del plan de edición).
    """
    response, result = _call(
        system=system,
        prompt=prompt,
        tenant=tenant,
        run_id=run_id,
        tier=tier,
        max_tokens=max_tokens,
        output_schema=schema.model_json_schema(),
        images=images,
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


def complete_turn(
    *,
    system: str,
    messages: list[Message],
    tools: list[dict[str, Any]],
    tenant: TenantContext,
    run_id: str,
    tier: Tier = "generation",
    max_tokens: int = 8000,
) -> TurnResponse:
    """Un turno de conversación con tools: el modelo responde con texto o pide tools.

    Las tools no se ejecutan acá: el turno devuelve los pedidos y quien llama
    (el grafo) los ejecuta con el registry y le devuelve los resultados.
    """
    provider, model = _route(tier)
    limits = _limits(tier)
    if limits.max_output_tokens is not None:
        max_tokens = min(max_tokens, limits.max_output_tokens)
    result, attempts, latency_ms = _with_retries(
        lambda: provider.generate_turn(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            timeout_seconds=limits.timeout_seconds,
        ),
        tenant=tenant,
        run_id=run_id,
        provider_name=provider.name,
        retry_timeouts=limits.retry_timeouts,
    )

    # Qué tools pidió el modelo en este turno: es la evidencia de que la
    # selección es dinámica. Solo nombres; los argumentos van a la traza como
    # dato privado (pueden traer texto del cliente) y nunca al log.
    emit(
        logger,
        "llm_turn",
        # Lista y no dict por nombre: el modelo puede pedir la misma tool dos veces.
        private={
            "tool_calls": [{"tool": call.tool_name, "arguments": call.arguments} for call in result.message.tool_calls]
        },
        run_id=run_id,
        client_id=tenant.client_id,
        provider=provider.name,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        stop_reason=result.stop_reason,
        attempts=attempts,
        tools_requested=[call.tool_name for call in result.message.tool_calls],
        replied_with_text=bool(result.message.text),
        malformed_tool_call=result.malformed_tool_call,
    )

    if result.refused:
        raise RefusedOutputError(f"run_id={run_id}: el modelo se negó a responder")
    if result.truncated:
        raise TruncatedOutputError(f"run_id={run_id}: respuesta cortada por max_tokens={max_tokens}")
    # Un pedido de tool mal armado no lanza: es recuperable y lo resuelve el grafo.

    return TurnResponse(
        message=result.message,
        provider=provider.name,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=latency_ms,
        stop_reason=result.stop_reason,
        run_id=run_id,
        attempts=attempts,
        malformed_tool_call=result.malformed_tool_call,
    )
