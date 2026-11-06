"""Eventos de trazabilidad: un solo punto que escribe el log y arma la traza del turno.

Cada módulo emite lo que pasa con `emit(logger, "llm_call", ...)`. Eso hace dos cosas:
- escribe la línea de log de siempre (`llm_call run_id=... model=...`);
- si hay una traza abierta para el run en curso, le suma el evento con sus campos.

Así la pantalla, el set de evaluación y los logs cuentan lo mismo, y un evento
nuevo aparece en los tres sin tocar nada más.

La traza en curso vive en una variable de contexto (`contextvars`): cada turno
tiene la suya y dos sesiones simultáneas no se mezclan. Las variables de
contexto no pasan solas a un hilo nuevo; `run_with_deadline` las copia.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

# Valores que se loguean: se recortan para que una línea de log no se vuelva un párrafo.
_MAX_LOG_VALUE_CHARS = 200


class TraceEvent(BaseModel):
    """Algo que pasó durante un turno, con sus datos estructurados."""

    run_id: str
    event: str  # nombre estable, por ejemplo "llm_call" o "tool_call"
    at_ms: float  # milisegundos desde que empezó el turno
    depth: int = 0  # 0: el agente; 1: algo que pasó adentro de una tool
    fields: dict[str, Any] = Field(default_factory=dict)
    # Datos que se muestran en la traza de la pantalla pero nunca van al log,
    # por ejemplo los argumentos de una tool (pueden traer texto del cliente).
    private: dict[str, Any] = Field(default_factory=dict)


class TraceCollector:
    """Junta los eventos de un turno. Varios hilos pueden emitir a la vez."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._started = perf_counter()
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()

    def add(self, event: str, depth: int, fields: dict[str, Any], private: dict[str, Any]) -> None:
        item = TraceEvent(
            run_id=self.run_id,
            event=event,
            at_ms=round((perf_counter() - self._started) * 1000, 2),
            depth=depth,
            fields=fields,
            private=private,
        )
        with self._lock:
            self._events.append(item)

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)


_collector: ContextVar[TraceCollector | None] = ContextVar("trace_collector", default=None)
_depth: ContextVar[int] = ContextVar("trace_depth", default=0)


@contextmanager
def collect(run_id: str) -> Iterator[TraceCollector]:
    """Abre la traza de un turno: todo lo que se emita adentro se suma a ella."""
    collector = TraceCollector(run_id)
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


@contextmanager
def nested() -> Iterator[None]:
    """Marca que lo que se emite adentro pasa dentro de otro paso (por ejemplo, una tool)."""
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value) or "-"
    text = str(value)
    return text if len(text) <= _MAX_LOG_VALUE_CHARS else text[:_MAX_LOG_VALUE_CHARS] + "…"


def emit(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    private: dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Registra un evento: lo escribe en el log y, si hay una traza abierta, lo suma.

    `fields` va a los dos lugares, así que nunca lleva contenido del cliente:
    solo nombres, conteos, tiempos y tipos de error. `private` va solo a la traza.
    """
    if logger.isEnabledFor(level):
        parts = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
        logger.log(level, "%s %s", event, parts)
    collector = _collector.get()
    if collector is not None:
        collector.add(event, _depth.get(), dict(fields), dict(private or {}))
