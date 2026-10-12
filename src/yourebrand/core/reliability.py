"""Confiabilidad común a todo el sistema: validar texto de usuarios y reintentar fallas temporales.

No depende de ningún proveedor. Los reintentos viven acá y no en los SDKs para
que el comportamiento sea el mismo con cualquier proveedor.
"""

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from yourebrand.core.errors import RateLimitError, TransientProviderError

MAX_USER_TEXT_CHARS = 4000

# Se permiten tabulación, salto de línea y retorno de carro.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InvalidUserInputError(ValueError):
    """El texto que escribió el usuario no se puede procesar."""


def validate_user_text(text: str, *, max_len: int = MAX_USER_TEXT_CHARS) -> str:
    """Devuelve el texto sin espacios sobrantes, o falla si no se puede usar.

    Controla la forma del texto, no su intención: no detecta ni frena una
    inyección de prompt.
    """
    # Los mensajes de error no citan el texto: puede traer datos del cliente.
    if not isinstance(text, str):
        raise InvalidUserInputError("el texto debe ser str")
    cleaned = text.strip()
    if not cleaned:
        raise InvalidUserInputError("el texto está vacío")
    if len(cleaned) > max_len:
        raise InvalidUserInputError(f"el texto supera los {max_len} caracteres")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise InvalidUserInputError("el texto contiene caracteres de control")
    return cleaned


@dataclass(frozen=True)
class RetryEvent:
    """Un intento fallido que se va a reintentar."""

    attempt: int
    error_type: str
    delay_seconds: float


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 30.0,
    jitter: bool = True,
    on_retry: Callable[[RetryEvent], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Ejecuta `fn` y la reintenta ante fallas temporales. Devuelve el resultado y la cantidad de intentos.

    Solo se reintenta `TransientProviderError`: cualquier otro error se propaga
    en el primer intento, porque repetir la llamada daría el mismo resultado.
    """
    attempt = 1
    while True:
        try:
            return fn(), attempt
        except TransientProviderError as exc:
            if attempt > max_retries:
                raise

            retry_after = exc.retry_after_seconds if isinstance(exc, RateLimitError) else None
            if retry_after is not None:
                delay = retry_after
            else:
                delay = base_delay_seconds * 2 ** (attempt - 1)
                if jitter:
                    # Evita que varias llamadas que fallaron juntas reintenten a la vez.
                    delay *= random.uniform(0.8, 1.2)
            # El tope también vale para la espera que pide el proveedor: hay una
            # persona esperando la respuesta.
            delay = min(delay, max_delay_seconds)

            if on_retry is not None:
                on_retry(RetryEvent(attempt=attempt, error_type=type(exc).__name__, delay_seconds=delay))
            sleep(delay)
            attempt += 1
