"""Identidad del cliente (tenant): todo el sistema trabaja siempre para un cliente validado."""

import re
from dataclasses import dataclass
from pathlib import Path

from yourebrand.core.config import get_settings

# Solo minúsculas, números y guiones: "acme", "cafe-del-sur", "cliente-2".
_CLIENT_ID_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


class InvalidClientIdError(ValueError):
    """El client_id no tiene formato de slug."""


class UnknownClientError(LookupError):
    """El client_id es válido pero no hay datos para ese cliente."""


@dataclass(frozen=True)
class TenantContext:
    """Cliente para el que se está trabajando. Se valida al crearlo."""

    client_id: str

    def __post_init__(self) -> None:
        # El client_id forma parte de rutas de archivo: sin esta validación,
        # un valor como "../otro-cliente" leería datos de otro cliente.
        if not _CLIENT_ID_PATTERN.fullmatch(self.client_id):
            raise InvalidClientIdError(f"client_id inválido: {self.client_id!r}")


def list_client_ids() -> list[str]:
    """Clientes con datos en data_dir, en orden alfabético.

    Solo carpetas cuyo nombre es un client_id válido: una carpeta suelta como
    "Borradores" no aparece como cliente.
    """
    data_dir = get_settings().data_dir
    if not data_dir.is_dir():
        return []
    return sorted(
        path.name for path in data_dir.iterdir() if path.is_dir() and _CLIENT_ID_PATTERN.fullmatch(path.name)
    )


def client_data_dir(tenant: TenantContext) -> Path:
    """Devuelve la carpeta de datos del cliente dentro de data_dir."""
    path = get_settings().data_dir / tenant.client_id
    if not path.is_dir():
        raise UnknownClientError(f"no hay datos para el cliente {tenant.client_id!r}")
    return path
