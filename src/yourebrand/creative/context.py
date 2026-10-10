"""Contexto del cliente para generar ideas: información de marca y posts anteriores.

Por ahora sale de archivos en `data/<client_id>/` y va entero al prompt. El resto
del sistema solo conoce `load_client_context`: cuando el material pase a Supabase
o a Qdrant cambia este módulo, no quienes lo usan.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from yourebrand.core.tenancy import TenantContext, client_data_dir

logger = logging.getLogger(__name__)

BRAND_FILENAME = "brand.md"
POSTS_FILENAME = "posts.md"


class MissingBrandError(LookupError):
    """El cliente existe pero no tiene información de marca cargada."""


@dataclass(frozen=True)
class ClientContext:
    """Material de un único cliente. `posts` es None si no hay posts anteriores."""

    client_id: str
    brand: str
    posts: str | None


def _read_text(path: Path) -> str | None:
    """Devuelve el contenido del archivo, o None si no existe o está en blanco."""
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def load_client_context(tenant: TenantContext) -> ClientContext:
    """Lee la información de marca y los posts anteriores del cliente."""
    # La carpeta se resuelve siempre a partir del tenant validado: es lo que
    # impide leer material de otro cliente.
    folder = client_data_dir(tenant)

    brand = _read_text(folder / BRAND_FILENAME)
    if brand is None:
        raise MissingBrandError(f"el cliente {tenant.client_id!r} no tiene {BRAND_FILENAME}")
    posts = _read_text(folder / POSTS_FILENAME)

    # Se registra el tamaño, nunca el contenido (datos del cliente). El tamaño es
    # la señal para saber cuándo el material deja de entrar entero en el prompt.
    logger.info(
        "client_context client_id=%s brand_chars=%d posts_chars=%d",
        tenant.client_id,
        len(brand),
        len(posts or ""),
    )
    return ClientContext(client_id=tenant.client_id, brand=brand, posts=posts)
