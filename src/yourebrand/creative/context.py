"""Contexto del cliente para generar ideas: información de marca y posts anteriores.

Por ahora sale de archivos en `data/<client_id>/` y va entero al prompt. El resto
del sistema solo conoce `load_client_context`: cuando el material pase a Supabase
o a Qdrant cambia este módulo, no quienes lo usan.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from yourebrand.core.tenancy import TenantContext, client_data_dir
from yourebrand.core.trace import emit

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


def load_brand(tenant: TenantContext) -> str:
    """Información de marca del cliente. Falla si no está cargada: sin marca no se puede trabajar."""
    # La carpeta se resuelve siempre a partir del tenant validado: es lo que
    # impide leer material de otro cliente.
    brand = _read_text(client_data_dir(tenant) / BRAND_FILENAME)
    if brand is None:
        raise MissingBrandError(f"el cliente {tenant.client_id!r} no tiene {BRAND_FILENAME}")
    return brand


def load_previous_posts(tenant: TenantContext) -> str | None:
    """Posts anteriores del cliente, o None si no hay: una marca nueva puede no tenerlos."""
    return _read_text(client_data_dir(tenant) / POSTS_FILENAME)


def load_client_context(tenant: TenantContext) -> ClientContext:
    """Lee la información de marca y los posts anteriores del cliente."""
    brand = load_brand(tenant)
    posts = load_previous_posts(tenant)

    # Se registra el tamaño, nunca el contenido (datos del cliente). El tamaño es
    # la señal para saber cuándo el material deja de entrar entero en el prompt.
    emit(
        logger,
        "client_context",
        client_id=tenant.client_id,
        brand_chars=len(brand),
        posts_chars=len(posts or ""),
        has_posts=posts is not None,
    )
    return ClientContext(client_id=tenant.client_id, brand=brand, posts=posts)
