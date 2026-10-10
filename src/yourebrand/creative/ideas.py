"""Generación de ideas de contenido para un cliente.

Une las piezas del Agente Creativo: carga el contexto del cliente, arma los
prompts y le pide al modelo una lista de ideas con forma validada.
"""

import logging
from typing import get_args

from yourebrand.core.llm import LLMOutputError, complete_structured, new_run_id
from yourebrand.core.tenancy import TenantContext
from yourebrand.creative.context import load_client_context
from yourebrand.creative.prompts import build_idea_prompt, build_system_prompt
from yourebrand.creative.schemas import ContentType, Idea, IdeaList

logger = logging.getLogger(__name__)

MAX_IDEAS = 20


class NotEnoughIdeasError(LLMOutputError):
    """El modelo devolvió menos ideas de las pedidas."""


def generate_ideas(
    tenant: TenantContext,
    content_type: ContentType,
    count: int,
    *,
    run_id: str | None = None,
) -> list[Idea]:
    """Devuelve `count` ideas de contenido del tipo pedido para el cliente."""
    # Se valida antes de leer datos o llamar al modelo: un pedido mal armado
    # no tiene que costar una llamada.
    if content_type not in get_args(ContentType):
        raise ValueError(f"content_type inválido: {content_type!r}")
    if not 1 <= count <= MAX_IDEAS:
        raise ValueError(f"count debe estar entre 1 y {MAX_IDEAS}, llegó {count}")

    run_id = run_id or new_run_id()
    context = load_client_context(tenant)

    result = complete_structured(
        system=build_system_prompt(content_type),
        prompt=build_idea_prompt(context, content_type, count),
        schema=IdeaList,
        tenant=tenant,
        run_id=run_id,
    )
    ideas = result.data.ideas

    # La cantidad no se puede fijar en el schema, así que se controla acá.
    if len(ideas) < count:
        raise NotEnoughIdeasError(f"run_id={run_id}: se pidieron {count} ideas y llegaron {len(ideas)}")
    if len(ideas) > count:
        logger.warning("generate_ideas run_id=%s llegaron %d ideas, se devuelven %d", run_id, len(ideas), count)
    return ideas[:count]
