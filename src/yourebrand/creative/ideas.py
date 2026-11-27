"""Generación de ideas de contenido para un cliente.

Une las piezas del Agente Creativo: carga el contexto del cliente, arma los
prompts y le pide al modelo una lista de ideas con forma validada.
"""

import logging
from dataclasses import dataclass
from typing import get_args

from yourebrand.core.errors import LLMOutputError
from yourebrand.core.llm import LLMResponse, complete_structured, new_run_id
from yourebrand.core.tenancy import TenantContext
from yourebrand.creative.context import load_client_context
from yourebrand.creative.prompts import build_idea_prompt, build_refine_prompt, build_system_prompt
from yourebrand.creative.schemas import ContentType, GeneratedIdea, Idea, IdeaList

logger = logging.getLogger(__name__)

MAX_IDEAS = 20


class NotEnoughIdeasError(LLMOutputError):
    """El modelo devolvió menos ideas de las pedidas.

    Lleva el uso de la llamada: los tokens se gastaron igual y la traza los cuenta.
    """

    def __init__(self, message: str, *, usage: LLMResponse | None = None) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class IdeaGeneration:
    """Ideas generadas más el uso de la llamada al modelo (tokens, latencia, modelo)."""

    ideas: list[Idea]
    usage: LLMResponse


def generate_ideas_with_usage(
    tenant: TenantContext,
    content_type: ContentType,
    count: int,
    *,
    brief: str | None = None,
    run_id: str | None = None,
) -> IdeaGeneration:
    """Como `generate_ideas`, pero devuelve también el uso de la llamada.

    Lo usa la tool del agente para que la traza del turno sume los tokens de
    esta llamada, que no pasa por el nodo del modelo.
    """
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
        prompt=build_idea_prompt(context, content_type, count, brief),
        schema=IdeaList,
        tenant=tenant,
        run_id=run_id,
    )
    ideas = result.data.ideas

    # La cantidad no se puede fijar en el schema, así que se controla acá.
    if len(ideas) < count:
        raise NotEnoughIdeasError(
            f"run_id={run_id}: se pidieron {count} ideas y llegaron {len(ideas)}", usage=result.response
        )
    if len(ideas) > count:
        logger.warning("generate_ideas run_id=%s llegaron %d ideas, se devuelven %d", run_id, len(ideas), count)
    return IdeaGeneration(ideas=ideas[:count], usage=result.response)


def generate_ideas(
    tenant: TenantContext,
    content_type: ContentType,
    count: int,
    *,
    brief: str | None = None,
    run_id: str | None = None,
) -> list[Idea]:
    """Devuelve `count` ideas de contenido del tipo pedido para el cliente.

    `brief` es el foco opcional del pedido; quien llama ya lo validó.
    """
    return generate_ideas_with_usage(tenant, content_type, count, brief=brief, run_id=run_id).ideas


@dataclass(frozen=True)
class IdeaRefinement:
    """Idea ajustada más el uso de la llamada al modelo."""

    idea: Idea
    usage: LLMResponse


def refine_idea_with_usage(
    tenant: TenantContext, idea: GeneratedIdea, comment: str, *, run_id: str | None = None
) -> IdeaRefinement:
    """Ajusta `idea` según `comment`, con el material del cliente y la skill de su tipo.

    `comment` ya viene validado por quien llama. Devuelve la idea sin id: el id
    nuevo lo pone la tool, igual que en generate_ideas.
    """
    run_id = run_id or new_run_id()
    context = load_client_context(tenant)
    result = complete_structured(
        system=build_system_prompt(idea.content_type),
        prompt=build_refine_prompt(context, idea, comment),
        schema=Idea,
        tenant=tenant,
        run_id=run_id,
    )
    return IdeaRefinement(idea=result.data, usage=result.response)
