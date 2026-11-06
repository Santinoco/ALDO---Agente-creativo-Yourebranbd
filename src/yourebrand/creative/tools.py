"""Tools del Agente Creativo y el registry que las agrupa.

Cada handler recibe el cliente del contexto de ejecución, nunca de los
argumentos del modelo. La lectura de archivos vive en `context.py`: las tools
solo deciden qué le muestran al modelo.

Las descripciones son instrucciones para el modelo: de ellas depende que elija
la tool correcta, así que dicen qué devuelve cada una y cuándo conviene usarla.
"""

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from yourebrand.core.config import get_settings
from yourebrand.core.errors import LLMError
from yourebrand.core.reliability import validate_user_text
from yourebrand.core.tools import ExecutionContext, Tool, ToolRegistry, ToolResult, model_usage_metadata
from yourebrand.core.trace import emit
from yourebrand.creative.context import MissingBrandError, load_brand, load_previous_posts
from yourebrand.creative.ideas import MAX_IDEAS, NotEnoughIdeasError, generate_ideas_with_usage
from yourebrand.creative.schemas import ContentType, GeneratedIdea

logger = logging.getLogger(__name__)

# Sin posts no es un error: una marca nueva todavía no publicó nada. El modelo
# tiene que saberlo para no inventar una línea de feed que no existe.
NO_POSTS_MESSAGE = "Esta marca todavía no tiene posts anteriores cargados."

# Clave de la metadata donde generate_ideas deja las ideas para el estado del
# grafo. La metadata no la ve el modelo: es para la aplicación.
IDEAS_METADATA_KEY = "ideas"
DEFAULT_IDEA_COUNT = 3
# Largo máximo del foco: es un resumen del pedido, no un brief completo.
MAX_BRIEF_CHARS = 1000


class GenerateIdeasArgs(BaseModel):
    """Lo que decide el modelo al pedir ideas. Las descripciones son instrucciones para él."""

    content_type: ContentType = Field(
        description="Tipo de contenido: 'ugc' (piezas protagonizadas por una persona, en video vertical) "
        "o 'design' (piezas gráficas: placas, carruseles, historias). Si el pedido no lo deja claro, "
        "preguntale a la persona antes de llamar a esta tool."
    )
    count: int = Field(default=DEFAULT_IDEA_COUNT, ge=1, le=MAX_IDEAS, description="Cuántas ideas proponer.")
    brief: str | None = Field(
        default=None,
        max_length=MAX_BRIEF_CHARS,
        description="Foco del pedido en pocas palabras, si la persona dio uno: una fecha, un producto, "
        "un objetivo o un público. Omitilo si pidió ideas en general.",
    )

    @field_validator("brief")
    @classmethod
    def _validate_brief(cls, value: str | None) -> str | None:
        # Lo arma el modelo a partir de lo que escribió la persona, y va a otro
        # modelo: pasa por la misma validación que cualquier texto de usuario.
        # InvalidUserInputError es un ValueError: Pydantic lo convierte en un
        # error de validación que vuelve al modelo con el detalle.
        return None if value is None else validate_user_text(value, max_len=MAX_BRIEF_CHARS)


def _new_idea_id() -> str:
    """Id corto y único: no depende del estado, así la tool funciona fuera de una conversación."""
    return f"idea-{uuid.uuid4().hex[:6]}"


def get_brand_info(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Información de marca del cliente de la sesión."""
    try:
        brand = load_brand(ctx.tenant)
    except MissingBrandError:
        # Error esperable y con un mensaje que el modelo puede transmitir.
        return ToolResult(
            ok=False,
            content="La marca no tiene información cargada. Avisale a la persona: sin eso no se pueden proponer ideas.",
            metadata={"error_type": "missing_brand"},
        )
    emit(logger, "brand_loaded", run_id=ctx.run_id, client_id=ctx.tenant.client_id, chars=len(brand))
    # Entre etiquetas, como en prompts.py: marca dónde empieza y termina el
    # material, que es referencia y no instrucciones.
    return ToolResult(ok=True, content=f"<marca>\n{brand}\n</marca>", metadata={"chars": len(brand)})


def get_previous_posts(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Posts anteriores del cliente de la sesión."""
    posts = load_previous_posts(ctx.tenant)
    emit(
        logger,
        "posts_loaded",
        run_id=ctx.run_id,
        client_id=ctx.tenant.client_id,
        chars=len(posts or ""),
        has_posts=posts is not None,
    )
    if posts is None:
        return ToolResult(ok=True, content=NO_POSTS_MESSAGE, metadata={"chars": 0, "has_posts": False})
    return ToolResult(
        ok=True,
        content=f"<posts_anteriores>\n{posts}\n</posts_anteriores>",
        metadata={"chars": len(posts), "has_posts": True},
    )


def generate_ideas_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Genera ideas para el cliente de la sesión, apoyadas en su marca y sus posts.

    Carga el material por su cuenta (vía context.py) y no depende de que el
    agente lo haya consultado antes: así se puede usar fuera de una conversación.
    """
    try:
        generation = generate_ideas_with_usage(
            ctx.tenant, args["content_type"], args["count"], brief=args["brief"], run_id=ctx.run_id
        )
    except MissingBrandError:
        return ToolResult(
            ok=False,
            content="La marca no tiene información cargada: no se pueden proponer ideas. Avisale a la persona.",
            metadata={"error_type": "missing_brand"},
        )
    except LLMError as exc:
        # Incluye menos ideas de las pedidas, salida inválida, rechazo y fallas
        # del proveedor después de los reintentos de core/llm.py. Se informa el
        # tipo, no el detalle: el modelo decide si reintenta o avisa.
        metadata: dict[str, Any] = {"error_type": type(exc).__name__}
        # Si la llamada llegó a responder, sus tokens se gastaron: van a la traza.
        if isinstance(exc, NotEnoughIdeasError) and exc.usage is not None:
            metadata |= model_usage_metadata(exc.usage.input_tokens, exc.usage.output_tokens)
        return ToolResult(
            ok=False,
            content=f"No se pudieron generar las ideas ({type(exc).__name__}). "
            "Podés intentar de nuevo una vez o avisarle a la persona.",
            metadata=metadata,
        )

    generated = [
        GeneratedIdea(id=_new_idea_id(), content_type=args["content_type"], **idea.model_dump())
        for idea in generation.ideas
    ]
    payload = [idea.model_dump() for idea in generated]
    return ToolResult(
        ok=True,
        # ensure_ascii=False: las tildes viajan como texto y no como á,
        # que suma tokens y es más difícil de leer en la traza.
        content=json.dumps(payload, ensure_ascii=False),
        metadata={
            IDEAS_METADATA_KEY: payload,
            "count": len(payload),
            **model_usage_metadata(generation.usage.input_tokens, generation.usage.output_tokens),
        },
    )


def _generation_timeout_s() -> float:
    """Plazo de generate_ideas: el peor caso de la llamada al modelo con sus reintentos.

    Si fuera menor, la tool cortaría mientras core/llm.py todavía reintenta. Es
    una red de seguridad: el plazo real de cada intento lo pone el proveedor.
    """
    settings = get_settings()
    attempts = settings.llm_max_retries + 1
    max_backoff_s = 30.0  # tope de espera entre reintentos en call_with_retry
    return settings.llm_timeout_seconds * attempts + max_backoff_s * settings.llm_max_retries + 5


def build_creative_registry(*, generation_timeout_s: float | None = None) -> ToolRegistry:
    """Registry con todas las tools del Agente Creativo.

    `generation_timeout_s` se calcula de la configuración si no se pasa; los
    tests lo pasan para no depender del `.env`.
    """
    if generation_timeout_s is None:
        generation_timeout_s = _generation_timeout_s()
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_brand_info",
            description=(
                "Devuelve la información de la marca del cliente de esta sesión: qué hace, a quién le "
                "habla, tono, pilares de contenido, identidad visual y qué evitar. Usala antes de "
                "proponer o ajustar ideas si todavía no la consultaste en esta conversación, y para "
                "responder preguntas sobre la marca. No inventes datos de la marca: consultalos acá."
            ),
            kind="read",
            fn=get_brand_info,
        )
    )
    registry.register(
        Tool(
            name="get_previous_posts",
            description=(
                "Devuelve los posts que la marca ya publicó: el copy y una descripción breve de cada "
                "pieza. Usala para que las ideas nuevas sigan la línea del feed y para revisar qué ya "
                "se hizo antes de proponer. Si la marca no tiene posts, lo indica."
            ),
            kind="read",
            fn=get_previous_posts,
        )
    )
    registry.register(
        Tool(
            name="generate_ideas",
            description=(
                "Genera ideas de contenido nuevas para esta marca, apoyadas en su información y en sus "
                "posts anteriores (la tool los lee sola). Devuelve una lista de ideas, cada una con un id "
                "para referirse a ella después. Usala cuando la persona pide ideas y ya sabés el tipo de "
                "contenido. Presentale las ideas con su id, sin reescribirlas."
            ),
            # compute: llama a un modelo pero no tiene efectos fuera del sistema.
            kind="compute",
            fn=generate_ideas_tool,
            args_model=GenerateIdeasArgs,
            timeout_s=generation_timeout_s,
            # Sin reintentos propios: los de la llamada al modelo ya los hace core/llm.py.
            retries=0,
        )
    )
    return registry
