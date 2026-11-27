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
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, field_validator

from yourebrand.core.config import get_settings
from yourebrand.core.errors import BudgetExceededError, LLMError
from yourebrand.core.reliability import validate_user_text
from yourebrand.core.tools import (
    ApprovalPreviewError,
    ExecutionContext,
    Tool,
    ToolRegistry,
    ToolResult,
    model_usage_metadata,
)
from yourebrand.core.trace import emit
from yourebrand.creative.context import MissingBrandError, load_brand, load_previous_posts
from yourebrand.creative.ideas import (
    MAX_IDEAS,
    NotEnoughIdeasError,
    generate_ideas_with_usage,
    refine_idea_with_usage,
)
from yourebrand.creative.schemas import ContentType, GeneratedIdea
from yourebrand.creative.similarity import check_idea_similarity
from yourebrand.core.media import MediaError
from yourebrand.core.storage import PLAN_ID_PATTERN, VIDEO_ID_PATTERN, UnknownPlanError, UnknownVideoError
from yourebrand.core.transcription import TranscriptionFailedError, TranscriptionUnavailableError
from yourebrand.core.video_prep import prepare_video
from yourebrand.creative.skills import Skill, available_skills, get_skill
from yourebrand.creative.video_edit import InvalidPlanError, plan_video_edit
from yourebrand.creative.video_render import PlanNotApprovedError, load_plan, start_render, store_plan

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


def _unknown_ideas_result(missing: list[str], known: dict[str, GeneratedIdea]) -> ToolResult:
    """Error para ids que no están en la sesión: le dice al modelo cuáles sí existen."""
    return ToolResult(
        ok=False,
        content=f"No existen en esta conversación: {', '.join(missing)}. "
        f"Ideas disponibles: {', '.join(known) or 'ninguna'}. Si recién las generaste, "
        "llamá a esta tool en el paso siguiente.",
        metadata={"error_type": "unknown_idea"},
    )


MAX_REFINE_COMMENT_CHARS = 1000


class RefineIdeaArgs(BaseModel):
    """El id y no el texto de la idea: se ajusta la idea real, no una copia reescrita por el modelo."""

    idea_id: str = Field(description="Id de la idea de esta conversación que hay que ajustar.")
    comment: str = Field(
        max_length=MAX_REFINE_COMMENT_CHARS,
        description="Qué quiere cambiar la persona, con sus palabras. No agregues cambios que no pidió.",
    )

    @field_validator("comment")
    @classmethod
    def _validate_comment(cls, value: str) -> str:
        # Lo escribió la persona y va a otro modelo: misma validación que un mensaje.
        return validate_user_text(value, max_len=MAX_REFINE_COMMENT_CHARS)


def refine_idea_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Ajusta una idea de la sesión y la devuelve como idea nueva, con otro id."""
    known = {idea.id: idea for idea in ctx.session_ideas}
    original = known.get(args["idea_id"])
    if original is None:
        return _unknown_ideas_result([args["idea_id"]], known)

    try:
        refinement = refine_idea_with_usage(ctx.tenant, original, args["comment"], run_id=ctx.run_id)
    except MissingBrandError:
        return ToolResult(
            ok=False,
            content="La marca no tiene información cargada: no se puede ajustar la idea. Avisale a la persona.",
            metadata={"error_type": "missing_brand"},
        )
    except LLMError as exc:
        return ToolResult(
            ok=False,
            content=f"No se pudo ajustar la idea ({type(exc).__name__}). "
            "Podés intentar de nuevo una vez o avisarle a la persona.",
            metadata={"error_type": type(exc).__name__},
        )

    refined = GeneratedIdea(
        id=_new_idea_id(),
        content_type=original.content_type,
        refined_from=original.id,
        **refinement.idea.model_dump(),
    )
    payload = refined.model_dump()
    return ToolResult(
        ok=True,
        content=json.dumps(payload, ensure_ascii=False),
        # Va al estado como cualquier idea generada: así se puede volver a
        # ajustar, verificar o guardar por su id.
        metadata={
            IDEAS_METADATA_KEY: [payload],
            "count": 1,
            **model_usage_metadata(refinement.usage.input_tokens, refinement.usage.output_tokens),
        },
    )


class CheckSimilarityArgs(BaseModel):
    """Ids y no texto: así se verifica la idea real, no una versión reescrita por el modelo."""

    idea_ids: list[str] = Field(
        min_length=1, max_length=MAX_IDEAS, description="Ids de las ideas de esta conversación a verificar."
    )


def check_similarity_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Compara ideas de la sesión contra lo que el cliente ya publicó."""
    idea_ids = list(dict.fromkeys(args["idea_ids"]))  # sin repetidos, en el mismo orden
    known = {idea.id: idea for idea in ctx.session_ideas}
    missing = [idea_id for idea_id in idea_ids if idea_id not in known]
    if missing:
        return _unknown_ideas_result(missing, known)

    try:
        check = check_idea_similarity(ctx.tenant, [known[idea_id] for idea_id in idea_ids], run_id=ctx.run_id)
    except LLMError as exc:
        metadata: dict[str, Any] = {"error_type": type(exc).__name__}
        usage = getattr(exc, "usage", None)
        if usage is not None:
            metadata |= model_usage_metadata(usage.input_tokens, usage.output_tokens)
        return ToolResult(
            ok=False,
            content=f"No se pudo verificar la similitud ({type(exc).__name__}). No presentes las ideas "
            "como verificadas: podés intentar una vez más o avisarle a la persona.",
            metadata=metadata,
        )

    if check.usage is None:
        # No es "distinta": no se comparó contra nada y el modelo tiene que saberlo.
        return ToolResult(
            ok=True,
            content=json.dumps({"checked": False, "message": NO_POSTS_MESSAGE}, ensure_ascii=False),
            metadata={"checked": False},
        )

    titles = {reference.id: reference.title for reference in check.references}
    results = [
        {
            "idea_id": verdict.idea_id,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
            "similar_to": [{"id": ref_id, "title": titles[ref_id]} for ref_id in verdict.reference_ids],
        }
        for verdict in check.verdicts
    ]
    emit(logger, "similarity_checked", run_id=ctx.run_id, client_id=ctx.tenant.client_id, ideas=len(results),
         flagged=sum(item["verdict"] != "distinta" for item in results))
    return ToolResult(
        ok=True,
        content=json.dumps({"checked": True, "results": results}, ensure_ascii=False),
        metadata={"checked": True, **model_usage_metadata(check.usage.input_tokens, check.usage.output_tokens)},
    )


# Clave de la metadata donde plan_video_edit deja el plan completo, para la
# aprobación y el render. Al modelo le vuelve solo un resumen.
VIDEO_PLAN_METADATA_KEY = "video_plan"
MAX_VIDEO_NOTES_CHARS = 1000
# Preparar (Whisper en CPU) puede tardar minutos con un video de 2 minutos.
VIDEO_PREP_TIMEOUT_S = 900


class PlanVideoEditArgs(BaseModel):
    video_id: str = Field(
        pattern=VIDEO_ID_PATTERN.pattern,
        description="Id del video adjunto que hay que editar, tal como figura en el mensaje (vid-...).",
    )
    notes: str | None = Field(
        default=None,
        max_length=MAX_VIDEO_NOTES_CHARS,
        description="Indicaciones de la persona para esta edición, si dio alguna: tipo de video, red, "
        "qué destacar o qué sacar. Omitilo si no dio ninguna.",
    )

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, value: str | None) -> str | None:
        return None if value is None else validate_user_text(value, max_len=MAX_VIDEO_NOTES_CHARS)


def _plan_summary(video_id: str, plan: Any, plan_id: str | None = None) -> dict[str, Any]:
    """Lo que vuelve al historial: lo necesario para presentar el plan, sin el detalle del render."""
    return {
        **({"plan_id": plan_id} if plan_id else {}),
        "video_id": video_id,
        "tipo": plan.video_type,
        "gancho": plan.hook,
        "duracion_estimada_s": plan.duration_s,
        "tramos": [
            {"de_s": seg.start_s, "a_s": seg.end_s, "para": seg.purpose, "b_roll": len(seg.b_roll)}
            for seg in plan.segments
        ],
        "subtitulos": plan.subtitle_style,
        "textos_en_pantalla": [t.text for t in plan.on_screen_texts],
        "musica": plan.music,
        "cierre": plan.call_to_action,
        "red": plan.target_network,
        "advertencias": plan.warnings,
    }


def plan_video_edit_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Prepara el video del cliente de la sesión y le pide el plan de edición a Claude."""
    video_id = args["video_id"]
    try:
        prep = prepare_video(ctx.tenant, video_id)
    except UnknownVideoError:
        # Mismo mensaje si no existe o es de otro cliente: no se revela cuál.
        return ToolResult(ok=False, content=f"No hay un video {video_id} en esta sesión. Revisá el id.",
                          metadata={"error_type": "unknown_video"})
    except (MediaError, TranscriptionUnavailableError, TranscriptionFailedError) as exc:
        return ToolResult(
            ok=False,
            content=f"No se pudo preparar el video ({exc}). Avisale a la persona: no es un problema del pedido.",
            metadata={"error_type": type(exc).__name__},
        )
    try:
        generation = plan_video_edit(ctx.tenant, prep, run_id=ctx.run_id, notes=args.get("notes"))
    except MissingBrandError:
        return ToolResult(ok=False, content="La marca no tiene información cargada: no se puede editar.",
                          metadata={"error_type": "missing_brand"})
    except BudgetExceededError:
        return ToolResult(
            ok=False,
            content="El video es demasiado largo para armar el plan dentro del tope de gasto configurado, "
            "y no se llamó al modelo. No lo reintentes: avisale a la persona (puede subir un video más corto).",
            metadata={"error_type": "BudgetExceededError"},
        )
    except LLMError as exc:
        metadata: dict[str, Any] = {"error_type": type(exc).__name__}
        if isinstance(exc, InvalidPlanError):
            metadata |= model_usage_metadata(exc.usage.input_tokens, exc.usage.output_tokens)
        return ToolResult(
            ok=False,
            content=f"No se pudo armar el plan de edición ({type(exc).__name__}). "
            "Podés intentar de nuevo una vez o avisarle a la persona.",
            metadata=metadata,
        )

    plan = generation.plan
    # El plan queda en disco con un id: render_video recibe solo ese id, así el
    # modelo no puede reescribir el plan entre la aprobación y el render.
    stored = store_plan(ctx.tenant, video_id, plan, run_id=ctx.run_id)
    emit(logger, "plan_stored", run_id=ctx.run_id, client_id=ctx.tenant.client_id, plan_id=stored.plan_id,
         video_id=video_id, segments=len(plan.segments))
    return ToolResult(
        ok=True,
        content=json.dumps(_plan_summary(video_id, plan, stored.plan_id), ensure_ascii=False),
        metadata={
            "plan_id": stored.plan_id,
            VIDEO_PLAN_METADATA_KEY: {"video_id": video_id, **plan.model_dump()},
            "frames": len(prep.frames),
            "words": len(prep.transcript.words),
            **model_usage_metadata(generation.usage.input_tokens, generation.usage.output_tokens),
        },
    )


# Clave de la metadata donde render_video deja el estado inicial del render,
# para que la pantalla lo siga. El modelo ve solo el id y las advertencias.
RENDER_METADATA_KEY = "render"


class RenderVideoArgs(BaseModel):
    plan_id: str = Field(
        pattern=PLAN_ID_PATTERN.pattern,
        description="Id del plan de edición que devolvió plan_video_edit (plan-...).",
    )


def render_video_preview(args: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Lo que ve la persona antes de aprobar: el plan guardado, no lo que diga el modelo de él."""
    try:
        stored = load_plan(ctx.tenant, args["plan_id"])
    except UnknownPlanError:
        raise ApprovalPreviewError(f"no hay un plan {args['plan_id']} en esta sesión") from None
    return _plan_summary(stored.video_id, stored.plan, stored.plan_id)


def render_video_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Arranca el render de un plan aprobado. No espera: el render sigue en segundo plano."""
    try:
        # ctx.approved lo pone el grafo después de la pausa; start_render lo vuelve a controlar.
        status = start_render(ctx.tenant, args["plan_id"], run_id=ctx.run_id, approved=ctx.approved)
    except UnknownPlanError:
        return ToolResult(ok=False, content=f"No hay un plan {args['plan_id']} en esta sesión. Revisá el id.",
                          metadata={"error_type": "unknown_plan"})
    except PlanNotApprovedError:
        return ToolResult(ok=False, content="El plan no está aprobado: no se renderizó.",
                          metadata={"error_type": "approval_required"})
    return ToolResult(
        ok=True,
        content=json.dumps(
            {
                "render_id": status.render_id,
                "estado": "en curso",
                "advertencias": status.warnings,
            },
            ensure_ascii=False,
        ),
        metadata={"render_id": status.render_id, RENDER_METADATA_KEY: status.model_dump(mode="json")},
    )


def load_skill_tool(args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
    """Criterio completo de una skill. El nombre ya viene validado contra las skills que existen."""
    skill = get_skill(args["name"])
    emit(logger, "skill_loaded", run_id=ctx.run_id, client_id=ctx.tenant.client_id, skill=skill.name,
         version=skill.version, status=skill.status, chars=len(skill.body))
    # Entre etiquetas, como el material del cliente. Una skill en borrador lo
    # dice, para que el modelo se lo aclare a la persona si lo usa.
    draft_note = (
        "Esta skill es un borrador: todavía no la validó el equipo de la agencia.\n\n"
        if skill.status == "borrador" else ""
    )
    return ToolResult(
        ok=True,
        content=f'<skill nombre="{skill.name}">\n{draft_note}{skill.body}\n</skill>',
        metadata={"skill": skill.name, "version": skill.version, "status": skill.status},
    )


def _load_skill_args_model(skills: list[Skill]) -> type[BaseModel]:
    """Argumentos de load_skill, con los nombres posibles como enum.

    Se arma con las skills que existen: el modelo ve la lista cerrada en el
    schema y un nombre inventado falla en la validación, sin llegar al handler.
    """
    names = tuple(skill.name for skill in skills)
    return create_model(
        "LoadSkillArgs",
        name=(Literal[names], Field(description="Nombre de la skill que querés cargar.")),
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


def build_creative_registry(
    *,
    generation_timeout_s: float | None = None,
    video_edit: bool | None = None,
    video_edit_timeout_s: float | None = None,
    video_edit_max_calls: int | None = None,
) -> ToolRegistry:
    """Registry con todas las tools del Agente Creativo.

    `generation_timeout_s` se calcula de la configuración si no se pasa, y
    `video_edit` (si se registra plan_video_edit) sale de MODEL_VIDEO_EDIT; los
    tests los pasan para no depender del `.env`.
    """
    if generation_timeout_s is None:
        generation_timeout_s = _generation_timeout_s()
    if video_edit is None:
        video_edit = get_settings().model_video_edit is not None
    if video_edit:
        if video_edit_timeout_s is None or video_edit_max_calls is None:
            settings = get_settings()
            video_edit_timeout_s = video_edit_timeout_s or settings.video_edit_timeout_seconds + 30
            video_edit_max_calls = video_edit_max_calls or settings.video_edit_max_calls_per_turn
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
    registry.register(
        Tool(
            name="refine_idea",
            description=(
                "Ajusta una idea ya generada en esta conversación según un comentario de la persona y "
                "devuelve una versión nueva, con otro id. La original se conserva, para que la persona "
                "elija. Usala cuando pide cambiar algo de una idea concreta, o para alejar de lo ya "
                "publicado una idea que check_similarity marcó como repetida. Si no queda claro a qué "
                "idea se refiere, preguntale. Presentá la versión nueva con su id, sin reescribirla."
            ),
            kind="compute",
            fn=refine_idea_tool,
            args_model=RefineIdeaArgs,
            timeout_s=generation_timeout_s,
            retries=0,
        )
    )
    registry.register(
        Tool(
            name="check_similarity",
            description=(
                "Compara ideas ya generadas en esta conversación contra lo que la marca publicó y dice, "
                "por idea, si es 'repetida', 'parecida' o 'distinta', a qué post se parece y por qué. "
                "Usala antes de presentar ideas cuando el pedido o alguna idea trate un tema, producto, "
                "formato o serie que la marca ya publicó, o cuando la persona pida evitar repetir. Usala "
                "aunque te parezca que la idea ya se diferencia: notar el parecido no reemplaza la "
                "verificación, porque la persona necesita saber a qué post se parece y por qué. Ante la "
                "duda, verificá: es una llamada rápida. Solo no hace falta para ideas sin relación con "
                "nada publicado. Si es repetida, no la presentes así: ajustala con refine_idea o generá "
                "otra, una sola vez. Si es parecida, mostrala y avisá a qué post se parece: decide la persona."
            ),
            kind="compute",
            fn=check_similarity_tool,
            args_model=CheckSimilarityArgs,
            timeout_s=generation_timeout_s,
            retries=0,
            max_calls_per_run=3,  # verificar, corregir y volver a verificar, sin bucles
        )
    )
    if video_edit:
        registry.register(
            Tool(
                name="plan_video_edit",
                description=(
                    "Arma el plan de edición de un video UGC adjunto: tipo de video, gancho, tramos con sus "
                    "tiempos, b-roll, subtítulos, textos, música, cierre y advertencias. No edita el video: "
                    "devuelve un plan con su plan_id. Usala cuando la persona adjuntó un video y pide "
                    "editarlo. Si no hay video adjunto, pedile que lo suba. Puede tardar unos minutos. "
                    "Con el plan listo, presentaselo resumido con las advertencias y llamá enseguida a "
                    "render_video con ese plan_id: la persona lo aprueba o lo rechaza ahí. Si lo rechaza "
                    "con un comentario, volvé a llamar a esta tool con ese comentario en `notes`."
                ),
                kind="compute",
                fn=plan_video_edit_tool,
                args_model=PlanVideoEditArgs,
                timeout_s=VIDEO_PREP_TIMEOUT_S + video_edit_timeout_s,
                retries=0,
                # Opus es la llamada cara: un bucle del modelo no puede multiplicarla.
                max_calls_per_run=video_edit_max_calls,
            )
        )
        registry.register(
            Tool(
                name="render_video",
                description=(
                    "Arma el video editado a partir de un plan de plan_video_edit. Antes de ejecutarse, la "
                    "app pausa y le muestra el plan a la persona para que lo apruebe o lo rechace: no le "
                    "preguntes vos si lo aprueba, llamala directamente con el plan_id. Si se aprueba, "
                    "devuelve el id del render: el video tarda unos minutos y aparece en la pantalla "
                    "cuando termina, avisale. Si se rechaza, no se renderiza y recibís su comentario."
                ),
                # local_write: deja un archivo en la máquina, no toca sistemas externos.
                kind="local_write",
                fn=render_video_tool,
                args_model=RenderVideoArgs,
                timeout_s=10,  # solo encola: el render corre en segundo plano
                retries=0,
                idempotent=False,  # repetirlo haría dos renders
                requires_approval=True,
                approval_preview=render_video_preview,
            )
        )
    skills =list(available_skills().values())
    if skills:
        registry.register(
            Tool(
                name="load_skill",
                description=(
                    "Carga el criterio completo de una skill de la agencia: cómo se piensa o se hace un "
                    "tipo de trabajo. Usala antes de una tarea que coincide con una skill del índice de tu "
                    "prompt, si todavía no la cargaste en esta conversación."
                ),
                kind="read",
                fn=load_skill_tool,
                args_model=_load_skill_args_model(skills),
            )
        )
    return registry
