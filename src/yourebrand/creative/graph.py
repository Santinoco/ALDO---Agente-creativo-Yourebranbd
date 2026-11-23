"""Grafo del Agente Creativo: el modelo decide en cada paso si usa tools o responde.

Cuatro nodos propios, sin los prearmados de LangGraph:

    START → call_model ─┬─ pidió tools y quedan pasos ──────────→ run_tools → call_model
                        ├─ pidió una tool que requiere aprobación → await_approval → run_tools
                        ├─ respondió solo texto ─────────────────→ END
                        └─ pidió tools sin pasos ────────────────→ step_limit → END

Pedir aclaraciones no es un nodo: es el camino en el que el modelo responde con
texto sin pedir ninguna tool.

Human-in-the-loop: `await_approval` pausa el grafo con `interrupt` y el turno
termina con la aprobación pendiente. `resume_turn` lo retoma con la decisión de
la persona. Es un nodo aparte y no un paso dentro de `run_tools` porque, al
retomar, LangGraph vuelve a ejecutar el nodo pausado desde el principio: si la
pausa estuviera en `run_tools`, las tools que corrieron antes se repetirían.

El cliente se fija al abrir la sesión y vive en el estado. Las tools lo reciben
del contexto que arma `run_tools`, nunca del modelo.
"""

import logging
import operator
import uuid
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from yourebrand.core import llm
from yourebrand.core.errors import LLMError
from yourebrand.core.providers import Message
from yourebrand.core.reliability import validate_user_text
from yourebrand.core.storage import VideoRef, get_video
from yourebrand.core.tenancy import TenantContext, client_data_dir
from yourebrand.core.tools import (
    MODEL_INPUT_TOKENS_KEY,
    MODEL_OUTPUT_TOKENS_KEY,
    ApprovalPreviewError,
    ExecutionContext,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from yourebrand.core.trace import TraceEvent, collect, emit
from yourebrand.creative.prompts import build_agent_system_prompt
from yourebrand.creative.schemas import GeneratedIdea
from yourebrand.creative.tools import IDEAS_METADATA_KEY, RENDER_METADATA_KEY

logger = logging.getLogger(__name__)

# Llamadas al modelo por turno del usuario. Cada vuelta del loop es una llamada
# más: sin tope, un modelo que pide tools sin parar giraría indefinidamente.
DEFAULT_MAX_STEPS = 6
# Reintentos cuando el pedido de tool llega mal armado. Con Gemini, repetir la
# misma llamada suele alcanzar; más de uno ya indica un problema que no se va solo.
MALFORMED_RETRIES = 1

STEP_LIMIT_REPLY = (
    "No llegué a terminar este pedido en los pasos que tengo por turno. "
    "Probá dividirlo en pedidos más chicos o decime por dónde sigo."
)
MALFORMED_REPLY = "No pude armar bien el pedido a mis herramientas. Probá reformular lo que necesitás."
MODEL_ERROR_REPLY = "No pude responder por un problema con el modelo. Probá de nuevo en un momento."
MAX_APPROVAL_COMMENT_CHARS = 1000


class ApprovalDecision(BaseModel):
    """Lo que decidió la persona sobre un pedido que requería aprobación.

    `error` lo pone el sistema cuando no hubo nada que aprobar (por ejemplo, el
    plan pedido no existe): ese pedido no pausa y vuelve al modelo como error.
    """

    approved: bool
    comment: str | None = None
    error: str | None = None


class StepTrace(BaseModel):
    """Un paso del grafo, para la traza del turno y el set de evaluación.

    No guarda argumentos ni contenido: solo nombres, resultados y uso.
    """

    run_id: str
    kind: Literal["model", "tool", "step_limit", "approval"]
    # Paso "model": qué tools pidió el modelo (vacío si respondió con texto).
    # Paso "approval": qué tools se sometieron a la decisión de la persona.
    tools_requested: list[str] = Field(default_factory=list)
    # Paso "tool": qué tool corrió y cómo terminó. Paso "approval": si se aprobó.
    tool_name: str | None = None
    ok: bool | None = None
    error_type: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class AgentState(TypedDict):
    """Estado de una sesión. Lo guarda el checkpointer entre turnos."""

    client_id: str
    run_id: str  # uno por turno del usuario: agrupa las llamadas y tools de ese turno
    # Annotated con operator.add: cada nodo devuelve solo los mensajes nuevos y
    # LangGraph los suma al final. Ningún nodo puede pisar el historial.
    messages: Annotated[list[Message], operator.add]
    steps: int  # llamadas al modelo en el turno actual; se reinicia en cada turno
    trace: Annotated[list[StepTrace], operator.add]
    # Ideas generadas en la sesión, con su id. Las tools que vengan después
    # (refinar, guardar en Notion) las buscan acá por id, sin que el modelo las reescriba.
    ideas: Annotated[list[GeneratedIdea], operator.add]
    # Videos adjuntados en la sesión: solo la referencia (id, duración, tamaño),
    # nunca el archivo, que vive en core/storage.py.
    videos: Annotated[list[VideoRef], operator.add]
    # Decisiones de la última pausa, por id del pedido. Sin reducer: las escribe
    # await_approval y run_tools las vacía al usarlas, así una aprobación no
    # sirve para otro pedido.
    approvals: dict[str, ApprovalDecision]


@dataclass(frozen=True)
class AgentSession:
    """Una conversación: su hilo en el checkpointer y el cliente, fijo desde que se abre."""

    thread_id: str
    client_id: str


@dataclass(frozen=True)
class TurnOutcome:
    """Lo que devuelve un turno: la respuesta para el usuario y su traza."""

    reply: str
    run_id: str
    trace: list[StepTrace]
    ideas: list[GeneratedIdea]  # las generadas en este turno, para mostrarlas aparte del texto
    # Todo lo que pasó en el turno, en orden: llamadas, reintentos, ruteo, tools,
    # material cargado. `trace` es el resumen por paso; `events`, el detalle.
    events: list[TraceEvent] = field(default_factory=list)
    # Si el turno quedó en pausa: lo que la persona tiene que aprobar o rechazar.
    # Se resuelve con resume_turn; mientras tanto la sesión no acepta mensajes.
    pending_approval: list[dict[str, Any]] | None = None
    # Renders que arrancaron en este turno (estado inicial). Siguen en segundo plano.
    renders: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tools_called(self) -> list[str]:
        """Tools que se ejecutaron en el turno, en orden."""
        return [step.tool_name for step in self.trace if step.kind == "tool" and step.tool_name]

    @property
    def input_tokens(self) -> int:
        """Tokens de entrada del turno: los del agente más los de las tools que llaman a un modelo."""
        return sum(step.input_tokens for step in self.trace)

    @property
    def output_tokens(self) -> int:
        """Tokens de salida del turno, con el mismo criterio que `input_tokens`."""
        return sum(step.output_tokens for step in self.trace)


class SessionClientMismatchError(ValueError):
    """La sesión ya pertenece a otro cliente."""


class PendingApprovalError(RuntimeError):
    """La sesión está en pausa esperando una aprobación: primero hay que resolverla."""


class NoPendingApprovalError(RuntimeError):
    """Se quiso resolver una pausa en una sesión que no está en pausa."""


# Clases propias que guarda el estado, incluidas las anidadas (un Message lleva
# ToolCall y ToolResult). Al leer un checkpoint, LangGraph solo reconstruye las
# que están en esta lista: si la base de checkpoints se adulterara, no podría
# hacer que se construya cualquier otra clase de Python. Si el estado suma un
# tipo nuevo, va acá; un test lo controla.
CHECKPOINT_TYPES: tuple[type, ...] = (
    Message, ToolCall, ToolResult, StepTrace, GeneratedIdea, VideoRef, ApprovalDecision,
)


def _rejection_result(tool_name: str, decision: ApprovalDecision) -> ToolResult:
    """Lo que vuelve al modelo cuando un pedido no se ejecutó por la decisión de la persona.

    El pedido queda respondido igual: un historial con un pedido sin resultado
    lo rechaza el proveedor en el turno siguiente.
    """
    if decision.error is not None:
        return ToolResult(ok=False, content=f"No se pidió aprobación porque el pedido no es válido: {decision.error}",
                          metadata={"error_type": "invalid_approval_request"})
    # El comentario va entre etiquetas: es lo que escribió la persona, no instrucciones del sistema.
    comment = (f"\n<comentario_de_la_persona>\n{decision.comment}\n</comentario_de_la_persona>"
               if decision.comment else "")
    advice = ("Corregí el pedido según su comentario y volvé a pedir aprobación." if decision.comment
              else "Preguntale qué quiere cambiar antes de volver a intentarlo.")
    return ToolResult(
        ok=False,
        content=f"La persona no aprobó {tool_name}: no se ejecutó. {advice}{comment}",
        metadata={"error_type": "rejected_by_user"},
    )


def checkpoint_serializer() -> JsonPlusSerializer:
    """Serializador de checkpoints que solo acepta los tipos del estado.

    Es el mismo para el checkpointer en memoria y para el de SQLite.
    """
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(cls.__module__, cls.__name__) for cls in CHECKPOINT_TYPES]
    )


def build_graph(
    registry: ToolRegistry,
    *,
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    checkpointer: Any = None,
):
    """Arma y compila el grafo del agente.

    El registry entra como parámetro para que los tests usen tools de prueba.
    Sin `checkpointer`, la conversación vive en memoria mientras dura el
    proceso; en la iteración 3 se pasa uno de SQLite y no cambia nada más.
    """
    if max_steps < 1:
        raise ValueError("max_steps debe ser al menos 1")
    system = system_prompt if system_prompt is not None else build_agent_system_prompt()

    def call_model(state: AgentState) -> dict[str, Any]:
        tenant = TenantContext(state["client_id"])
        run_id = state["run_id"]
        traces: list[StepTrace] = []
        message: Message | None = None

        # Se intenta una vez más si el pedido de tool llega mal armado. El
        # intento fallido no entra al historial: el modelo no tiene que leerlo.
        for _ in range(1 + MALFORMED_RETRIES):
            try:
                turn = llm.complete_turn(
                    system=system,
                    messages=state["messages"],
                    tools=registry.function_declarations(),
                    tenant=tenant,
                    run_id=run_id,
                )
            except LLMError as exc:
                # core/llm.py ya reintentó y registró la falla. Se responde con
                # un texto fijo para que el historial quede consistente y la
                # sesión siga usable en el próximo turno.
                traces.append(StepTrace(run_id=run_id, kind="model", error_type=type(exc).__name__))
                # El mensaje de core/errors.py dice proveedor, tipo y código
                # HTTP: es lo que explica en la traza por qué falló.
                emit(logger, "model_error", level=logging.ERROR, private={"message": str(exc)},
                     run_id=run_id, client_id=tenant.client_id, error=type(exc).__name__)
                message = Message(role="model", text=MODEL_ERROR_REPLY)
                break

            traces.append(
                StepTrace(
                    run_id=run_id,
                    kind="model",
                    tools_requested=[call.tool_name for call in turn.message.tool_calls],
                    error_type="malformed_tool_call" if turn.malformed_tool_call else None,
                    input_tokens=turn.input_tokens,
                    output_tokens=turn.output_tokens,
                    latency_ms=turn.latency_ms,
                )
            )
            if not turn.malformed_tool_call:
                message = turn.message
                break
            emit(logger, "malformed_tool_call", level=logging.WARNING, run_id=run_id,
                 client_id=tenant.client_id, will_retry=len(traces) <= MALFORMED_RETRIES)

        if message is None:  # se agotaron los intentos con pedidos mal armados
            message = Message(role="model", text=MALFORMED_REPLY)

        return {"messages": [message], "steps": state["steps"] + 1, "trace": traces}

    def run_tools(state: AgentState) -> dict[str, Any]:
        # El contexto sale del estado, no del modelo: así ninguna respuesta del
        # modelo puede hacer que una tool trabaje para otro cliente.
        context = ExecutionContext(tenant=TenantContext(state["client_id"]), run_id=state["run_id"], session_ideas=tuple(state.get("ideas", [])))
        pending = state["messages"][-1].tool_calls
        approvals = state.get("approvals") or {}

        messages: list[Message] = []
        traces: list[StepTrace] = []
        ideas: list[GeneratedIdea] = []
        for call in pending:
            # Los argumentos que eligió el modelo son la decisión que muestra la
            # traza; van como dato privado: a la pantalla sí, al log nunca.
            emit(logger, "tool_start", private={"arguments": call.arguments}, run_id=context.run_id,
                 client_id=context.tenant.client_id, tool=call.tool_name)
            call_context = context
            decision = approvals.get(call.id)
            if registry.requires_approval(call.tool_name) and decision is not None:
                if not decision.approved:
                    result = _rejection_result(call.tool_name, decision)
                    emit(logger, "tool_skipped", run_id=context.run_id, client_id=context.tenant.client_id,
                         tool=call.tool_name, error_type=result.metadata["error_type"])
                    messages.append(Message.tool(call, result))
                    traces.append(StepTrace(run_id=context.run_id, kind="tool", tool_name=call.tool_name,
                                            ok=False, error_type=result.metadata["error_type"]))
                    continue
                # La aprobación la pone el grafo, a partir de lo que decidió la persona.
                call_context = replace(context, approved=True)
            # Sin decisión, una tool que requiere aprobación no corre: registry.run
            # la frena con approval_required. Es la segunda barrera.
            # registry.run nunca lanza por un error esperable: lo devuelve como
            # ToolResult(ok=False) y vuelve al modelo como resultado de la tool.
            result = registry.run(call, call_context)
            if result.ok and IDEAS_METADATA_KEY in result.metadata:
                # Las ideas pasan al estado y salen de la metadata: el modelo ya
                # las ve en el contenido, y guardarlas también en el historial
                # las duplicaría en cada checkpoint.
                metadata = dict(result.metadata)
                new_ideas = [GeneratedIdea.model_validate(item) for item in metadata.pop(IDEAS_METADATA_KEY)]
                ideas += new_ideas
                result = result.model_copy(update={"metadata": metadata})
                emit(logger, "ideas_added", run_id=context.run_id, client_id=context.tenant.client_id,
                     count=len(new_ideas), ids=[idea.id for idea in new_ideas])
            messages.append(Message.tool(call, result))
            traces.append(
                StepTrace(
                    run_id=context.run_id,
                    kind="tool",
                    tool_name=call.tool_name,
                    ok=result.ok,
                    error_type=result.metadata.get("error_type"),
                    # Tokens de las llamadas al modelo que hizo la tool por dentro
                    # (0 si no llamó a ninguno): sin esto el costo del turno queda corto.
                    input_tokens=result.metadata.get(MODEL_INPUT_TOKENS_KEY, 0),
                    output_tokens=result.metadata.get(MODEL_OUTPUT_TOKENS_KEY, 0),
                    latency_ms=result.metadata.get("latency_ms", 0.0),
                )
            )
        # Las decisiones se usan una sola vez.
        return {"messages": messages, "trace": traces, "ideas": ideas, "approvals": {}}

    def await_approval(state: AgentState) -> dict[str, Any]:
        # Este nodo se ejecuta dos veces: al pausar y al retomar (LangGraph
        # repite el nodo pausado desde el principio). Por eso solo arma el
        # pedido, que es barato y no tiene efectos, y no ejecuta nada.
        context = ExecutionContext(tenant=TenantContext(state["client_id"]), run_id=state["run_id"])
        decisions: dict[str, ApprovalDecision] = {}
        requests: list[dict[str, Any]] = []
        for call in state["messages"][-1].tool_calls:
            if not registry.requires_approval(call.tool_name):
                continue
            try:
                requests.append(registry.approval_request(call, context))
            except ApprovalPreviewError as exc:
                decisions[call.id] = ApprovalDecision(approved=False, error=str(exc))
        if not requests:
            return {"approvals": decisions}

        # Primera vez: interrupt corta el grafo y el turno termina en pausa.
        # Al retomar: devuelve lo que mandó resume_turn.
        answer = interrupt({"requests": requests})
        decision = ApprovalDecision(approved=bool(answer["approved"]), comment=answer.get("comment"))
        for request in requests:
            decisions[request["call_id"]] = decision
        emit(logger, "approval_resolved", run_id=context.run_id, client_id=context.tenant.client_id,
             tools=[request["tool"] for request in requests], approved=decision.approved,
             has_comment=decision.comment is not None)
        return {
            "approvals": decisions,
            "trace": [StepTrace(run_id=context.run_id, kind="approval", ok=decision.approved,
                                tools_requested=[request["tool"] for request in requests])],
        }

    def step_limit(state: AgentState) -> dict[str, Any]:
        # Los pedidos pendientes se responden igual, con error: un historial con
        # pedidos de tools sin su resultado lo rechaza el proveedor en el turno
        # siguiente y la sesión quedaría rota.
        pending = state["messages"][-1].tool_calls
        closing = [
            Message.tool(
                call,
                ToolResult(ok=False, content="No se ejecutó: se alcanzó el límite de pasos del turno.",
                           metadata={"error_type": "step_limit"}),
            )
            for call in pending
        ]
        emit(logger, "step_limit", level=logging.WARNING, run_id=state["run_id"], client_id=state["client_id"],
             steps=state["steps"], max_steps=max_steps, not_executed=[call.tool_name for call in pending])
        return {
            "messages": [*closing, Message(role="model", text=STEP_LIMIT_REPLY)],
            "trace": [StepTrace(run_id=state["run_id"], kind="step_limit",
                                tools_requested=[call.tool_name for call in pending])],
        }

    def route_after_model(state: AgentState) -> Literal["run_tools", "await_approval", "step_limit", "__end__"]:
        last = state["messages"][-1]
        if not last.tool_calls:
            destination, reason = END, "el modelo respondió con texto"  # respuesta o pedido de aclaración
        elif state["steps"] >= max_steps:
            destination, reason = "step_limit", "pidió tools sin pasos disponibles"
        elif any(registry.requires_approval(call.tool_name) for call in last.tool_calls):
            destination, reason = "await_approval", "pidió una tool que requiere aprobación"
        else:
            destination, reason = "run_tools", "pidió tools"
        emit(logger, "route", run_id=state["run_id"], client_id=state["client_id"], to=destination,
             reason=reason, step=state["steps"], max_steps=max_steps)
        return destination

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("run_tools", run_tools)
    builder.add_node("await_approval", await_approval)
    builder.add_node("step_limit", step_limit)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("await_approval", "run_tools")
    builder.add_edge("run_tools", "call_model")
    builder.add_edge("step_limit", END)
    if checkpointer is None:
        checkpointer = InMemorySaver(serde=checkpoint_serializer())
    return builder.compile(checkpointer=checkpointer)


def open_session(client_id: str) -> AgentSession:
    """Abre una conversación para un cliente. Falla si el cliente no es válido o no existe.

    Se valida acá, una sola vez, para que un error de cliente no aparezca recién
    cuando una tool intente leer sus datos.
    """
    tenant = TenantContext(client_id)
    client_data_dir(tenant)  # lanza UnknownClientError si no hay datos del cliente
    return AgentSession(thread_id=uuid.uuid4().hex, client_id=tenant.client_id)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "duración desconocida"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def attachment_note(video: VideoRef) -> str:
    """Línea que se suma al mensaje del usuario para que el modelo sepa que hay un video.

    La escribe el sistema, no la persona: el formato es fijo y el nombre del
    archivo ya viene limpio de core/storage.py.
    """
    size_mb = video.size_bytes / (1024 * 1024)
    return (
        f"[Video adjunto: id {video.id} · «{video.filename}» · {_format_duration(video.duration_s)} · "
        f"{size_mb:.1f} MB]"
    )


def run_turn(
    graph, session: AgentSession, user_text: str, video_ids: list[str] | tuple[str, ...] = ()
) -> TurnOutcome:
    """Corre un turno completo: mensaje del usuario, tools que decida el modelo y respuesta.

    `video_ids` son videos ya subidos que la persona adjunta a este mensaje.
    Se buscan con el cliente de la sesión: un id de otro cliente no existe acá.
    """
    text = validate_user_text(user_text)
    tenant = TenantContext(session.client_id)
    # Sin repetidos y en orden. get_video lanza UnknownVideoError si el id no es
    # de este cliente, antes de tocar el grafo o llamar al modelo.
    videos = [get_video(tenant, video_id) for video_id in dict.fromkeys(video_ids)]
    if videos:
        text = "\n\n".join([text, *(attachment_note(video) for video in videos)])
    config = _config(session)
    snapshot = _checked_snapshot(graph, session)
    # Un mensaje nuevo con una pausa abierta dejaría el pedido pendiente sin
    # resultado, y el proveedor rechaza ese historial: primero se resuelve.
    if snapshot.interrupts:
        raise PendingApprovalError("la sesión espera una aprobación")
    previous = snapshot.values
    # Un video que ya estaba en la sesión se vuelve a mencionar en el mensaje,
    # pero no se suma dos veces al estado.
    known_videos = {video.id for video in previous.get("videos", [])}
    new_videos = [video for video in videos if video.id not in known_videos]

    run_id = llm.new_run_id()

    def start() -> None:
        emit(logger, "turn_start", run_id=run_id, client_id=session.client_id, message_chars=len(text),
             history_messages=len(previous.get("messages", [])))
        for video in videos:
            emit(logger, "video_attached", run_id=run_id, client_id=session.client_id, video_id=video.id,
                 container=video.container, size_bytes=video.size_bytes, duration_s=video.duration_s)

    graph_input = {
        "client_id": session.client_id,
        "run_id": run_id,
        "messages": [Message.user(text)],
        "steps": 0,  # sin reducer: este valor reemplaza al del turno anterior
        "videos": new_videos,
        "approvals": {},
    }
    return _drive(graph, session, config, previous, run_id, graph_input, start)


def resume_turn(graph, session: AgentSession, approved: bool, comment: str | None = None) -> TurnOutcome:
    """Retoma un turno en pausa con la decisión de la persona sobre lo que había que aprobar.

    Es el mismo turno (mismo run_id): sigue donde se cortó. Si se aprueba, la
    tool corre; si se rechaza, no corre y el modelo recibe el rechazo con el
    comentario, para corregir o preguntar.
    """
    if comment is not None:
        # Va al modelo: pasa por la misma validación que cualquier mensaje.
        comment = validate_user_text(comment, max_len=MAX_APPROVAL_COMMENT_CHARS) if comment.strip() else None
    config = _config(session)
    snapshot = _checked_snapshot(graph, session)
    if not snapshot.interrupts:
        raise NoPendingApprovalError("la sesión no tiene una aprobación pendiente")
    previous = snapshot.values
    run_id = previous["run_id"]

    def start() -> None:
        emit(logger, "turn_resumed", run_id=run_id, client_id=session.client_id, approved=approved,
             has_comment=comment is not None)

    return _drive(graph, session, config, previous, run_id,
                  Command(resume={"approved": approved, "comment": comment}), start)


def pending_approval(graph, session: AgentSession) -> list[dict[str, Any]] | None:
    """Lo que la sesión espera que se apruebe, o None si no está en pausa."""
    snapshot = _checked_snapshot(graph, session)
    return _interrupt_requests(snapshot.interrupts)


def _config(session: AgentSession) -> dict[str, Any]:
    return {"configurable": {"thread_id": session.thread_id}}


def _checked_snapshot(graph, session: AgentSession):
    """Estado actual del hilo, verificando que sea del cliente de la sesión.

    Es la defensa para cuando el thread_id llega de afuera (la API).
    """
    snapshot = graph.get_state(_config(session))
    existing = snapshot.values.get("client_id")
    if existing is not None and existing != session.client_id:
        raise SessionClientMismatchError("la sesión pertenece a otro cliente")
    return snapshot


def _interrupt_requests(interrupts) -> list[dict[str, Any]] | None:
    requests = [request for item in interrupts for request in item.value.get("requests", [])]
    return requests or None


def _drive(graph, session: AgentSession, config, previous: dict[str, Any], run_id: str, graph_input,
           start) -> TurnOutcome:
    """Corre el grafo (un turno nuevo o uno retomado) y arma el resultado con lo que se sumó."""
    # El estado se acumula en la sesión: lo de esta llamada es lo que se suma
    # después de lo que ya había. Así un turno retomado no repite en la
    # pantalla los pasos que ya mostró antes de la pausa.
    ideas_before = len(previous.get("ideas", []))
    trace_before = len(previous.get("trace", []))
    messages_before = len(previous.get("messages", []))

    # Todo lo que se emita durante el turno, en este hilo o en los de las
    # tools y el proveedor, queda en `collector`.
    with collect(run_id) as collector:
        start()
        final = graph.invoke(graph_input, config)
        pending = _interrupt_requests(final.get("__interrupt__", ()))
        trace = [step for step in final["trace"][trace_before:] if step.run_id == run_id]
        ideas = final.get("ideas", [])[ideas_before:]
        renders = [
            message.result.metadata[RENDER_METADATA_KEY]
            for message in final["messages"][messages_before:]
            if message.role == "tool" and message.result is not None and message.result.ok
            and RENDER_METADATA_KEY in message.result.metadata
        ]
        if pending:
            emit(logger, "approval_requested", run_id=run_id, client_id=session.client_id,
                 tools=[request["tool"] for request in pending],
                 private={"requests": pending})
        emit(
            logger,
            "turn_paused" if pending else "turn_end",
            run_id=run_id,
            client_id=session.client_id,
            steps=final["steps"],
            tools_called=[step.tool_name for step in trace if step.kind == "tool" and step.tool_name],
            ideas_added=len(ideas),
            input_tokens=sum(step.input_tokens for step in trace),
            output_tokens=sum(step.output_tokens for step in trace),
        )

    last = final["messages"][-1]
    return TurnOutcome(
        # En pausa, el último mensaje es el del modelo que pidió la tool: puede
        # traer texto ("te paso el plan para que lo apruebes") o nada.
        reply=last.text or "",
        run_id=run_id,
        trace=trace,
        ideas=ideas,
        events=collector.events,
        pending_approval=pending,
        renders=renders,
    )
