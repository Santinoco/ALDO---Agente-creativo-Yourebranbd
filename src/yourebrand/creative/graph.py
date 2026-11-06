"""Grafo del Agente Creativo: el modelo decide en cada paso si usa tools o responde.

Tres nodos propios, sin los prearmados de LangGraph:

    START → call_model ─┬─ pidió tools y quedan pasos → run_tools → call_model
                        ├─ respondió solo texto ──────→ END
                        └─ pidió tools sin pasos ─────→ step_limit → END

Pedir aclaraciones no es un nodo: es el camino en el que el modelo responde con
texto sin pedir ninguna tool.

El cliente se fija al abrir la sesión y vive en el estado. Las tools lo reciben
del contexto que arma `run_tools`, nunca del modelo.
"""

import logging
import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from yourebrand.core import llm
from yourebrand.core.errors import LLMError
from yourebrand.core.providers import Message
from yourebrand.core.reliability import validate_user_text
from yourebrand.core.tenancy import TenantContext, client_data_dir
from yourebrand.core.tools import (
    MODEL_INPUT_TOKENS_KEY,
    MODEL_OUTPUT_TOKENS_KEY,
    ExecutionContext,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from yourebrand.core.trace import TraceEvent, collect, emit
from yourebrand.creative.prompts import build_agent_system_prompt
from yourebrand.creative.schemas import GeneratedIdea
from yourebrand.creative.tools import IDEAS_METADATA_KEY

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


class StepTrace(BaseModel):
    """Un paso del grafo, para la traza del turno y el set de evaluación.

    No guarda argumentos ni contenido: solo nombres, resultados y uso.
    """

    run_id: str
    kind: Literal["model", "tool", "step_limit"]
    # Paso "model": qué tools pidió el modelo (vacío si respondió con texto).
    tools_requested: list[str] = Field(default_factory=list)
    # Paso "tool": qué tool corrió y cómo terminó.
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


# Clases propias que guarda el estado, incluidas las anidadas (un Message lleva
# ToolCall y ToolResult). Al leer un checkpoint, LangGraph solo reconstruye las
# que están en esta lista: si la base de checkpoints se adulterara, no podría
# hacer que se construya cualquier otra clase de Python. Si el estado suma un
# tipo nuevo, va acá; un test lo controla.
CHECKPOINT_TYPES: tuple[type, ...] = (Message, ToolCall, ToolResult, StepTrace, GeneratedIdea)


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
        context = ExecutionContext(tenant=TenantContext(state["client_id"]), run_id=state["run_id"])
        pending = state["messages"][-1].tool_calls

        messages: list[Message] = []
        traces: list[StepTrace] = []
        ideas: list[GeneratedIdea] = []
        for call in pending:
            # Los argumentos que eligió el modelo son la decisión que muestra la
            # traza; van como dato privado: a la pantalla sí, al log nunca.
            emit(logger, "tool_start", private={"arguments": call.arguments}, run_id=context.run_id,
                 client_id=context.tenant.client_id, tool=call.tool_name)
            # registry.run nunca lanza por un error esperable: lo devuelve como
            # ToolResult(ok=False) y vuelve al modelo como resultado de la tool.
            result = registry.run(call, context)
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
        return {"messages": messages, "trace": traces, "ideas": ideas}

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

    def route_after_model(state: AgentState) -> Literal["run_tools", "step_limit", "__end__"]:
        last = state["messages"][-1]
        if not last.tool_calls:
            destination, reason = END, "el modelo respondió con texto"  # respuesta o pedido de aclaración
        elif state["steps"] >= max_steps:
            destination, reason = "step_limit", "pidió tools sin pasos disponibles"
        else:
            destination, reason = "run_tools", "pidió tools"
        emit(logger, "route", run_id=state["run_id"], client_id=state["client_id"], to=destination,
             reason=reason, step=state["steps"], max_steps=max_steps)
        return destination

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("run_tools", run_tools)
    builder.add_node("step_limit", step_limit)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
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


def run_turn(graph, session: AgentSession, user_text: str) -> TurnOutcome:
    """Corre un turno completo: mensaje del usuario, tools que decida el modelo y respuesta."""
    text = validate_user_text(user_text)
    config = {"configurable": {"thread_id": session.thread_id}}

    # Si el hilo ya existe, tiene que ser del mismo cliente. Es la defensa para
    # cuando el thread_id llegue de afuera (la API, en la iteración 4).
    previous = graph.get_state(config).values
    existing = previous.get("client_id")
    if existing is not None and existing != session.client_id:
        raise SessionClientMismatchError("la sesión pertenece a otro cliente")
    # Las ideas se acumulan en la sesión: las de este turno son las que se
    # suman después de las que ya había.
    ideas_before = len(previous.get("ideas", []))

    run_id = llm.new_run_id()
    # Todo lo que se emita durante el turno, en este hilo o en los de las
    # tools y el proveedor, queda en `collector`.
    with collect(run_id) as collector:
        emit(logger, "turn_start", run_id=run_id, client_id=session.client_id, message_chars=len(text),
             history_messages=len(previous.get("messages", [])))
        final = graph.invoke(
            {
                "client_id": session.client_id,
                "run_id": run_id,
                "messages": [Message.user(text)],
                "steps": 0,  # sin reducer: este valor reemplaza al del turno anterior
            },
            config,
        )
        trace = [step for step in final["trace"] if step.run_id == run_id]
        ideas = final.get("ideas", [])[ideas_before:]
        input_tokens = sum(step.input_tokens for step in trace)
        output_tokens = sum(step.output_tokens for step in trace)
        emit(
            logger,
            "turn_end",
            run_id=run_id,
            client_id=session.client_id,
            steps=final["steps"],
            tools_called=[step.tool_name for step in trace if step.kind == "tool" and step.tool_name],
            ideas_added=len(ideas),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return TurnOutcome(
        reply=final["messages"][-1].text or "",
        run_id=run_id,
        trace=trace,
        ideas=ideas,
        events=collector.events,
    )
