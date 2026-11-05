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
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from yourebrand.core import llm
from yourebrand.core.errors import LLMError
from yourebrand.core.providers import Message
from yourebrand.core.reliability import validate_user_text
from yourebrand.core.tenancy import TenantContext, client_data_dir
from yourebrand.core.tools import ExecutionContext, ToolRegistry, ToolResult
from yourebrand.creative.prompts import build_agent_system_prompt

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

    @property
    def tools_called(self) -> list[str]:
        """Tools que se ejecutaron en el turno, en orden."""
        return [step.tool_name for step in self.trace if step.kind == "tool" and step.tool_name]


class SessionClientMismatchError(ValueError):
    """La sesión ya pertenece a otro cliente."""


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
        for call in pending:
            # registry.run nunca lanza por un error esperable: lo devuelve como
            # ToolResult(ok=False) y vuelve al modelo como resultado de la tool.
            result = registry.run(call, context)
            messages.append(Message.tool(call, result))
            traces.append(
                StepTrace(
                    run_id=context.run_id,
                    kind="tool",
                    tool_name=call.tool_name,
                    ok=result.ok,
                    error_type=result.metadata.get("error_type"),
                    latency_ms=result.metadata.get("latency_ms", 0.0),
                )
            )
        return {"messages": messages, "trace": traces}

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
        logger.warning("agent_step_limit run_id=%s client_id=%s steps=%d", state["run_id"], state["client_id"],
                       state["steps"])
        return {
            "messages": [*closing, Message(role="model", text=STEP_LIMIT_REPLY)],
            "trace": [StepTrace(run_id=state["run_id"], kind="step_limit",
                                tools_requested=[call.tool_name for call in pending])],
        }

    def route_after_model(state: AgentState) -> Literal["run_tools", "step_limit", "__end__"]:
        last = state["messages"][-1]
        if not last.tool_calls:
            return END  # respuesta final o pedido de aclaración
        if state["steps"] >= max_steps:
            return "step_limit"
        return "run_tools"

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("run_tools", run_tools)
    builder.add_node("step_limit", step_limit)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("run_tools", "call_model")
    builder.add_edge("step_limit", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


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
    existing = graph.get_state(config).values.get("client_id")
    if existing is not None and existing != session.client_id:
        raise SessionClientMismatchError("la sesión pertenece a otro cliente")

    run_id = llm.new_run_id()
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
    outcome = TurnOutcome(reply=final["messages"][-1].text or "", run_id=run_id, trace=trace)
    logger.info(
        "agent_turn run_id=%s client_id=%s steps=%d tools_called=%s",
        run_id,
        session.client_id,
        final["steps"],
        ",".join(outcome.tools_called) or "-",
    )
    return outcome
