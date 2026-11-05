"""Grafo del Agente Creativo con un proveedor simulado: ruteo, tools, errores y tope de pasos."""

from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from yourebrand.core import errors, llm, tenancy
from yourebrand.core.providers import Message, TurnResult
from yourebrand.core.reliability import InvalidUserInputError, call_with_retry
from yourebrand.core.tenancy import InvalidClientIdError, UnknownClientError
from yourebrand.core.tools import Tool, ToolCall, ToolRegistry, ToolResult
from yourebrand.creative import graph as agent_graph
from yourebrand.creative.graph import AgentSession, build_graph, open_session, run_turn


class ScriptedProvider:
    """Devuelve turnos armados de antemano, uno por llamada, y guarda lo que recibe."""

    name = "scripted"

    def __init__(self):
        self.script = []  # TurnResult o excepción, en orden
        self.calls = []

    def generate_turn(self, **kwargs):
        # Copia del historial: la lista que recibe puede seguir cambiando después.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def text_turn(text):
    return TurnResult(
        message=Message(role="model", text=text), model="m", input_tokens=10, output_tokens=5,
        stop_reason="stop", truncated=False,
    )


def tool_turn(*names, arguments=None):
    calls = [ToolCall(id=f"call-{i}", tool_name=name, arguments=arguments or {"text": "hola"})
             for i, name in enumerate(names)]
    return TurnResult(
        message=Message(role="model", tool_calls=calls), model="m", input_tokens=20, output_tokens=3,
        stop_reason="stop", truncated=False,
    )


def malformed_turn():
    return TurnResult(
        message=Message(role="model"), model="m", input_tokens=20, output_tokens=0,
        stop_reason="MALFORMED_FUNCTION_CALL", truncated=False, malformed_tool_call=True,
    )


class EchoArgs(BaseModel):
    text: str


@pytest.fixture
def provider(monkeypatch):
    """El grafo usa el core/llm.py real; solo se reemplaza el proveedor y la configuración."""
    stub = ScriptedProvider()
    monkeypatch.setattr(llm, "get_provider", lambda: stub)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(
            model_generation="model-gen", model_fast="model-fast", llm_timeout_seconds=30, llm_max_retries=0
        ),
    )
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


@pytest.fixture
def seen_clients():
    """Clientes con los que se ejecutó la tool de prueba."""
    return []


@pytest.fixture
def registry(seen_clients):
    def echo(args, ctx):
        seen_clients.append(ctx.tenant.client_id)
        return ToolResult(ok=True, content=f"eco: {args['text']}")

    def broken(args, ctx):
        raise ValueError("se rompió")

    reg = ToolRegistry()
    reg.register(Tool("echo", "Devuelve el texto", "read", echo, EchoArgs))
    reg.register(Tool("broken", "Siempre falla", "read", broken, EchoArgs))
    return reg


SESSION = AgentSession(thread_id="hilo-1", client_id="acme")


def test_text_only_reply_ends_turn_without_tools(provider, registry):
    provider.script = [text_turn("¿Las ideas son de UGC o de diseño?")]
    outcome = run_turn(build_graph(registry), SESSION, "quiero ideas")

    assert outcome.reply == "¿Las ideas son de UGC o de diseño?"
    assert outcome.tools_called == []
    assert [step.kind for step in outcome.trace] == ["model"]
    assert len(provider.calls) == 1


def test_model_receives_tool_declarations(provider, registry):
    provider.script = [text_turn("hola")]
    run_turn(build_graph(registry), SESSION, "hola")

    names = [tool["name"] for tool in provider.calls[0]["tools"]]
    assert names == ["echo", "broken"]


def test_tool_call_runs_and_result_goes_back_to_model(provider, registry, seen_clients):
    provider.script = [tool_turn("echo"), text_turn("listo")]
    outcome = run_turn(build_graph(registry), SESSION, "usá echo")

    assert outcome.reply == "listo"
    assert outcome.tools_called == ["echo"]
    # La segunda llamada al modelo ve el resultado de la tool, emparejado con su pedido.
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-0"
    assert tool_message.result.ok is True
    assert tool_message.result.content == "eco: hola"


def test_tool_gets_client_from_session_not_from_model(provider, registry, seen_clients):
    # Aunque el modelo mande un client_id en los argumentos, la tool trabaja
    # para el cliente de la sesión.
    provider.script = [tool_turn("echo", arguments={"text": "hola", "client_id": "otro"}), text_turn("ok")]
    run_turn(build_graph(registry), SESSION, "usá echo")

    assert seen_clients == ["acme"]


def test_several_tool_calls_in_one_turn(provider, registry):
    provider.script = [tool_turn("echo", "echo"), text_turn("listo")]
    outcome = run_turn(build_graph(registry), SESSION, "dos veces")

    assert outcome.tools_called == ["echo", "echo"]
    tool_messages = [m for m in provider.calls[1]["messages"] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["call-0", "call-1"]


def test_tool_error_goes_back_to_model_and_conversation_continues(provider, registry):
    provider.script = [tool_turn("broken"), text_turn("La herramienta falló, probemos de otra forma.")]
    outcome = run_turn(build_graph(registry), SESSION, "usá broken")

    assert outcome.reply == "La herramienta falló, probemos de otra forma."
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message.result.ok is False
    tool_step = next(step for step in outcome.trace if step.kind == "tool")
    assert tool_step.ok is False
    assert tool_step.error_type == "ValueError"


def test_unknown_tool_goes_back_to_model(provider, registry):
    provider.script = [tool_turn("inventada"), text_turn("ok")]
    outcome = run_turn(build_graph(registry), SESSION, "hola")

    assert provider.calls[1]["messages"][-1].result.metadata["error_type"] == "unknown_tool"
    assert outcome.reply == "ok"


def test_step_limit_closes_pending_calls_and_replies(provider, registry):
    provider.script = [tool_turn("echo"), tool_turn("echo")]
    graph = build_graph(registry, max_steps=2)
    outcome = run_turn(graph, SESSION, "loop")

    assert outcome.reply == agent_graph.STEP_LIMIT_REPLY
    assert len(provider.calls) == 2  # nunca pasa del tope
    assert outcome.tools_called == ["echo"]  # el segundo pedido no se ejecutó
    assert outcome.trace[-1].kind == "step_limit"

    # Todo pedido de tool del historial tiene su resultado: la sesión sigue usable.
    messages = graph.get_state({"configurable": {"thread_id": SESSION.thread_id}}).values["messages"]
    requested = {call.id for m in messages for call in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    assert requested <= answered


def test_steps_reset_on_each_turn(provider, registry):
    graph = build_graph(registry, max_steps=2)
    provider.script = [tool_turn("echo"), text_turn("uno")]
    run_turn(graph, SESSION, "primero")
    # Si los pasos no se reiniciaran, este turno arrancaría ya en el tope.
    provider.script = [tool_turn("echo"), text_turn("dos")]
    outcome = run_turn(graph, SESSION, "segundo")

    assert outcome.reply == "dos"


def test_malformed_tool_call_is_retried_without_entering_history(provider, registry):
    provider.script = [malformed_turn(), text_turn("bien")]
    graph = build_graph(registry)
    outcome = run_turn(graph, SESSION, "hola")

    assert outcome.reply == "bien"
    assert len(provider.calls) == 2
    messages = graph.get_state({"configurable": {"thread_id": SESSION.thread_id}}).values["messages"]
    assert [m.role for m in messages] == ["user", "model"]
    assert outcome.trace[0].error_type == "malformed_tool_call"


def test_malformed_tool_call_twice_gives_fixed_reply(provider, registry):
    provider.script = [malformed_turn(), malformed_turn()]
    outcome = run_turn(build_graph(registry), SESSION, "hola")

    assert outcome.reply == agent_graph.MALFORMED_REPLY


def test_model_error_gives_fixed_reply_and_session_keeps_working(provider, registry):
    graph = build_graph(registry)
    provider.script = [errors.InvalidRequestError("pedido inválido")]
    outcome = run_turn(graph, SESSION, "hola")

    assert outcome.reply == agent_graph.MODEL_ERROR_REPLY
    assert outcome.trace[0].error_type == "InvalidRequestError"

    provider.script = [text_turn("ahora sí")]
    assert run_turn(graph, SESSION, "de nuevo").reply == "ahora sí"


def test_history_persists_between_turns(provider, registry):
    graph = build_graph(registry)
    provider.script = [text_turn("¿UGC o diseño?")]
    run_turn(graph, SESSION, "quiero ideas")
    provider.script = [text_turn("dale")]
    run_turn(graph, SESSION, "UGC")

    texts = [m.text for m in provider.calls[1]["messages"]]
    assert texts == ["quiero ideas", "¿UGC o diseño?", "UGC"]


def test_each_turn_has_its_own_run_id_and_trace(provider, registry):
    graph = build_graph(registry)
    provider.script = [tool_turn("echo"), text_turn("uno")]
    first = run_turn(graph, SESSION, "primero")
    provider.script = [text_turn("dos")]
    second = run_turn(graph, SESSION, "segundo")

    assert first.run_id != second.run_id
    assert all(step.run_id == second.run_id for step in second.trace)
    assert [step.kind for step in second.trace] == ["model"]


def test_session_cannot_switch_client(provider, registry):
    graph = build_graph(registry)
    provider.script = [text_turn("hola")]
    run_turn(graph, SESSION, "hola")

    with pytest.raises(agent_graph.SessionClientMismatchError):
        run_turn(graph, AgentSession(thread_id=SESSION.thread_id, client_id="otro"), "hola")
    assert len(provider.calls) == 1


def test_invalid_user_text_fails_before_calling_model(provider, registry):
    with pytest.raises(InvalidUserInputError):
        run_turn(build_graph(registry), SESSION, "   ")
    assert provider.calls == []


def test_max_steps_must_be_positive(registry):
    with pytest.raises(ValueError):
        build_graph(registry, max_steps=0)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    return tmp_path


def test_open_session_for_existing_client(data_dir):
    (data_dir / "acme").mkdir()
    session = open_session("acme")

    assert session.client_id == "acme"
    assert session.thread_id
    assert open_session("acme").thread_id != session.thread_id


def test_open_session_rejects_unknown_client(data_dir):
    with pytest.raises(UnknownClientError):
        open_session("nadie")


def test_open_session_rejects_invalid_client_id(data_dir):
    with pytest.raises(InvalidClientIdError):
        open_session("../otro")
