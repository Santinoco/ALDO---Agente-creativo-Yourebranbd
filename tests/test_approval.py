"""Pausa de aprobación (human-in-the-loop) en el grafo y en la API, con proveedor simulado."""

from functools import partial
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from yourebrand.api.app import create_app
from yourebrand.core import llm, tenancy
from yourebrand.core.providers import Message, TurnResult
from yourebrand.core.reliability import InvalidUserInputError, call_with_retry
from yourebrand.core.tools import ApprovalPreviewError, Tool, ToolCall, ToolRegistry, ToolResult
from yourebrand.creative.graph import (
    AgentSession,
    NoPendingApprovalError,
    PendingApprovalError,
    build_graph,
    pending_approval,
    resume_turn,
    run_turn,
)


class ScriptedProvider:
    name = "scripted"

    def __init__(self):
        self.script = []
        self.calls = []

    def generate_turn(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


def text_turn(text):
    return TurnResult(message=Message(role="model", text=text), model="m", input_tokens=10, output_tokens=5,
                      stop_reason="stop", truncated=False)


def tool_turn(*calls, text=None):
    return TurnResult(
        message=Message(role="model", text=text,
                        tool_calls=[ToolCall(id=f"call-{i}", tool_name=name, arguments=args)
                                    for i, (name, args) in enumerate(calls)]),
        model="m", input_tokens=20, output_tokens=3, stop_reason="stop", truncated=False,
    )


@pytest.fixture
def provider(monkeypatch):
    stub = ScriptedProvider()
    monkeypatch.setattr(llm, "get_provider", lambda *args: stub)
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        model_generation="model-gen", model_fast="model-fast", llm_timeout_seconds=30, llm_max_retries=0))
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


class PublishArgs(BaseModel):
    item: str


@pytest.fixture
def ran():
    """Qué tools se ejecutaron, con qué y si llegó la aprobación."""
    return []


@pytest.fixture
def registry(ran):
    def echo(args, ctx):
        ran.append(("echo", ctx.approved))
        return ToolResult(ok=True, content="eco")

    def publish(args, ctx):
        ran.append(("publish", args["item"], ctx.approved, ctx.tenant.client_id))
        return ToolResult(ok=True, content=f"publicado {args['item']}")

    def preview(args, ctx):
        if args["item"] == "nada":
            raise ApprovalPreviewError("no existe 'nada'")
        return {"item": args["item"], "cliente": ctx.tenant.client_id}

    reg = ToolRegistry()
    reg.register(Tool("echo", "Eco", "read", echo))
    reg.register(Tool("publish", "Publica algo", "local_write", publish, PublishArgs, idempotent=False,
                      requires_approval=True, approval_preview=preview))
    return reg


SESSION = AgentSession(thread_id="hilo-1", client_id="acme")


def test_a_tool_that_requires_approval_pauses_the_turn_without_running(provider, registry, ran):
    provider.script = [tool_turn(("publish", {"item": "plan-1"}), text="Te paso el plan para aprobar.")]
    graph = build_graph(registry)
    outcome = run_turn(graph, SESSION, "publicalo")

    assert ran == []  # nada corrió antes de la decisión
    assert outcome.reply == "Te paso el plan para aprobar."
    assert outcome.pending_approval == [{
        "call_id": "call-0", "tool": "publish", "arguments": {"item": "plan-1"},
        "preview": {"item": "plan-1", "cliente": "acme"},
    }]
    assert pending_approval(graph, SESSION) == outcome.pending_approval
    assert "approval_requested" in [e.event for e in outcome.events]
    assert "turn_paused" in [e.event for e in outcome.events]


def test_a_new_message_is_refused_while_paused(provider, registry):
    provider.script = [tool_turn(("publish", {"item": "plan-1"}))]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "publicalo")

    with pytest.raises(PendingApprovalError):
        run_turn(graph, SESSION, "otra cosa")
    assert len(provider.calls) == 1


def test_approving_runs_the_tool_with_approval_and_continues_the_same_turn(provider, registry, ran):
    provider.script = [tool_turn(("publish", {"item": "plan-1"})), text_turn("Listo, publicado.")]
    graph = build_graph(registry)
    paused = run_turn(graph, SESSION, "publicalo")
    outcome = resume_turn(graph, SESSION, approved=True)

    assert ran == [("publish", "plan-1", True, "acme")]
    assert outcome.run_id == paused.run_id  # mismo turno
    assert outcome.reply == "Listo, publicado."
    assert outcome.pending_approval is None
    # La traza del turno retomado no repite el paso del modelo de antes de la pausa.
    assert [step.kind for step in outcome.trace] == ["approval", "tool", "model"]
    assert outcome.trace[0].ok is True
    assert pending_approval(graph, SESSION) is None


def test_rejecting_skips_the_tool_and_gives_the_model_the_comment(provider, registry, ran):
    provider.script = [tool_turn(("publish", {"item": "plan-1"})), text_turn("Ok, lo cambio.")]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "publicalo")
    outcome = resume_turn(graph, SESSION, approved=False, comment="sacá el último tramo")

    assert ran == []
    assert outcome.tools_called == ["publish"]
    assert outcome.trace[1].ok is False and outcome.trace[1].error_type == "rejected_by_user"
    # El pedido queda respondido: el historial que recibe el modelo es válido.
    result = provider.calls[1]["messages"][-1]
    assert result.role == "tool" and result.tool_call_id == "call-0"
    assert result.result.ok is False
    assert "sacá el último tramo" in result.result.content
    assert "<comentario_de_la_persona>" in result.result.content


def test_after_a_rejection_the_session_accepts_messages_again(provider, registry):
    provider.script = [tool_turn(("publish", {"item": "plan-1"})), text_turn("Ok."), text_turn("Hola.")]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "publicalo")
    resume_turn(graph, SESSION, approved=False)

    assert run_turn(graph, SESSION, "hola").reply == "Hola."


def test_an_approval_is_used_once(provider, registry, ran):
    # Aprobar un pedido no aprueba el siguiente: el modelo pide otra vez y se vuelve a pausar.
    provider.script = [
        tool_turn(("publish", {"item": "a"})),
        tool_turn(("publish", {"item": "b"})),
    ]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "publicá a")
    outcome = resume_turn(graph, SESSION, approved=True)

    assert ran == [("publish", "a", True, "acme")]
    assert outcome.pending_approval[0]["arguments"] == {"item": "b"}


def test_other_tools_in_the_same_step_run_only_once(provider, registry, ran):
    # Al retomar, LangGraph repite el nodo pausado: las tools sin aprobación no
    # pueden correr dos veces por eso.
    provider.script = [tool_turn(("echo", {}), ("publish", {"item": "a"})), text_turn("ok")]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "hacé las dos")
    assert ran == []
    resume_turn(graph, SESSION, approved=True)

    assert ran == [("echo", False), ("publish", "a", True, "acme")]


def test_nothing_to_approve_goes_back_to_the_model_without_pausing(provider, registry, ran):
    provider.script = [tool_turn(("publish", {"item": "nada"})), text_turn("Ese plan no existe.")]
    outcome = run_turn(build_graph(registry), SESSION, "publicá nada")

    assert outcome.pending_approval is None
    assert ran == []
    result = provider.calls[1]["messages"][-1].result
    assert result.metadata["error_type"] == "invalid_approval_request"
    assert "no existe" in result.content


def test_invalid_arguments_do_not_pause(provider, registry, ran):
    provider.script = [tool_turn(("publish", {})), text_turn("Me faltó el item.")]
    outcome = run_turn(build_graph(registry), SESSION, "publicá")

    assert outcome.pending_approval is None and ran == []
    assert provider.calls[1]["messages"][-1].result.metadata["error_type"] == "invalid_approval_request"


def test_resume_without_a_pause_fails(provider, registry):
    provider.script = [text_turn("hola")]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "hola")
    with pytest.raises(NoPendingApprovalError):
        resume_turn(graph, SESSION, approved=True)


def test_the_rejection_comment_is_validated(provider, registry):
    provider.script = [tool_turn(("publish", {"item": "a"}))]
    graph = build_graph(registry)
    run_turn(graph, SESSION, "publicá a")
    with pytest.raises(InvalidUserInputError):
        resume_turn(graph, SESSION, approved=False, comment="x" * 5000)
    # La pausa sigue abierta: se puede volver a decidir.
    assert pending_approval(graph, SESSION) is not None


def test_running_the_tool_outside_the_pause_is_blocked(registry, ran):
    from yourebrand.core.tenancy import TenantContext
    from yourebrand.core.tools import ExecutionContext

    result = registry.run(ToolCall(id="x", tool_name="publish", arguments={"item": "a"}),
                          ExecutionContext(tenant=TenantContext("acme"), run_id="r"))
    assert result.metadata["error_type"] == "approval_required" and ran == []


def test_a_preview_needs_requires_approval():
    with pytest.raises(ValueError):
        Tool("t", "d", "read", lambda a, c: None, approval_preview=lambda a, c: {})


# --- API ------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch, provider, registry):
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    (tmp_path / "acme").mkdir()
    with TestClient(create_app(graph=build_graph(registry))) as test_client:
        yield test_client


def open_thread(client):
    return client.post("/sessions", json={"client_id": "acme"}).json()["thread_id"]


def test_api_pauses_then_resolves_the_approval(client, provider, ran):
    provider.script = [tool_turn(("publish", {"item": "plan-1"})), text_turn("Publicado.")]
    thread = open_thread(client)

    paused = client.post(f"/sessions/{thread}/messages", json={"text": "publicalo"}).json()
    assert paused["pending_approval"][0]["tool"] == "publish"

    blocked = client.post(f"/sessions/{thread}/messages", json={"text": "otra cosa"})
    assert blocked.status_code == 409

    done = client.post(f"/sessions/{thread}/approval", json={"approved": True})
    assert done.status_code == 200
    assert done.json()["reply"] == "Publicado." and done.json()["pending_approval"] is None
    assert ran == [("publish", "plan-1", True, "acme")]


def test_api_approval_without_a_pause_is_a_conflict(client):
    thread = open_thread(client)
    assert client.post(f"/sessions/{thread}/approval", json={"approved": True}).status_code == 409


def test_api_approval_on_an_unknown_session_is_404(client):
    assert client.post("/sessions/nada/approval", json={"approved": True}).status_code == 404


def test_api_unknown_render_is_404(client, tmp_path, monkeypatch):
    from yourebrand.core import storage

    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(storage_dir=tmp_path / "storage"))
    thread = open_thread(client)
    assert client.get(f"/sessions/{thread}/renders/render-000000000000").status_code == 404
    assert client.get(f"/sessions/{thread}/renders/../../etc").status_code == 404
