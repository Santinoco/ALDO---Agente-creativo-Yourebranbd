"""API de la pantalla de demo: transporte HTTP sobre el grafo, con proveedor simulado."""

from functools import partial
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yourebrand.core import llm, tenancy
from yourebrand.core.providers import Message, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tools import Tool, ToolCall, ToolRegistry, ToolResult
from yourebrand.api.app import create_app
from yourebrand.creative.graph import build_graph


class ScriptedProvider:
    name = "scripted"

    def __init__(self):
        self.turns = []
        self.calls = []

    def generate_turn(self, **kwargs):
        self.calls.append(kwargs)
        return self.turns.pop(0)


def turn(*, text=None, calls=()):
    return TurnResult(
        message=Message(role="model", text=text, tool_calls=list(calls)),
        model="m", input_tokens=10, output_tokens=2, stop_reason="stop", truncated=False,
    )


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    for client_id in ("acme", "globex"):
        (tmp_path / client_id).mkdir()
    (tmp_path / "Borradores").mkdir()  # no es un client_id válido: no aparece
    return tmp_path


@pytest.fixture
def provider(monkeypatch):
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
    return []


@pytest.fixture
def client(data_dir, provider, seen_clients):
    def echo(args, ctx):
        seen_clients.append(ctx.tenant.client_id)
        return ToolResult(ok=True, content="eco")

    registry = ToolRegistry()
    registry.register(Tool("echo", "Devuelve eco", "read", echo))
    # Sin `with`: no corre el lifespan, que armaría el grafo real con el .env.
    return TestClient(create_app(build_graph(registry)))


def open_session(client, client_id="acme"):
    response = client.post("/sessions", json={"client_id": client_id})
    assert response.status_code == 200
    return response.json()["thread_id"]


def test_index_serves_the_screen(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Agente Creativo" in response.text


def test_clients_lists_only_valid_client_folders(client):
    assert client.get("/clients").json() == ["acme", "globex"]


def test_create_session(client):
    response = client.post("/sessions", json={"client_id": "acme"})

    assert response.status_code == 200
    body = response.json()
    assert body["client_id"] == "acme"
    assert body["thread_id"]


@pytest.mark.parametrize(("client_id", "status"), [("nadie", 404), ("../acme", 422)])
def test_create_session_rejects_bad_clients(client, client_id, status):
    assert client.post("/sessions", json={"client_id": client_id}).status_code == status


def test_message_returns_reply_trace_and_ideas(client, provider):
    thread_id = open_session(client)
    provider.turns = [turn(calls=[ToolCall(id="c1", tool_name="echo")]), turn(text="listo")]

    response = client.post(f"/sessions/{thread_id}/messages", json={"text": "hola"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "listo"
    assert body["tools_called"] == ["echo"]
    assert [step["kind"] for step in body["trace"]] == ["model", "tool", "model"]
    assert (body["input_tokens"], body["output_tokens"]) == (20, 4)  # dos llamadas del agente
    events = [event["event"] for event in body["events"]]
    assert events[0] == "turn_start" and events[-1] == "turn_end"
    assert "tool_start" in events and "route" in events
    assert body["ideas"] == []
    assert body["run_id"]


def test_session_keeps_its_client(client, provider, seen_clients):
    acme = open_session(client, "acme")
    globex = open_session(client, "globex")
    provider.turns = [
        turn(calls=[ToolCall(id="a", tool_name="echo")]), turn(text="a"),
        turn(calls=[ToolCall(id="g", tool_name="echo")]), turn(text="g"),
    ]

    client.post(f"/sessions/{acme}/messages", json={"text": "hola"})
    client.post(f"/sessions/{globex}/messages", json={"text": "hola"})

    # El cliente sale de la sesión abierta en el servidor, no de la petición.
    assert seen_clients == ["acme", "globex"]


def test_message_to_unknown_session_is_404(client, provider):
    response = client.post("/sessions/no-existe/messages", json={"text": "hola"})

    assert response.status_code == 404
    assert provider.calls == []


def test_invalid_text_is_422_without_calling_model(client, provider):
    thread_id = open_session(client)

    response = client.post(f"/sessions/{thread_id}/messages", json={"text": "   "})

    assert response.status_code == 422
    assert provider.calls == []
