"""refine_idea: ajuste de una idea de la sesión, sus errores y el recorrido por el grafo."""

import json
from functools import partial
from types import SimpleNamespace

import pytest

from yourebrand.core import llm, tenancy
from yourebrand.core.providers import Message, ProviderResult, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import ExecutionContext, ToolCall
from yourebrand.creative.graph import AgentSession, build_graph, run_turn
from yourebrand.creative.schemas import GeneratedIdea
from yourebrand.creative.tools import build_creative_registry

ORIGINAL = GeneratedIdea(id="idea-aaaaaa", content_type="ugc", title="Visita al taller", format="reel",
                         pillar="Cómo funciona", description="Recorrida por el taller.", rationale="Muestra el oficio.")
REFINED = {"title": "Visita corta al taller", "format": "reel", "pillar": "Cómo funciona",
           "description": "Recorrida de 15 segundos.", "rationale": "Muestra el oficio sin aburrir."}


class StubProvider:
    name = "stub"

    def __init__(self):
        self.generate_calls = []
        self.turns = []
        self.outputs = []  # respuestas de generate en orden; vacía = la idea ajustada

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        output = self.outputs.pop(0) if self.outputs else REFINED
        return ProviderResult(text=json.dumps(output), model=kwargs["model"], input_tokens=40, output_tokens=20,
                              stop_reason="end_turn", truncated=False)

    def generate_turn(self, **kwargs):
        return self.turns.pop(0)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / "brand.md").write_text("Acme vende yunques artesanales.", encoding="utf-8")
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    stub = StubProvider()
    monkeypatch.setattr(llm, "get_provider", lambda *args: stub)
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        model_generation="model-gen", model_fast="model-fast", llm_timeout_seconds=30, llm_max_retries=0))
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


def run_refine(arguments, ideas=(ORIGINAL,)):
    context = ExecutionContext(tenant=TenantContext("acme"), run_id="run-1", session_ideas=tuple(ideas))
    registry = build_creative_registry(generation_timeout_s=5, video_edit=False)
    return registry.run(ToolCall(id="c1", tool_name="refine_idea", arguments=arguments), context)


def test_refine_returns_a_new_idea_and_keeps_the_link_to_the_original(provider):
    result = run_refine({"idea_id": ORIGINAL.id, "comment": "hacela más corta"})

    assert result.ok
    refined = GeneratedIdea.model_validate(json.loads(result.content))
    assert refined.id != ORIGINAL.id and refined.id.startswith("idea-")
    assert refined.refined_from == ORIGINAL.id
    assert refined.content_type == "ugc"
    assert refined.title == "Visita corta al taller"
    # Pasa al estado como cualquier idea, y la traza cuenta los tokens de la llamada.
    assert result.metadata["ideas"] == [refined.model_dump()]
    assert result.metadata["model_input_tokens"] == 40

    prompt = provider.generate_calls[0]["prompt"]
    assert "hacela más corta" in prompt and "Visita al taller" in prompt and "Acme vende yunques" in prompt


def test_unknown_idea_does_not_call_the_model(provider):
    result = run_refine({"idea_id": "idea-zzzzzz", "comment": "más corta"})

    assert not result.ok and result.metadata["error_type"] == "unknown_idea"
    assert ORIGINAL.id in result.content  # le dice al modelo cuáles existen
    assert provider.generate_calls == []


@pytest.mark.parametrize("comment", ["   ", "con \x07 un carácter de control", "x" * 1001])
def test_invalid_comment_is_rejected_before_the_model(provider, comment):
    result = run_refine({"idea_id": ORIGINAL.id, "comment": comment})

    assert not result.ok
    assert provider.generate_calls == []


def test_refine_in_a_later_turn_uses_the_ideas_saved_in_the_session(provider):
    def turn(text=None, calls=()):
        return TurnResult(message=Message(role="model", text=text, tool_calls=list(calls)), model="m",
                          input_tokens=1, output_tokens=1, stop_reason="stop", truncated=False)

    graph = build_graph(build_creative_registry(generation_timeout_s=5, video_edit=False))
    session = AgentSession(thread_id="hilo", client_id="acme")

    # Turno 1: genera una idea.
    provider.outputs = [{"ideas": [REFINED | {"title": "Original"}]}]
    provider.turns = [turn(calls=[ToolCall(id="g", tool_name="generate_ideas",
                                           arguments={"content_type": "ugc", "count": 1})]), turn("Te dejo una.")]
    original_id = run_turn(graph, session, "una idea UGC").ideas[0].id

    # Turno 2: la ajusta. La idea sale del estado guardado, no del mensaje.
    provider.turns = [turn(calls=[ToolCall(id="r", tool_name="refine_idea",
                                           arguments={"idea_id": original_id, "comment": "más corta"})]),
                      turn("Ajustada.")]
    second = run_turn(graph, session, f"hacé más corta la {original_id}")

    assert second.reply == "Ajustada."
    assert [idea.refined_from for idea in second.ideas] == [original_id]
    state_ideas = graph.get_state({"configurable": {"thread_id": "hilo"}}).values["ideas"]
    assert [idea.id for idea in state_ideas] == [original_id, second.ideas[0].id]
