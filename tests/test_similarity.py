"""check_similarity: validación de la respuesta y recorrido por el grafo."""

import json
from functools import partial
from types import SimpleNamespace

import pytest

from yourebrand.core import llm
from yourebrand.core.providers import Message, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import ExecutionContext, ToolCall
from yourebrand.creative import similarity, tools
from yourebrand.creative.graph import AgentSession, build_graph, run_turn
from yourebrand.creative.schemas import GeneratedIdea, Idea, SimilarityReport, SimilarityVerdict

IDEA = GeneratedIdea(id="idea-aaaaaa", content_type="ugc", title="Cafés", format="reel", pillar="Rankings",
                     description="Top de cafés.", rationale="Encaja.")
POSTS = "# Posts\n\n## 2026-11-07 · Reel · Rankings\n\nCafés.\n\n## 2026-10-24 · Carrusel · Escapadas\n\nSierras.\n"


def answer(monkeypatch, *verdicts):
    usage = SimpleNamespace(input_tokens=11, output_tokens=7)
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=SimilarityReport(verdicts=list(verdicts)), response=usage)
    monkeypatch.setattr(similarity, "complete_structured", fake)
    monkeypatch.setattr(similarity, "load_previous_posts", lambda tenant: POSTS)
    return calls


def ctx(*ideas):
    return ExecutionContext(tenant=TenantContext("tambor"), run_id="r", session_ideas=tuple(ideas))


def verdict(label, refs):
    return SimilarityVerdict(idea_id=IDEA.id, reason="motivo", verdict=label, reference_ids=refs)


def test_split_posts():
    refs = similarity.split_posts(POSTS)
    assert [r.id for r in refs] == ["post-1", "post-2"]
    assert refs[1].title == "2026-10-24 · Carrusel · Escapadas" and refs[1].text == "Sierras."


def test_flags_a_repeated_idea_with_the_fast_model(monkeypatch):
    calls = answer(monkeypatch, verdict("repetida", ["post-1"]))
    result = tools.check_similarity_tool({"idea_ids": [IDEA.id]}, ctx(IDEA))
    body = json.loads(result.content)
    assert result.ok and calls[0]["tier"] == "fast"
    assert body["results"][0]["similar_to"][0]["title"].startswith("2026-11-07")


def test_invented_reference_is_an_error_with_usage(monkeypatch):
    answer(monkeypatch, verdict("parecida", ["post-99"]))
    result = tools.check_similarity_tool({"idea_ids": [IDEA.id]}, ctx(IDEA))
    assert not result.ok and result.metadata["error_type"] == "InvalidSimilarityError"


def test_unknown_idea_does_not_call_the_model(monkeypatch):
    calls = answer(monkeypatch)
    result = tools.check_similarity_tool({"idea_ids": ["idea-zzzzzz"]}, ctx(IDEA))
    assert not result.ok and result.metadata["error_type"] == "unknown_idea" and calls == []


def test_without_posts_it_says_not_checked(monkeypatch):
    calls = answer(monkeypatch)
    monkeypatch.setattr(similarity, "load_previous_posts", lambda tenant: None)
    result = tools.check_similarity_tool({"idea_ids": [IDEA.id]}, ctx(IDEA))
    assert result.ok and json.loads(result.content)["checked"] is False and calls == []


def test_graph_passes_session_ideas_across_turns(monkeypatch):
    class Scripted:
        name = "s"
        def __init__(self): self.script = []
        def generate_turn(self, **kwargs): return self.script.pop(0)
    def turn(text=None, calls=()):
        return TurnResult(message=Message(role="model", text=text, tool_calls=list(calls)), model="m",
                          input_tokens=1, output_tokens=1, stop_reason="stop", truncated=False)

    stub = Scripted()
    monkeypatch.setattr(llm, "get_provider", lambda *a: stub)
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        model_generation="g", model_fast="f", llm_timeout_seconds=30, llm_max_retries=0))
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda s: None))
    monkeypatch.setattr(tools, "generate_ideas_with_usage", lambda *a, **k: SimpleNamespace(
        ideas=[Idea(**IDEA.model_dump(exclude={"id", "content_type"}))], usage=SimpleNamespace(input_tokens=1, output_tokens=1)))
    monkeypatch.setattr(tools, "_new_idea_id", lambda: IDEA.id)
    answer(monkeypatch, verdict("distinta", ["post-1"]))

    stub.script = [
        turn(calls=[ToolCall(id="c1", tool_name="generate_ideas", arguments={"content_type": "ugc"})]),
        turn(calls=[ToolCall(id="c2", tool_name="check_similarity", arguments={"idea_ids": [IDEA.id]})]),
        turn(text="Listo."),
    ]
    graph = build_graph(tools.build_creative_registry(generation_timeout_s=5, video_edit=False))
    outcome = run_turn(graph, AgentSession(thread_id="h", client_id="tambor"), "ideas")
    assert outcome.reply == "Listo."
    steps = [(s.tool_name, s.ok) for s in outcome.trace if s.kind == "tool"]
    assert ("check_similarity", True) in steps
