"""Set de evaluación: que los casos sean válidos y que las verificaciones detecten lo que deben."""

import json
from functools import partial
from types import SimpleNamespace

import pytest

from yourebrand.core import llm, tenancy
from yourebrand.core.providers import Message, ProviderResult, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tools import ToolCall
from yourebrand.core.trace import TraceEvent
from yourebrand.creative import evaluation
from yourebrand.creative.evaluation import EvalCase, TurnExpectation, check_turn, load_cases, run_case
from yourebrand.creative.graph import StepTrace, TurnOutcome, build_graph
from yourebrand.creative.tools import build_creative_registry


def outcome(*, tools=(), reply="", flagged=None, pending=None):
    trace = [StepTrace(run_id="r", kind="tool", tool_name=name, ok=True) for name in tools]
    events = [] if flagged is None else [TraceEvent(run_id="r", event="similarity_checked", at_ms=0,
                                                    fields={"flagged": flagged})]
    return TurnOutcome(reply=reply, run_id="r", trace=trace, ideas=[], events=events, pending_approval=pending)


def failed(checks):
    return [check.name for check in checks if not check.ok]


def test_the_real_case_set_is_valid_and_uses_existing_clients():
    cases = load_cases()
    assert len(cases) >= 10
    assert {case.criterion for case in cases} >= {"tools", "repeticion", "alcance"}
    for case in cases:
        assert (evaluation.ROOT / "data" / case.client_id / "brand.md").is_file(), case.id


def test_must_call_and_must_not_call():
    expect = TurnExpectation(must_call=["refine_idea"], must_not_call=["generate_ideas"])
    assert failed(check_turn(expect, outcome(tools=["refine_idea"]))) == []
    assert failed(check_turn(expect, outcome(tools=["generate_ideas"]))) == ["usa refine_idea", "no usa generate_ideas"]


def test_any_tool_wildcard():
    expect = TurnExpectation(must_not_call=["*"])
    assert failed(check_turn(expect, outcome())) == []
    assert failed(check_turn(expect, outcome(tools=["get_brand_info"]))) == ["no usa tools"]


@pytest.mark.parametrize(("flagged", "expected", "ok"), [(1, True, True), (0, True, False), (None, True, False),
                                                         (0, False, True), (2, False, False)])
def test_similarity_flag(flagged, expected, ok):
    checks = check_turn(TurnExpectation(similarity_flagged=expected), outcome(flagged=flagged))
    assert checks[0].ok is ok


def test_reply_checks_and_unexpected_pause():
    expect = TurnExpectation(reply_must_not_contain=["```"], reply_max_chars=20)
    assert failed(check_turn(expect, outcome(reply="Acá no puedo."))) == []
    names = failed(check_turn(expect, outcome(reply="```python\nprint(1)\n```", pending=[{"tool": "x"}])))
    assert names == ["la respuesta no contiene '```'", "respuesta de hasta 20 caracteres", "no queda en pausa"]


def test_run_case_fills_the_idea_placeholder_and_records_each_turn(tmp_path, monkeypatch):
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / "brand.md").write_text("Acme vende yunques.", encoding="utf-8")
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    idea = {"title": "T", "format": "reel", "pillar": "P", "description": "D", "rationale": "R"}

    class Stub:
        name = "stub"
        turns = []

        def generate(self, **kwargs):
            return ProviderResult(text=json.dumps({"ideas": [idea]}), model="g", input_tokens=10, output_tokens=5,
                                  stop_reason="end_turn", truncated=False)

        def generate_turn(self, **kwargs):
            # Guarda lo que recibió el agente: el segundo turno tiene que traer el id real.
            Stub.seen.append(kwargs["messages"][-1].text)
            return self.turns.pop(0)

    Stub.seen = []
    stub = Stub()
    monkeypatch.setattr(llm, "get_provider", lambda *args: stub)
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        model_generation="g", model_fast="f", llm_timeout_seconds=30, llm_max_retries=0))
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))

    def turn(text=None, calls=()):
        return TurnResult(message=Message(role="model", text=text, tool_calls=list(calls)), model="m",
                          input_tokens=1, output_tokens=1, stop_reason="stop", truncated=False)

    stub.turns = [turn(calls=[ToolCall(id="g", tool_name="generate_ideas", arguments={"content_type": "ugc", "count": 1})]),
                  turn("Una idea."), turn("¿Qué cambio?")]
    case = EvalCase.model_validate({
        "id": "c", "criterion": "tools", "client_id": "acme", "description": "d",
        "turns": [{"message": "una idea", "expect": {"must_call": ["generate_ideas"]}},
                  {"message": "cambiá la {idea_1}", "expect": {"must_call": ["refine_idea"]}}],
    })

    result = run_case(build_graph(build_creative_registry(generation_timeout_s=5, video_edit=False)), case)

    first, second = result.turns
    assert first.passed and first.tools_called == ["generate_ideas"] and first.run_id
    assert first.input_tokens > 0 and first.steps == 2
    assert second.message.startswith("cambiá la idea-")
    assert not second.passed and not result.passed  # el agente no llamó a refine_idea
    artifact = evaluation.to_artifact([result], model_info={})
    assert artifact["summary"] == {**artifact["summary"], "cases": 1, "passed": 0}
    json.dumps(artifact, ensure_ascii=False)


def test_a_missing_idea_is_a_case_error_not_a_crash(monkeypatch):
    case = EvalCase.model_validate({"id": "c", "criterion": "tools", "client_id": "acme", "description": "d",
                                    "turns": [{"message": "cambiá la {idea_1}"}]})
    result = run_case(graph=None, case=case)
    assert result.turns[0].error.startswith("LookupError")
    assert not result.passed
