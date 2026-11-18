"""Ruteo por tarea: la edición de video va siempre a Claude; el resto, al proveedor de la conversación."""

import logging
from functools import partial
from types import SimpleNamespace

import pytest

from yourebrand.core import errors, llm, providers
from yourebrand.core.providers import Message, ProviderResult, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tenancy import TenantContext

TENANT = TenantContext("acme")


class StubProvider:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResult(text="ok", model=kwargs["model"], input_tokens=10, output_tokens=2,
                              stop_reason="end_turn", truncated=False)

    def generate_turn(self, **kwargs):
        self.calls.append(kwargs)
        return TurnResult(message=Message(role="model", text="ok"), model=kwargs["model"], input_tokens=10,
                          output_tokens=2, stop_reason="end_turn", truncated=False)


def settings(model_video_edit="claude-opus-5-5"):
    return SimpleNamespace(
        model_generation="gemini-gen", model_fast="gemini-fast", model_video_edit=model_video_edit,
        llm_timeout_seconds=30, llm_max_retries=0, video_edit_timeout_seconds=600,
        video_edit_max_output_tokens=16000, video_edit_max_input_tokens=60000,
    )


@pytest.fixture
def stubs(monkeypatch):
    """Un proveedor simulado por nombre; None es el de la conversación (Gemini en estos tests)."""
    by_name = {None: StubProvider("gemini"), "anthropic": StubProvider("anthropic")}
    monkeypatch.setattr(llm, "get_provider", lambda name=None: by_name[name])
    monkeypatch.setattr(llm, "get_settings", settings)
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return by_name


def test_video_edit_goes_to_claude_with_its_model(stubs, caplog):
    with caplog.at_level(logging.INFO, logger=llm.__name__):
        response = llm.complete(system="s", prompt="p", tenant=TENANT, run_id="r1", tier="video_edit")

    assert response.provider == "anthropic"
    assert response.model == "claude-opus-5-5"
    assert stubs["anthropic"].calls[0]["model"] == "claude-opus-5-5"
    assert stubs[None].calls == []
    assert "provider=anthropic model=claude-opus-5-5" in caplog.text


@pytest.mark.parametrize(("tier", "model"), [("generation", "gemini-gen"), ("fast", "gemini-fast")])
def test_other_tasks_stay_with_the_conversation_provider(stubs, tier, model):
    response = llm.complete(system="s", prompt="p", tenant=TENANT, run_id="r1", tier=tier)

    assert (response.provider, response.model) == ("gemini", model)
    assert stubs["anthropic"].calls == []


def test_turns_are_routed_too(stubs):
    llm.complete_turn(system="s", messages=[Message.user("hola")], tools=[], tenant=TENANT, run_id="r1",
                      tier="video_edit")

    assert stubs["anthropic"].calls[0]["model"] == "claude-opus-5-5"


def test_video_edit_disabled_fails_clearly(stubs, monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: settings(model_video_edit=None))

    with pytest.raises(errors.ProviderConfigurationError, match="MODEL_VIDEO_EDIT"):
        llm.complete(system="s", prompt="p", tenant=TENANT, run_id="r1", tier="video_edit")
    assert stubs["anthropic"].calls == []


def test_get_provider_builds_each_provider_once(monkeypatch):
    built = []

    def fake_build(name, config):
        built.append(name)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(providers, "build_provider", fake_build)
    monkeypatch.setattr(providers, "get_settings", lambda: SimpleNamespace(llm_provider="gemini"))
    providers.get_provider.cache_clear()
    try:
        conversation = providers.get_provider()
        video = providers.get_provider("anthropic")

        assert providers.get_provider("anthropic") is video
        assert providers.get_provider() is conversation
        assert (conversation.name, video.name) == ("gemini", "anthropic")
        assert built == ["gemini", "anthropic"]
    finally:
        # El caché es del proceso: se limpia para no dejarle estos falsos a otros tests.
        providers.get_provider.cache_clear()
