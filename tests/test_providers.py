from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from yourebrand.core import providers
from yourebrand.core.providers import AnthropicProvider, build_provider


class FakeAnthropic:
    """Reemplaza al SDK de Anthropic: guarda lo que recibe y devuelve una respuesta fija."""

    stop_reason = "end_turn"

    def __init__(self, api_key):
        self.api_key = api_key
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hola "), SimpleNamespace(type="text", text="mundo")],
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason=self.stop_reason,
        )


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setattr(providers, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(FakeAnthropic, "stop_reason", "end_turn")
    return FakeAnthropic


def generate(provider):
    return provider.generate(model="model-gen", system="sos un asistente", prompt="hola", max_tokens=300)


def test_anthropic_sends_request_in_sdk_format(fake_sdk):
    provider = AnthropicProvider("sk-test")

    generate(provider)

    assert provider._client.calls[0] == {
        "model": "model-gen",
        "max_tokens": 300,
        "system": "sos un asistente",
        "messages": [{"role": "user", "content": "hola"}],
    }


def test_anthropic_normalizes_response(fake_sdk):
    result = generate(AnthropicProvider("sk-test"))

    assert result.text == "hola mundo"
    assert result.model == "model-gen"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.truncated is False


def test_anthropic_marks_truncated_response(fake_sdk, monkeypatch):
    monkeypatch.setattr(fake_sdk, "stop_reason", "max_tokens")

    assert generate(AnthropicProvider("sk-test")).truncated is True


def test_anthropic_marks_refused_response(fake_sdk, monkeypatch):
    monkeypatch.setattr(fake_sdk, "stop_reason", "refusal")

    assert generate(AnthropicProvider("sk-test")).refused is True


def test_anthropic_sends_output_schema_as_json_schema_format(fake_sdk):
    provider = AnthropicProvider("sk-test")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }

    provider.generate(model="model-gen", system="s", prompt="p", max_tokens=300, output_schema=schema)

    output_format = provider._client.calls[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"] == {"title": {"type": "string"}}


def test_anthropic_removes_constraints_the_api_does_not_support(fake_sdk):
    provider = AnthropicProvider("sk-test")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string", "minLength": 1}},
        "required": ["title"],
        "additionalProperties": False,
    }

    provider.generate(model="model-gen", system="s", prompt="p", max_tokens=300, output_schema=schema)

    sent = provider._client.calls[0]["output_config"]["format"]["schema"]
    assert "minLength" not in sent["properties"]["title"]


def test_build_provider_uses_configured_provider(fake_sdk):
    settings = SimpleNamespace(llm_provider="anthropic", anthropic_api_key=SecretStr("sk-test"))

    provider = build_provider(settings)

    assert provider.name == "anthropic"
    assert provider._client.api_key == "sk-test"
