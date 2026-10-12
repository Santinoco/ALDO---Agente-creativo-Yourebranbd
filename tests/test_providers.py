from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from yourebrand.core import errors, providers
from yourebrand.core.providers import AnthropicProvider, build_provider


class FakeAnthropic:
    """Reemplaza al SDK de Anthropic: guarda lo que recibe y devuelve una respuesta fija."""

    stop_reason = "end_turn"

    def __init__(self, api_key, max_retries):
        self.api_key = api_key
        self.max_retries = max_retries
        self.calls = []
        self.error = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
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
    return provider.generate(
        model="model-gen", system="sos un asistente", prompt="hola", max_tokens=300, timeout_seconds=45
    )


def test_anthropic_sends_request_in_sdk_format(fake_sdk):
    provider = AnthropicProvider("sk-test")

    generate(provider)

    assert provider._client.calls[0] == {
        "model": "model-gen",
        "max_tokens": 300,
        "system": "sos un asistente",
        "messages": [{"role": "user", "content": "hola"}],
        "timeout": 45,
    }


def test_anthropic_disables_sdk_retries(fake_sdk):
    assert AnthropicProvider("sk-test")._client.max_retries == 0


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

    provider.generate(
        model="model-gen", system="s", prompt="p", max_tokens=300, timeout_seconds=45, output_schema=schema
    )

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

    provider.generate(
        model="model-gen", system="s", prompt="p", max_tokens=300, timeout_seconds=45, output_schema=schema
    )

    sent = provider._client.calls[0]["output_config"]["format"]["schema"]
    assert "minLength" not in sent["properties"]["title"]


SDK_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(error_class, status, headers=None):
    """Excepción real del SDK para una respuesta HTTP con ese código."""
    response = httpx2.Response(status, request=SDK_REQUEST, headers=headers)
    return error_class("detalle del proveedor", response=response, body=None)


def generate_failing_with(sdk_error):
    provider = AnthropicProvider("sk-test")
    provider._client.error = sdk_error
    return generate(provider)


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (anthropic.APITimeoutError(SDK_REQUEST), errors.ProviderTimeoutError),
        (anthropic.APIConnectionError(request=SDK_REQUEST), errors.TransientProviderError),
        (status_error(anthropic.RateLimitError, 429), errors.RateLimitError),
        (status_error(anthropic.InternalServerError, 500), errors.TransientProviderError),
        (status_error(anthropic.InternalServerError, 529), errors.TransientProviderError),
        (status_error(anthropic.APIStatusError, 408), errors.TransientProviderError),
        (status_error(anthropic.AuthenticationError, 401), errors.ProviderConfigurationError),
        (status_error(anthropic.PermissionDeniedError, 403), errors.ProviderConfigurationError),
        (status_error(anthropic.NotFoundError, 404), errors.ProviderConfigurationError),
        (status_error(anthropic.BadRequestError, 400), errors.InvalidRequestError),
        (status_error(anthropic.UnprocessableEntityError, 422), errors.InvalidRequestError),
        (anthropic.APIError("raro", SDK_REQUEST, body=None), errors.LLMError),
    ],
)
def test_anthropic_translates_sdk_errors(fake_sdk, sdk_error, expected):
    with pytest.raises(errors.LLMError) as error:
        generate_failing_with(sdk_error)

    # Tipo exacto: un timeout no puede quedar como falla temporal genérica.
    assert type(error.value) is expected
    assert error.value.__cause__ is sdk_error


def test_anthropic_rate_limit_carries_retry_after(fake_sdk):
    with pytest.raises(errors.RateLimitError) as error:
        generate_failing_with(status_error(anthropic.RateLimitError, 429, headers={"retry-after": "7"}))

    assert error.value.retry_after_seconds == 7.0


@pytest.mark.parametrize("headers", [None, {"retry-after": "pronto"}])
def test_anthropic_rate_limit_without_usable_retry_after(fake_sdk, headers):
    with pytest.raises(errors.RateLimitError) as error:
        generate_failing_with(status_error(anthropic.RateLimitError, 429, headers=headers))

    assert error.value.retry_after_seconds is None


def test_anthropic_error_message_does_not_carry_sdk_detail(fake_sdk):
    with pytest.raises(errors.InvalidRequestError) as error:
        generate_failing_with(status_error(anthropic.BadRequestError, 400))

    assert "detalle del proveedor" not in str(error.value)
    assert "HTTP 400" in str(error.value)


def test_build_provider_uses_configured_provider(fake_sdk):
    settings = SimpleNamespace(llm_provider="anthropic", anthropic_api_key=SecretStr("sk-test"))

    provider = build_provider(settings)

    assert provider.name == "anthropic"
    assert provider._client.api_key == "sk-test"
