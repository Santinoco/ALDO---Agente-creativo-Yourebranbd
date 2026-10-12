import logging
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from yourebrand.core import errors, llm
from yourebrand.core.providers import ProviderResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tenancy import TenantContext

TENANT = TenantContext("acme")


class StubProvider:
    """Reemplaza al proveedor real: guarda lo que recibe y devuelve una respuesta fija."""

    name = "stub"

    def __init__(self):
        self.calls = []
        self.text = "hola mundo"
        self.truncated = False
        self.refused = False
        self.errors = []  # errores a lanzar, uno por llamada, antes de responder bien

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return ProviderResult(
            text=self.text,
            model=kwargs["model"],
            input_tokens=10,
            output_tokens=5,
            stop_reason="max_tokens" if self.truncated else "end_turn",
            truncated=self.truncated,
            refused=self.refused,
        )


@pytest.fixture
def provider(monkeypatch):
    """Los tests nunca llaman a una API real ni leen la configuración real."""
    stub = StubProvider()
    monkeypatch.setattr(llm, "get_provider", lambda: stub)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(
            model_generation="model-gen", model_fast="model-fast", llm_timeout_seconds=30, llm_max_retries=2
        ),
    )
    # Los reintentos no esperan de verdad.
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


def test_complete_returns_text_and_usage(provider):
    response = llm.complete(system="sos un asistente", prompt="hola", tenant=TENANT, run_id="run-1")

    assert response.text == "hola mundo"
    assert response.provider == "stub"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.latency_ms >= 0
    assert response.run_id == "run-1"


def test_complete_passes_request_to_provider(provider):
    llm.complete(system="sos un asistente", prompt="hola", tenant=TENANT, run_id="run-1", max_tokens=300)

    assert provider.calls[0] == {
        "model": "model-gen",
        "system": "sos un asistente",
        "prompt": "hola",
        "max_tokens": 300,
        "timeout_seconds": 30,
    }


@pytest.mark.parametrize(("tier", "model"), [("generation", "model-gen"), ("fast", "model-fast")])
def test_tier_selects_model(provider, tier, model):
    response = llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1", tier=tier)

    assert provider.calls[0]["model"] == model
    assert response.model == model


def test_logs_usage_without_prompt_content(provider, caplog):
    with caplog.at_level(logging.INFO, logger=llm.__name__):
        llm.complete(system="secreto de marca", prompt="dato confidencial", tenant=TENANT, run_id="run-1")

    assert "run_id=run-1" in caplog.text
    assert "client_id=acme" in caplog.text
    assert "provider=stub" in caplog.text
    assert "input_tokens=10" in caplog.text
    assert "latency_ms=" in caplog.text
    assert "secreto de marca" not in caplog.text
    assert "dato confidencial" not in caplog.text


def test_warns_when_response_is_truncated(provider, caplog):
    provider.truncated = True

    with caplog.at_level(logging.WARNING, logger=llm.__name__):
        llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    assert "respuesta cortada" in caplog.text


def test_new_run_id_is_unique():
    assert llm.new_run_id() != llm.new_run_id()


def test_single_attempt_when_nothing_fails(provider, caplog):
    with caplog.at_level(logging.INFO, logger=llm.__name__):
        response = llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    assert response.attempts == 1
    assert "attempts=1" in caplog.text


def test_retries_transient_errors_and_reports_attempts(provider, caplog):
    provider.errors = [errors.TransientProviderError("caído"), errors.ProviderTimeoutError("lento")]

    with caplog.at_level(logging.INFO, logger=llm.__name__):
        response = llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    assert response.text == "hola mundo"
    assert response.attempts == 3
    assert len(provider.calls) == 3
    assert "llm_retry run_id=run-1 client_id=acme attempt=1 error=TransientProviderError" in caplog.text
    assert "llm_retry run_id=run-1 client_id=acme attempt=2 error=ProviderTimeoutError" in caplog.text
    assert "attempts=3" in caplog.text


def test_fails_when_retries_are_exhausted(provider, caplog):
    provider.errors = [errors.TransientProviderError("caído")] * 3

    with caplog.at_level(logging.ERROR, logger=llm.__name__), pytest.raises(errors.TransientProviderError):
        llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    # 1 intento + llm_max_retries=2 reintentos.
    assert len(provider.calls) == 3
    assert "llm_call_failed run_id=run-1 client_id=acme provider=stub error=TransientProviderError" in caplog.text


@pytest.mark.parametrize(
    "error", [errors.InvalidRequestError("mal armado"), errors.ProviderConfigurationError("clave inválida")]
)
def test_non_transient_errors_are_not_retried(provider, error):
    provider.errors = [error]

    with pytest.raises(type(error)):
        llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    assert len(provider.calls) == 1


def test_retry_logs_do_not_carry_the_error_message(provider, caplog):
    provider.errors = [errors.TransientProviderError("dato confidencial")] * 3

    with caplog.at_level(logging.INFO, logger=llm.__name__), pytest.raises(errors.TransientProviderError):
        llm.complete(system="s", prompt="p", tenant=TENANT, run_id="run-1")

    assert "dato confidencial" not in caplog.text


class Note(BaseModel):
    """Schema mínimo para probar la salida estructurada."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)


def complete_note(**overrides):
    return llm.complete_structured(
        system="sos un asistente", prompt="hola", schema=Note, tenant=TENANT, run_id="run-1", **overrides
    )


def test_structured_returns_validated_data_and_usage(provider):
    provider.text = '{"title": "Primera idea"}'

    result = complete_note()

    assert result.data == Note(title="Primera idea")
    assert result.response.provider == "stub"
    assert result.response.input_tokens == 10
    assert result.response.run_id == "run-1"


def test_structured_sends_standard_json_schema_to_provider(provider):
    provider.text = '{"title": "Primera idea"}'

    complete_note(max_tokens=300)

    assert provider.calls[0] == {
        "model": "model-gen",
        "system": "sos un asistente",
        "prompt": "hola",
        "max_tokens": 300,
        "timeout_seconds": 30,
        "output_schema": Note.model_json_schema(),
    }


def test_structured_fails_when_model_refuses(provider):
    provider.refused = True

    with pytest.raises(llm.RefusedOutputError):
        complete_note()


def test_structured_fails_when_response_is_truncated(provider):
    provider.text = '{"title": "Prim'
    provider.truncated = True

    with pytest.raises(llm.TruncatedOutputError):
        complete_note()


@pytest.mark.parametrize(
    "text",
    ["no es json", '{"title": ""}', '{"title": "ok", "extra": "dato"}', "{}"],
)
def test_structured_fails_when_output_does_not_match_schema(provider, text):
    provider.text = text

    with pytest.raises(llm.InvalidOutputError):
        complete_note()


def test_structured_error_names_fields_without_content(provider):
    provider.text = '{"title": "", "extra": "dato confidencial"}'

    with pytest.raises(llm.InvalidOutputError) as error:
        complete_note()

    assert "title" in str(error.value)
    assert "run_id=run-1" in str(error.value)
    assert "dato confidencial" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_output_errors_share_a_base_class():
    for error in (llm.RefusedOutputError, llm.TruncatedOutputError, llm.InvalidOutputError):
        assert issubclass(error, llm.LLMOutputError)
    assert issubclass(llm.LLMOutputError, errors.LLMError)


def test_output_errors_are_not_retried(provider):
    provider.refused = True

    with pytest.raises(llm.RefusedOutputError):
        complete_note()

    assert len(provider.calls) == 1
