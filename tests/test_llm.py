import logging
from types import SimpleNamespace

import pytest

from yourebrand.core import llm
from yourebrand.core.providers import ProviderResult
from yourebrand.core.tenancy import TenantContext

TENANT = TenantContext("acme")


class StubProvider:
    """Reemplaza al proveedor real: guarda lo que recibe y devuelve una respuesta fija."""

    name = "stub"

    def __init__(self):
        self.calls = []
        self.truncated = False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResult(
            text="hola mundo",
            model=kwargs["model"],
            input_tokens=10,
            output_tokens=5,
            stop_reason="max_tokens" if self.truncated else "end_turn",
            truncated=self.truncated,
        )


@pytest.fixture
def provider(monkeypatch):
    """Los tests nunca llaman a una API real ni leen la configuración real."""
    stub = StubProvider()
    monkeypatch.setattr(llm, "get_provider", lambda: stub)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(model_generation="model-gen", model_fast="model-fast"),
    )
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
