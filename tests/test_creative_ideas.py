import json
import logging
from types import SimpleNamespace

import pytest

from yourebrand.core import llm, tenancy
from yourebrand.core.providers import ProviderResult
from yourebrand.core.tenancy import TenantContext, UnknownClientError
from yourebrand.creative import ideas
from yourebrand.creative.ideas import NotEnoughIdeasError, generate_ideas
from yourebrand.creative.schemas import Idea, IdeaList

ACME = TenantContext("acme")


def idea_json(number):
    return {
        "title": f"Idea {number}",
        "format": "reel",
        "pillar": "Cómo funciona",
        "description": f"Descripción de la idea {number}.",
        "rationale": "Aprovecha un diferencial de la marca.",
    }


class StubProvider:
    """Reemplaza al proveedor real: guarda lo que recibe y devuelve tantas ideas como se le indique."""

    name = "stub"

    def __init__(self):
        self.calls = []
        self.ideas_to_return = 1

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResult(
            text=json.dumps({"ideas": [idea_json(n) for n in range(1, self.ideas_to_return + 1)]}),
            model=kwargs["model"],
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            truncated=False,
        )


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """Dos clientes en una carpeta temporal y un proveedor simulado: sin API ni configuración real."""
    for client_id, brand, post in [
        ("acme", "Acme vende yunques.", "Post del yunque rojo."),
        ("globex", "Globex vende cohetes.", "Post del cohete azul."),
    ]:
        folder = tmp_path / client_id
        folder.mkdir()
        (folder / "brand.md").write_text(brand, encoding="utf-8")
        (folder / "posts.md").write_text(post, encoding="utf-8")

    stub = StubProvider()
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(llm, "get_provider", lambda: stub)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(model_generation="model-gen", model_fast="model-fast"),
    )
    return stub


def test_returns_the_requested_ideas(provider):
    provider.ideas_to_return = 3

    result = generate_ideas(ACME, "ugc", 3)

    assert [idea.title for idea in result] == ["Idea 1", "Idea 2", "Idea 3"]
    assert all(isinstance(idea, Idea) for idea in result)


def test_sends_brand_material_and_request_to_the_model(provider):
    provider.ideas_to_return = 2

    generate_ideas(ACME, "design", 2)

    call = provider.calls[0]
    assert call["model"] == "model-gen"
    assert "Contenido de diseño" in call["system"]
    assert "Acme vende yunques." in call["prompt"]
    assert "Post del yunque rojo." in call["prompt"]
    assert call["prompt"].endswith("Proponé 2 ideas de contenido de diseño para esta marca.")
    assert call["output_schema"] == IdeaList.model_json_schema()


def test_request_never_contains_another_client(provider):
    provider.ideas_to_return = 1

    generate_ideas(ACME, "ugc", 1)

    call = provider.calls[0]
    assert "Globex" not in call["system"] + call["prompt"]
    assert "cohete" not in call["system"] + call["prompt"]


def test_fails_when_the_model_returns_fewer_ideas(provider):
    provider.ideas_to_return = 2

    with pytest.raises(NotEnoughIdeasError):
        generate_ideas(ACME, "ugc", 5)


def test_trims_and_warns_when_the_model_returns_extra_ideas(provider, caplog):
    provider.ideas_to_return = 7

    with caplog.at_level(logging.WARNING, logger=ideas.__name__):
        result = generate_ideas(ACME, "ugc", 5)

    assert len(result) == 5
    assert "llegaron 7 ideas" in caplog.text


def test_uses_the_given_run_id(provider, caplog):
    provider.ideas_to_return = 1

    with caplog.at_level(logging.INFO, logger=llm.__name__):
        generate_ideas(ACME, "ugc", 1, run_id="run-42")

    assert "run_id=run-42" in caplog.text
    assert "client_id=acme" in caplog.text


@pytest.mark.parametrize("count", [0, -1, 21])
def test_invalid_count_fails_without_calling_the_model(provider, count):
    with pytest.raises(ValueError):
        generate_ideas(ACME, "ugc", count)

    assert provider.calls == []


def test_invalid_content_type_fails_without_calling_the_model(provider):
    with pytest.raises(ValueError):
        generate_ideas(ACME, "video", 3)

    assert provider.calls == []


def test_unknown_client_fails_without_calling_the_model(provider):
    with pytest.raises(UnknownClientError):
        generate_ideas(TenantContext("initech"), "ugc", 3)

    assert provider.calls == []
