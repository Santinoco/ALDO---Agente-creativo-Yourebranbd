from yourebrand.core.errors import ProviderConfigurationError
from yourebrand.creative import cli
from yourebrand.creative.schemas import Idea

IDEA = Idea(
    title="Idea 1",
    format="reel",
    pillar="Cómo funciona",
    description="Descripción de la idea.",
    rationale="Aprovecha un diferencial de la marca.",
)


def test_cli_prints_ideas_and_run_id(monkeypatch, capsys):
    calls = []

    def fake_generate_ideas(tenant, content_type, count, *, run_id):
        calls.append((tenant.client_id, content_type, count, run_id))
        return [IDEA]

    monkeypatch.setattr(cli, "generate_ideas", fake_generate_ideas)

    assert cli.main(["tambor", "ugc", "1"]) == 0

    client_id, content_type, count, run_id = calls[0]
    assert (client_id, content_type, count) == ("tambor", "ugc", 1)
    out = capsys.readouterr().out
    assert f"run_id={run_id}" in out
    assert "1. Idea 1" in out


def test_cli_reports_llm_error_without_traceback(monkeypatch, capsys):
    def failing_generate_ideas(*args, **kwargs):
        raise ProviderConfigurationError("anthropic: clave, permiso o modelo inválido")

    monkeypatch.setattr(cli, "generate_ideas", failing_generate_ideas)

    assert cli.main(["tambor", "ugc"]) == 1
    assert "ProviderConfigurationError" in capsys.readouterr().err


def test_cli_reports_invalid_client_id(capsys):
    assert cli.main(["../otro", "ugc"]) == 1
    assert "InvalidClientIdError" in capsys.readouterr().err
