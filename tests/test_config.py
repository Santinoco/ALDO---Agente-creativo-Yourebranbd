import pytest
from pydantic import ValidationError

from yourebrand.core.config import Settings

REQUIRED = {
    "ANTHROPIC_API_KEY": "sk-test-123",
    "MODEL_GENERATION": "model-gen",
    "MODEL_FAST": "model-fast",
}
OPTIONAL = ["APP_ENV", "QDRANT_URL", "QDRANT_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "NOTION_API_KEY"]


@pytest.fixture
def clean_env(monkeypatch):
    """Aísla el test del entorno real: sin variables heredadas."""
    for name in [*REQUIRED, *OPTIONAL]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def load(**kwargs) -> Settings:
    # _env_file=None: los tests nunca leen el .env real del desarrollador.
    return Settings(_env_file=None, **kwargs)


def test_loads_from_environment(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    settings = load()

    assert settings.app_env == "development"
    assert settings.model_generation == "model-gen"
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-123"
    assert settings.qdrant_url is None


def test_missing_required_key_fails(clean_env):
    with pytest.raises(ValidationError):
        load()


def test_secrets_are_not_printed(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    assert "sk-test-123" not in repr(load())


def test_invalid_app_env_fails(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("APP_ENV", "staging")

    with pytest.raises(ValidationError):
        load()
