import pytest
from pydantic import ValidationError

from yourebrand.core.config import Settings

REQUIRED = {
    "GEMINI_API_KEY": "sk-test-123",
    "MODEL_GENERATION": "model-gen",
    "MODEL_FAST": "model-fast",
}
OPTIONAL = [
    "LLM_PROVIDER",
    "ANTHROPIC_API_KEY",
    "APP_ENV",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "NOTION_API_KEY",
]


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
    assert settings.llm_provider == "gemini"
    assert settings.model_generation == "model-gen"
    assert settings.gemini_api_key.get_secret_value() == "sk-test-123"
    assert settings.anthropic_api_key is None
    assert settings.qdrant_url is None


def test_anthropic_provider_needs_only_its_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    clean_env.setenv("MODEL_GENERATION", "model-gen")
    clean_env.setenv("MODEL_FAST", "model-fast")

    settings = load()

    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert settings.gemini_api_key is None


@pytest.mark.parametrize(
    ("provider", "only_key"),
    [("gemini", "ANTHROPIC_API_KEY"), ("anthropic", "GEMINI_API_KEY")],
)
def test_missing_key_of_chosen_provider_fails(clean_env, provider, only_key):
    # Tener la clave del otro proveedor no alcanza.
    clean_env.setenv("LLM_PROVIDER", provider)
    clean_env.setenv(only_key, "sk-test-123")
    clean_env.setenv("MODEL_GENERATION", "model-gen")
    clean_env.setenv("MODEL_FAST", "model-fast")

    with pytest.raises(ValidationError) as error:
        load()

    assert f"{provider.upper()}_API_KEY" in str(error.value)
    assert "sk-test-123" not in str(error.value)


def test_empty_key_counts_as_missing(clean_env):
    # Así queda la clave al copiar .env.example sin completarla.
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "")
    clean_env.setenv("GEMINI_API_KEY", "sk-test-123")
    clean_env.setenv("MODEL_GENERATION", "model-gen")
    clean_env.setenv("MODEL_FAST", "model-fast")

    with pytest.raises(ValidationError) as error:
        load()

    assert "ANTHROPIC_API_KEY" in str(error.value)


def test_unknown_provider_fails(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError):
        load()


def test_missing_required_key_fails(clean_env):
    with pytest.raises(ValidationError):
        load()


def test_secrets_are_not_printed(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    assert "sk-test-123" not in repr(load())


def test_validation_error_does_not_show_secrets(clean_env):
    # Falta MODEL_GENERATION: el error no puede traer los valores que sí llegaron.
    clean_env.setenv("GEMINI_API_KEY", "sk-test-123")

    with pytest.raises(ValidationError) as error:
        load()

    assert "sk-test-123" not in str(error.value)
    assert "model_generation" in str(error.value)


def test_invalid_app_env_fails(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("APP_ENV", "staging")

    with pytest.raises(ValidationError):
        load()


def test_llm_reliability_defaults(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)

    settings = load()

    assert settings.llm_timeout_seconds == 120
    assert settings.llm_max_retries == 2


def test_llm_reliability_reads_environment(clean_env):
    for name, value in REQUIRED.items():
        clean_env.setenv(name, value)
    clean_env.setenv("LLM_TIMEOUT_SECONDS", "45.5")
    clean_env.setenv("LLM_MAX_RETRIES", "0")

    settings = load()

    assert settings.llm_timeout_seconds == 45.5
    assert settings.llm_max_retries == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [("LLM_TIMEOUT_SECONDS", "0"), ("LLM_TIMEOUT_SECONDS", "-5"), ("LLM_MAX_RETRIES", "-1"), ("LLM_MAX_RETRIES", "6")],
)
def test_invalid_llm_reliability_fails(clean_env, name, value):
    for required, required_value in REQUIRED.items():
        clean_env.setenv(required, required_value)
    clean_env.setenv(name, value)

    with pytest.raises(ValidationError):
        load()
