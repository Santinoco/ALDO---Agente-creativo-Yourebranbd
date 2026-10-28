from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import SecretStr

from yourebrand.core import errors
from yourebrand.core.providers import gemini_provider
from yourebrand.core.providers import GeminiProvider, build_provider


def sdk_response(
    *,
    parts=("hola ", "mundo"),
    finish_reason=genai_types.FinishReason.STOP,
    block_reason=None,
    thoughts=None,
):
    """Respuesta con los tipos reales del SDK, para que el test note si cambian."""
    candidates = None
    if parts is not None:
        candidates = [
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text=p) for p in parts]),
                finish_reason=finish_reason,
            )
        ]
    feedback = None
    if block_reason is not None:
        feedback = genai_types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
    return genai_types.GenerateContentResponse(
        candidates=candidates,
        prompt_feedback=feedback,
        model_version="gemini-test-001",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=thoughts
        ),
    )


class FakeGeminiClient:
    """Reemplaza al cliente de Google: guarda lo que recibe y devuelve una respuesta fija."""

    def __init__(self, api_key, http_options):
        self.api_key = api_key
        self.http_options = http_options
        self.calls = []
        self.error = None
        self.response = sdk_response()
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    monkeypatch.setattr(gemini_provider, "GeminiClient", FakeGeminiClient)


def generate(provider, **kwargs):
    return provider.generate(
        model="model-gen", system="sos un asistente", prompt="hola", max_tokens=300, timeout_seconds=45, **kwargs
    )


def test_gemini_sends_request_in_sdk_format():
    provider = GeminiProvider("gm-test")

    generate(provider)

    call = provider._client.calls[0]
    assert call["model"] == "model-gen"
    assert call["contents"] == "hola"
    assert call["config"].system_instruction == "sos un asistente"
    assert call["config"].max_output_tokens == 300
    assert call["config"].response_json_schema is None


def test_gemini_timeout_goes_in_milliseconds():
    # El contrato habla en segundos y el SDK en milisegundos: 45 s son 45000 ms.
    provider = GeminiProvider("gm-test")

    generate(provider)

    assert provider._client.calls[0]["config"].http_options.timeout == 45000


def test_gemini_rejects_timeout_below_api_minimum():
    # Google responde 400 a un plazo menor a 10 s: se corta antes de llamar.
    provider = GeminiProvider("gm-test")

    with pytest.raises(errors.ProviderConfigurationError):
        provider.generate(model="m", system="s", prompt="p", max_tokens=300, timeout_seconds=5)

    assert provider._client.calls == []


def test_gemini_call_runs_under_the_requested_deadline(monkeypatch):
    # El timeout del SDK no corta la duración total: el plazo lo pone core.
    seen = {}

    def fake_deadline(fn, timeout_seconds, *, label):
        seen["timeout"] = timeout_seconds
        return fn()

    monkeypatch.setattr(gemini_provider, "call_with_deadline", fake_deadline)

    generate(GeminiProvider("gm-test"))

    assert seen["timeout"] == 45


def test_gemini_disables_automatic_function_calling():
    # Las tools las ejecuta el grafo, no el SDK por su cuenta.
    provider = GeminiProvider("gm-test")

    generate(provider)

    assert provider._client.calls[0]["config"].automatic_function_calling.disable is True


def test_gemini_disables_sdk_retries():
    provider = GeminiProvider("gm-test")

    assert provider._client.api_key == "gm-test"
    assert provider._client.http_options.retry_options.attempts == 1


def test_gemini_sends_output_schema_as_json_schema():
    provider = GeminiProvider("gm-test")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string", "minLength": 1}},
        "required": ["title"],
    }

    generate(provider, output_schema=schema)

    config = provider._client.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    # Se manda tal cual: a diferencia de Claude, no hace falta adaptarlo.
    assert config.response_json_schema == schema


def test_gemini_normalizes_response():
    result = generate(GeminiProvider("gm-test"))

    assert result.text == "hola mundo"
    assert result.model == "gemini-test-001"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.stop_reason == "STOP"
    assert result.truncated is False
    assert result.refused is False


def test_gemini_counts_thinking_tokens_as_output():
    provider = GeminiProvider("gm-test")
    provider._client.response = sdk_response(thoughts=7)

    # Se cobran como salida aunque no lleguen en el texto.
    assert generate(provider).output_tokens == 12


def test_gemini_marks_truncated_response():
    provider = GeminiProvider("gm-test")
    provider._client.response = sdk_response(finish_reason=genai_types.FinishReason.MAX_TOKENS)

    assert generate(provider).truncated is True


@pytest.mark.parametrize(
    "finish_reason",
    [genai_types.FinishReason.SAFETY, genai_types.FinishReason.PROHIBITED_CONTENT],
)
def test_gemini_marks_refused_response(finish_reason):
    provider = GeminiProvider("gm-test")
    provider._client.response = sdk_response(finish_reason=finish_reason)

    assert generate(provider).refused is True


def test_gemini_blocked_prompt_is_refused_with_empty_text():
    provider = GeminiProvider("gm-test")
    provider._client.response = sdk_response(parts=None, block_reason=genai_types.BlockedReason.SAFETY)

    result = generate(provider)

    assert result.refused is True
    assert result.text == ""
    assert result.stop_reason == "PROMPT_BLOCKED_SAFETY"


def api_error(error_class, code, *, details=None):
    """Excepción real del SDK, con el cuerpo que manda Google."""
    body = {"error": {"code": code, "message": "detalle del proveedor", "status": "X", "details": details or []}}
    return error_class(code, body)


def generate_failing_with(sdk_error):
    provider = GeminiProvider("gm-test")
    provider._client.error = sdk_error
    return generate(provider)


API_KEY_INVALID = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (api_error(genai_errors.ClientError, 429), errors.RateLimitError),
        (api_error(genai_errors.ClientError, 400, details=API_KEY_INVALID), errors.ProviderConfigurationError),
        (api_error(genai_errors.ClientError, 403), errors.ProviderConfigurationError),
        (api_error(genai_errors.ClientError, 404), errors.ProviderConfigurationError),
        (api_error(genai_errors.ClientError, 408), errors.TransientProviderError),
        (api_error(genai_errors.ServerError, 500), errors.TransientProviderError),
        (api_error(genai_errors.ServerError, 503), errors.TransientProviderError),
        # Google corta con 504 cuando vence el plazo que le manda el SDK.
        (api_error(genai_errors.ServerError, 504), errors.ProviderTimeoutError),
        (api_error(genai_errors.ClientError, 400), errors.InvalidRequestError),
        (httpx.ReadTimeout("lento"), errors.ProviderTimeoutError),
        (httpx.ConnectError("sin red"), errors.TransientProviderError),
    ],
)
def test_gemini_translates_sdk_errors(sdk_error, expected):
    with pytest.raises(errors.LLMError) as error:
        generate_failing_with(sdk_error)

    # Tipo exacto: un timeout no puede quedar como falla temporal genérica.
    assert type(error.value) is expected
    assert error.value.__cause__ is sdk_error


def test_gemini_translates_httpx2_errors_too():
    # El SDK usa httpx2 si está instalado: sus clases son otras aunque se llamen igual.
    httpx2 = pytest.importorskip("httpx2")

    with pytest.raises(errors.ProviderTimeoutError):
        generate_failing_with(httpx2.ReadTimeout("lento"))


def test_gemini_rate_limit_reads_retry_delay_from_body():
    retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]

    with pytest.raises(errors.RateLimitError) as error:
        generate_failing_with(api_error(genai_errors.ClientError, 429, details=retry_info))

    assert error.value.retry_after_seconds == 37.0


def test_gemini_rate_limit_without_retry_delay():
    with pytest.raises(errors.RateLimitError) as error:
        generate_failing_with(api_error(genai_errors.ClientError, 429))

    assert error.value.retry_after_seconds is None


def test_gemini_error_message_does_not_carry_sdk_detail():
    with pytest.raises(errors.InvalidRequestError) as error:
        generate_failing_with(api_error(genai_errors.ClientError, 400))

    assert "detalle del proveedor" not in str(error.value)
    assert "HTTP 400" in str(error.value)


def test_build_provider_builds_gemini():
    settings = SimpleNamespace(llm_provider="gemini", gemini_api_key=SecretStr("gm-test"), anthropic_api_key=None)

    provider = build_provider(settings)

    assert provider.name == "gemini"
    assert provider._client.api_key == "gm-test"


def test_build_provider_without_key_fails():
    settings = SimpleNamespace(llm_provider="gemini", gemini_api_key=None, anthropic_api_key=None)

    with pytest.raises(errors.ProviderConfigurationError):
        build_provider(settings)
