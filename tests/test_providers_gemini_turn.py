"""Turnos con tools en Gemini: conversión del historial y lectura de la respuesta."""

import json
from types import SimpleNamespace

import pytest
from google.genai import types as genai_types

from yourebrand.core import errors
from yourebrand.core.providers import AnthropicProvider, GeminiProvider, Message
from yourebrand.core.providers import anthropic_provider, gemini_provider
from yourebrand.core.tools import ToolCall, ToolResult

# Firma con bytes que no son UTF-8 válido: si se guardara como texto, se rompería.
SIGNATURE = bytes([0xFF, 0xFE, 0x00, 0x80, 0x41])

DECLARATION = {
    "name": "get_brand_info",
    "description": "Información de la marca",
    "parameters": {"type": "object", "properties": {}},
}


def model_content(*parts):
    return genai_types.Content(role="model", parts=list(parts))


def call_part(name="get_brand_info", call_id="call-1", args=None, signature=None):
    return genai_types.Part(
        function_call=genai_types.FunctionCall(id=call_id, name=name, args=args or {}),
        thought_signature=signature,
    )


def sdk_response(content, finish_reason=genai_types.FinishReason.STOP):
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=content, finish_reason=finish_reason)],
        model_version="gemini-test-001",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )


class FakeGeminiClient:
    def __init__(self, api_key, http_options):
        self.calls = []
        self.response = sdk_response(model_content(genai_types.Part(text="hola")))
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(gemini_provider, "GeminiClient", FakeGeminiClient)
    return GeminiProvider("gm-test")


def turn(provider, messages, tools=(DECLARATION,)):
    return provider.generate_turn(
        model="model-gen", system="s", messages=messages, tools=list(tools), max_tokens=300, timeout_seconds=45
    )


def sent_contents(provider):
    return provider._client.calls[0]["contents"]


def test_declarations_go_as_function_declarations_in_auto_mode(provider):
    turn(provider, [Message.user("hola")])

    config = provider._client.calls[0]["config"]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "get_brand_info"
    assert declaration.parameters_json_schema == {"type": "object", "properties": {}}
    assert config.tool_config.function_calling_config.mode == genai_types.FunctionCallingConfigMode.AUTO
    # Las tools las ejecuta el grafo, nunca el SDK.
    assert config.automatic_function_calling.disable is True


def test_without_tools_no_tool_config_is_sent(provider):
    turn(provider, [Message.user("hola")], tools=())

    assert provider._client.calls[0]["config"].tools is None


def test_user_message_is_a_user_text_content(provider):
    turn(provider, [Message.user("hola")])

    [content] = sent_contents(provider)
    assert content.role == "user"
    assert content.parts[0].text == "hola"


def test_model_message_is_replayed_identical_from_provider_raw(provider):
    # Así viajan de vuelta las firmas del razonamiento, en la parte que estén.
    original = model_content(call_part(signature=SIGNATURE))
    provider._client.response = sdk_response(original)
    first = turn(provider, [Message.user("¿qué vende la marca?")])

    provider._client.calls.clear()
    turn(provider, [Message.user("¿qué vende la marca?"), first.message])

    assert sent_contents(provider)[1] == original


def test_provider_raw_survives_a_json_round_trip(provider):
    # Simula el checkpointer: el historial se guarda como JSON y se vuelve a cargar.
    provider._client.response = sdk_response(model_content(call_part(signature=SIGNATURE)))
    message = turn(provider, [Message.user("hola")]).message

    reloaded = Message.model_validate(json.loads(message.model_dump_json()))
    provider._client.calls.clear()
    turn(provider, [Message.user("hola"), reloaded])

    assert sent_contents(provider)[1].parts[0].thought_signature == SIGNATURE


def test_model_message_without_raw_is_built_from_neutral_fields(provider):
    message = Message(
        role="model",
        text="Busco la marca",
        tool_calls=[ToolCall(id="call-9", tool_name="get_brand_info", arguments={"x": 1})],
    )

    turn(provider, [Message.user("hola"), message])

    content = sent_contents(provider)[1]
    assert content.role == "model"
    assert content.parts[0].text == "Busco la marca"
    assert content.parts[1].function_call.id == "call-9"
    assert content.parts[1].function_call.args == {"x": 1}


def test_consecutive_tool_results_go_together_as_function_responses(provider):
    ok_call = ToolCall(id="call-1", tool_name="get_brand_info")
    bad_call = ToolCall(id="call-2", tool_name="generate_ideas")
    messages = [
        Message.user("hola"),
        Message(role="model", tool_calls=[ok_call, bad_call]),
        Message.tool(ok_call, ToolResult(ok=True, content="Tambor es una lavandería", metadata={"latency_ms": 3})),
        Message.tool(
            bad_call,
            ToolResult(
                ok=False,
                content="Argumentos inválidos",
                metadata={"error_type": "validation_error", "details": [{"loc": ["count"]}]},
            ),
        ),
    ]

    turn(provider, messages)

    contents = sent_contents(provider)
    assert len(contents) == 3  # usuario, modelo y un solo mensaje con los dos resultados
    results = contents[2]
    assert results.role == "user"
    ok, bad = (part.function_response for part in results.parts)
    assert (ok.id, ok.name) == ("call-1", "get_brand_info")
    # La metadata de runtime no le llega al modelo.
    assert ok.response == {"result": "Tambor es una lavandería"}
    assert bad.response == {
        "error": "Argumentos inválidos",
        "error_type": "validation_error",
        "details": [{"loc": ["count"]}],
    }


def test_response_tool_calls_become_tool_calls(provider):
    provider._client.response = sdk_response(
        model_content(
            genai_types.Part(text="pensando...", thought=True),
            genai_types.Part(text="Voy a buscar la marca."),
            call_part(args={"detail": "corta"}),
        )
    )

    result = turn(provider, [Message.user("hola")])

    assert result.message.role == "model"
    # El razonamiento no es respuesta para la persona.
    assert result.message.text == "Voy a buscar la marca."
    [call] = result.message.tool_calls
    assert (call.id, call.tool_name, call.arguments) == ("call-1", "get_brand_info", {"detail": "corta"})
    assert result.input_tokens == 10
    assert result.malformed_tool_call is False


def test_call_without_id_gets_a_local_id_that_is_not_sent_back(provider):
    provider._client.response = sdk_response(model_content(call_part(call_id=None)))
    first = turn(provider, [Message.user("hola")])
    [call] = first.message.tool_calls
    assert call.id.startswith("local-")

    provider._client.calls.clear()
    turn(provider, [Message.user("hola"), first.message, Message.tool(call, ToolResult(ok=True, content="ok"))])

    # Gemini nunca generó ese id: se le devuelve el resultado sin id.
    assert sent_contents(provider)[2].parts[0].function_response.id is None


def test_malformed_tool_call_is_flagged(provider):
    provider._client.response = sdk_response(
        model_content(genai_types.Part(text="")), finish_reason=genai_types.FinishReason.MALFORMED_FUNCTION_CALL
    )

    result = turn(provider, [Message.user("hola")])

    assert result.malformed_tool_call is True
    assert result.stop_reason == "MALFORMED_FUNCTION_CALL"


def test_anthropic_does_not_support_tools_yet(monkeypatch):
    monkeypatch.setattr(anthropic_provider, "Anthropic", lambda api_key, max_retries: None)

    with pytest.raises(errors.ProviderConfigurationError):
        AnthropicProvider("sk-test").generate_turn(
            model="m", system="s", messages=[], tools=[], max_tokens=10, timeout_seconds=30
        )
