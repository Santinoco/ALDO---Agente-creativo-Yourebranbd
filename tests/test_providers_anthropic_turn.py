"""Turnos con tools en Claude: conversión del historial y lectura de la respuesta.

Las respuestas se arman con los tipos del SDK (anthropic.types.Message), así el
test lee lo mismo que leería de la API.
"""

import json
from types import SimpleNamespace

import pytest
from anthropic.types import Message as SDKMessage

from yourebrand.core.providers import AnthropicProvider, Message
from yourebrand.core.providers import anthropic_provider
from yourebrand.core.tools import ToolCall, ToolResult

DECLARATION = {
    "name": "get_brand_info",
    "description": "Información de la marca",
    "parameters": {"type": "object", "properties": {}},
}
THINKING = {"type": "thinking", "thinking": "", "signature": "firma-opaca"}


def sdk_response(*content, stop_reason="end_turn"):
    return SDKMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": list(content), "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    })


class FakeAnthropicClient:
    def __init__(self, api_key, max_retries):
        self.calls = []
        self.response = sdk_response({"type": "text", "text": "hola"})
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(anthropic_provider, "Anthropic", FakeAnthropicClient)
    return AnthropicProvider("sk-test")


def turn(provider, messages, tools=(DECLARATION,)):
    return provider.generate_turn(
        model="claude-haiku-5-5", system="s", messages=messages, tools=list(tools), max_tokens=300,
        timeout_seconds=45,
    )


def sent(provider):
    return provider._client.calls[0]


def test_tools_are_declared_and_the_model_chooses(provider):
    turn(provider, [Message.user("hola")])

    request = sent(provider)
    assert request["tools"] == [
        {"name": "get_brand_info", "description": "Información de la marca",
         "input_schema": {"type": "object", "properties": {}}}
    ]
    # Sin tool_choice: queda en "auto" (forzar una tool da 400 en los modelos actuales).
    assert "tool_choice" not in request
    assert request["max_tokens"] == 300 and request["timeout"] == 45
    assert request["messages"] == [{"role": "user", "content": "hola"}]


def test_text_reply(provider):
    result = turn(provider, [Message.user("hola")])

    assert result.message.text == "hola" and result.message.tool_calls == []
    assert (result.input_tokens, result.output_tokens) == (120, 30)
    assert result.model == "claude-haiku-5-5" and not result.truncated and not result.refused


def test_tool_request_is_read_and_thinking_is_not_shown(provider):
    provider._client.response = sdk_response(
        THINKING,
        {"type": "tool_use", "id": "toolu_1", "name": "get_brand_info", "input": {}},
        stop_reason="tool_use",
    )

    message = turn(provider, [Message.user("ideas")]).message

    assert message.text is None
    assert message.tool_calls == [ToolCall(id="toolu_1", tool_name="get_brand_info", arguments={})]
    assert message.provider == "anthropic"


def test_history_with_tools_is_replayed_with_thinking_intact(provider):
    """Los bloques thinking tienen que volver idénticos; los resultados, juntos en un mensaje user."""
    provider._client.response = sdk_response(
        THINKING,
        {"type": "tool_use", "id": "toolu_1", "name": "get_brand_info", "input": {}},
        {"type": "tool_use", "id": "toolu_2", "name": "get_previous_posts", "input": {}},
        stop_reason="tool_use",
    )
    model_message = turn(provider, [Message.user("ideas")]).message
    # El historial pasa por el checkpointer como JSON: la firma tiene que sobrevivir.
    model_message = Message.model_validate_json(model_message.model_dump_json())
    history = [
        Message.user("ideas"),
        model_message,
        Message.tool(model_message.tool_calls[0], ToolResult(ok=True, content="<marca>Tambor</marca>")),
        Message.tool(model_message.tool_calls[1], ToolResult(ok=True, content="sin posts")),
    ]
    provider._client.calls.clear()

    turn(provider, history)

    messages = sent(provider)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0] == THINKING
    assert [b["id"] for b in messages[1]["content"][1:]] == ["toolu_1", "toolu_2"]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "<marca>Tambor</marca>"},
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "sin posts"},
    ]


def test_a_failed_tool_goes_back_as_an_error_with_details(provider):
    call = ToolCall(id="toolu_1", tool_name="generate_ideas", arguments={"count": 99})
    failed = ToolResult(ok=False, content="Argumentos inválidos",
                        metadata={"error_type": "validation_error", "details": [{"loc": ["count"]}],
                                  "latency_ms": 3})
    history = [Message.user("ideas"), Message(role="model", tool_calls=[call]), Message.tool(call, failed)]

    turn(provider, history)

    block = sent(provider)["messages"][2]["content"][0]
    assert block["is_error"] is True
    payload = json.loads(block["content"])
    assert payload == {"error": "Argumentos inválidos", "error_type": "validation_error",
                       "details": [{"loc": ["count"]}]}


def test_a_gemini_message_is_rebuilt_from_neutral_fields(provider):
    """Una sesión que empezó con Gemini sigue con Claude: la respuesta original de Gemini se ignora."""
    call = ToolCall(id="local-abc", tool_name="get_brand_info")
    gemini_message = Message(role="model", text="Miro la marca.", tool_calls=[call],
                             provider_raw={"role": "model", "parts": []}, provider="gemini")
    history = [Message.user("ideas"), gemini_message, Message.tool(call, ToolResult(ok=True, content="ok"))]

    turn(provider, history)

    assistant = sent(provider)["messages"][1]
    assert assistant["content"] == [
        {"type": "text", "text": "Miro la marca."},
        {"type": "tool_use", "id": "local-abc", "name": "get_brand_info", "input": {}},
    ]


def test_truncated_and_refused_turns_are_flagged(provider):
    provider._client.response = sdk_response({"type": "text", "text": "a medias"}, stop_reason="max_tokens")
    assert turn(provider, [Message.user("hola")]).truncated

    provider._client.response = sdk_response({"type": "text", "text": ""}, stop_reason="refusal")
    assert turn(provider, [Message.user("hola")]).refused
