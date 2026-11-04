import logging
import time

import pytest
from pydantic import BaseModel, Field

from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import (
    ExecutionContext,
    RetryableToolError,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
)

CONTEXT = ExecutionContext(tenant=TenantContext("tambor"), run_id="run-test")


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, description="Texto a devolver")


def echo(args, context):
    return ToolResult(ok=True, content=f"{context.tenant.client_id}: {args['text']}")


def registry_with(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def call(name="echo", **arguments):
    return ToolCall(id="call-1", tool_name=name, arguments=arguments)


def test_runs_tool_with_validated_args_and_context():
    registry = registry_with(Tool("echo", "Devuelve el texto", "read", echo, EchoArgs))

    result = registry.run(call(text="hola"), CONTEXT)

    # El client_id sale del contexto, no de los argumentos del modelo.
    assert result.ok is True
    assert result.content == "tambor: hola"
    assert result.metadata["tool"] == "echo"
    assert result.metadata["attempt"] == 1


def test_unknown_tool_is_a_controlled_error():
    result = registry_with().run(call("no-existe"), CONTEXT)

    assert result.ok is False
    assert result.metadata["error_type"] == "unknown_tool"


def test_tool_outside_allowed_kinds_is_not_run():
    ran = []

    def save(args, context):
        ran.append(1)
        return ToolResult(ok=True, content="")

    result = registry_with(Tool("notion", "Guarda", "external_effect", save)).run(call("notion"), CONTEXT)

    assert result.metadata["error_type"] == "forbidden_tool"
    assert ran == []


def test_invalid_arguments_do_not_reach_the_handler():
    registry = registry_with(Tool("echo", "Devuelve el texto", "read", echo, EchoArgs))

    result = registry.run(call(text=""), CONTEXT)

    assert result.ok is False
    assert result.metadata["error_type"] == "validation_error"


def test_slow_tool_times_out_without_blocking():
    tool = Tool("lenta", "Tarda", "read", lambda args, context: time.sleep(5), timeout_s=0.2)
    started = time.perf_counter()

    result = registry_with(tool).run(call("lenta"), CONTEXT)

    assert result.metadata["error_type"] == "timeout"
    assert time.perf_counter() - started < 2


def test_retryable_error_is_retried_until_success():
    attempts = []

    def flaky(args, context):
        attempts.append(1)
        if len(attempts) < 2:
            raise RetryableToolError("falla temporal")
        return ToolResult(ok=True, content="ok")

    tool = Tool("flaky", "A veces falla", "read", flaky, retries=2, backoff_s=0)

    result = registry_with(tool).run(call("flaky"), CONTEXT)

    assert result.ok is True
    assert result.metadata["attempt"] == 2


def test_retries_exhausted():
    def always_fails(args, context):
        raise RetryableToolError("falla temporal")

    tool = Tool("rota", "Siempre falla", "read", always_fails, retries=1, backoff_s=0)

    result = registry_with(tool).run(call("rota"), CONTEXT)

    assert result.metadata["error_type"] == "retry_exhausted"
    assert result.metadata["attempts"] == 2


def test_non_retryable_error_is_not_retried():
    attempts = []

    def broken(args, context):
        attempts.append(1)
        raise ValueError("bug")

    tool = Tool("bug", "Tiene un bug", "read", broken, retries=2, backoff_s=0)

    result = registry_with(tool).run(call("bug"), CONTEXT)

    assert result.metadata["error_type"] == "ValueError"
    assert attempts == [1]


def test_invalid_output_is_a_controlled_error():
    tool = Tool("mal", "Devuelve cualquier cosa", "read", lambda args, context: "no es un ToolResult")

    result = registry_with(tool).run(call("mal"), CONTEXT)

    assert result.metadata["error_type"] == "output_validation_error"


def test_declaration_shows_schema_without_titles():
    declaration = Tool("echo", "Devuelve el texto", "read", echo, EchoArgs).function_declaration()

    assert declaration["name"] == "echo"
    assert "title" not in declaration["parameters"]
    assert "title" not in declaration["parameters"]["properties"]["text"]
    assert declaration["parameters"]["required"] == ["text"]


def test_declarations_only_include_allowed_kinds():
    registry = registry_with(
        Tool("echo", "Lee", "read", echo),
        Tool("notion", "Guarda", "external_effect", echo),
    )

    assert [d["name"] for d in registry.function_declarations()] == ["echo"]


def test_empty_allowed_kinds_means_no_tools():
    registry = registry_with(Tool("echo", "Lee", "read", echo))

    assert registry.function_declarations(allowed_kinds=set()) == []


def test_duplicate_tool_is_rejected():
    registry = registry_with(Tool("echo", "Lee", "read", echo))

    with pytest.raises(ValueError):
        registry.register(Tool("echo", "Otra", "read", echo))


def test_non_idempotent_tool_cannot_retry():
    with pytest.raises(ValueError):
        Tool("notion", "Guarda", "external_effect", echo, retries=1, idempotent=False)


def test_each_run_is_logged_with_run_id_and_client(caplog):
    registry = registry_with(Tool("echo", "Devuelve el texto", "read", echo, EchoArgs))

    with caplog.at_level(logging.INFO, logger="yourebrand.core.tools"):
        registry.run(call(text="dato del cliente"), CONTEXT)

    assert "run_id=run-test" in caplog.text
    assert "client_id=tambor" in caplog.text
    # Los argumentos pueden traer datos del cliente: no se loguean.
    assert "dato del cliente" not in caplog.text
