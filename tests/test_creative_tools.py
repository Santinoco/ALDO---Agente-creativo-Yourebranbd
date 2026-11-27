"""Tools del Agente Creativo: contenido, validación, errores, aislamiento y uso desde el grafo."""

import json
import logging
from functools import partial
from types import SimpleNamespace

import pytest

from yourebrand.core import llm, tenancy
from yourebrand.core.config import PROJECT_ROOT
from yourebrand.core.providers import Message, ProviderResult, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import MODEL_INPUT_TOKENS_KEY, MODEL_OUTPUT_TOKENS_KEY, ExecutionContext, ToolCall
from yourebrand.creative import tools as creative_tools
from yourebrand.creative.graph import AgentSession, build_graph, run_turn
from yourebrand.creative.tools import IDEAS_METADATA_KEY, NO_POSTS_MESSAGE, build_creative_registry


def make_registry():
    # Plazo fijo: los tests no dependen del .env.
    return build_creative_registry(generation_timeout_s=5, video_edit=False)


def idea_json(number):
    return {
        "title": f"Idea {number}",
        "format": "reel",
        "pillar": "Cómo funciona",
        "description": f"Descripción de la idea {number}.",
        "rationale": "Aprovecha un diferencial de la marca.",
    }


class StubProvider:
    """Proveedor simulado: `generate` para generate_ideas y `generate_turn` para el agente."""

    name = "stub"

    def __init__(self):
        self.generate_calls = []
        self.turn_calls = []
        self.ideas_to_return = 3
        self.turns = []  # TurnResult en orden

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ProviderResult(
            text=json.dumps({"ideas": [idea_json(n) for n in range(1, self.ideas_to_return + 1)]}),
            model=kwargs["model"], input_tokens=100, output_tokens=50, stop_reason="end_turn", truncated=False,
        )

    def generate_turn(self, **kwargs):
        self.turn_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.turns.pop(0)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Usa una carpeta temporal como data_dir, sin leer la configuración real."""
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    return tmp_path


@pytest.fixture
def provider(monkeypatch):
    stub = StubProvider()
    monkeypatch.setattr(llm, "get_provider", lambda *args: stub)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(
            model_generation="model-gen", model_fast="model-fast", llm_timeout_seconds=30, llm_max_retries=0
        ),
    )
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


def write_client(data_dir, client_id, *, brand=None, posts=None):
    folder = data_dir / client_id
    folder.mkdir()
    if brand is not None:
        (folder / "brand.md").write_text(brand, encoding="utf-8")
    if posts is not None:
        (folder / "posts.md").write_text(posts, encoding="utf-8")


def run_tool(name, client_id, arguments=None):
    """Ejecuta una tool como lo hace el grafo: por el registry y con el contexto de la sesión."""
    context = ExecutionContext(tenant=TenantContext(client_id), run_id="run-1")
    return make_registry().run(ToolCall(id="c1", tool_name=name, arguments=arguments or {}), context)


# --- Registry ---------------------------------------------------------------


def test_registry_exposes_creative_tools():
    declarations = {d["name"]: d for d in make_registry().function_declarations()}

    assert list(declarations) == ["get_brand_info", "get_previous_posts", "generate_ideas", "refine_idea", "check_similarity", "load_skill"]
    # Las de lectura no tienen argumentos: el cliente no es algo que el modelo pueda elegir.
    assert declarations["get_brand_info"]["parameters"] == {"type": "object", "properties": {}}
    assert declarations["get_previous_posts"]["parameters"] == {"type": "object", "properties": {}}


def test_generate_ideas_declaration_has_no_client_argument():
    params = {d["name"]: d for d in make_registry().function_declarations()}["generate_ideas"]["parameters"]

    assert set(params["properties"]) == {"content_type", "count", "brief"}
    assert params["required"] == ["content_type"]
    assert params["properties"]["content_type"]["enum"] == ["ugc", "design"]


def test_generation_timeout_covers_every_retry(monkeypatch):
    monkeypatch.setattr(
        creative_tools, "get_settings", lambda: SimpleNamespace(llm_timeout_seconds=60, llm_max_retries=2)
    )
    # 3 intentos de 60 s, 2 esperas de hasta 30 s y 5 s de margen.
    assert creative_tools._generation_timeout_s() == 60 * 3 + 30 * 2 + 5


# --- Tools de lectura ------------------------------------------------------


def test_get_brand_info_returns_brand_between_tags(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme\n")

    result = run_tool("get_brand_info", "acme")

    assert result.ok is True
    assert result.content == "<marca>\nMarca Acme\n</marca>"
    assert result.metadata["chars"] == len("Marca Acme")


def test_get_brand_info_without_brand_is_a_tool_error(data_dir):
    write_client(data_dir, "acme", posts="Post de Acme")

    result = run_tool("get_brand_info", "acme")

    assert result.ok is False
    assert result.metadata["error_type"] == "missing_brand"


def test_get_previous_posts_returns_posts_between_tags(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme", posts="Post de Acme")

    result = run_tool("get_previous_posts", "acme")

    assert result.ok is True
    assert result.content == "<posts_anteriores>\nPost de Acme\n</posts_anteriores>"
    assert result.metadata["has_posts"] is True


def test_get_previous_posts_without_posts_is_not_an_error(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme")

    result = run_tool("get_previous_posts", "acme")

    assert result.ok is True
    assert result.content == NO_POSTS_MESSAGE
    assert result.metadata["has_posts"] is False


def test_posts_do_not_require_brand(data_dir):
    # Cada tool lee solo lo suyo: que falte la marca no rompe la lectura de posts.
    write_client(data_dir, "acme", posts="Post de Acme")

    assert run_tool("get_previous_posts", "acme").ok is True


def test_read_tools_read_only_the_session_client(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme", posts="Post de Acme")
    write_client(data_dir, "globex", brand="Marca Globex", posts="Post de Globex")

    brand = run_tool("get_brand_info", "acme")
    posts = run_tool("get_previous_posts", "acme")

    assert "Globex" not in brand.content
    assert "Globex" not in posts.content


def test_client_id_in_model_arguments_is_ignored(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme")
    write_client(data_dir, "globex", brand="Marca Globex")

    result = run_tool("get_brand_info", "acme", arguments={"client_id": "globex"})

    assert "Marca Acme" in result.content
    assert "Globex" not in result.content


def test_bundled_sample_client(monkeypatch):
    """Las tools de lectura funcionan con el cliente ficticio del repo tal como está."""
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=PROJECT_ROOT / "data"))

    brand = run_tool("get_brand_info", "tambor")
    posts = run_tool("get_previous_posts", "tambor")

    assert brand.ok and "Tambor" in brand.content
    assert posts.ok and posts.metadata["has_posts"] is True


# --- generate_ideas ----------------------------------------------------------


def test_generate_ideas_returns_ideas_with_ids(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.", posts="Post del yunque rojo.")

    result = run_tool("generate_ideas", "acme", {"content_type": "ugc", "count": 3})

    assert result.ok is True
    shown = json.loads(result.content)
    stored = result.metadata[IDEAS_METADATA_KEY]
    assert shown == stored
    assert [idea["title"] for idea in shown] == ["Idea 1", "Idea 2", "Idea 3"]
    assert all(idea["content_type"] == "ugc" for idea in shown)
    ids = [idea["id"] for idea in shown]
    assert len(set(ids)) == 3
    assert all(idea_id.startswith("idea-") for idea_id in ids)


def test_generate_ideas_defaults_to_three(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")

    result = run_tool("generate_ideas", "acme", {"content_type": "design"})

    assert result.metadata["count"] == 3
    assert "Proponé 3 ideas de contenido de diseño" in provider.generate_calls[0]["prompt"]


def test_generate_ideas_loads_the_session_client_material(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.", posts="Post del yunque rojo.")
    write_client(data_dir, "globex", brand="Globex vende cohetes.", posts="Post del cohete azul.")

    run_tool("generate_ideas", "acme", {"content_type": "ugc", "client_id": "globex"})

    prompt = provider.generate_calls[0]["prompt"]
    assert "Acme vende yunques." in prompt
    assert "Post del yunque rojo." in prompt
    assert "Globex" not in prompt


def test_generate_ideas_sends_the_brief(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")

    run_tool("generate_ideas", "acme", {"content_type": "ugc", "brief": "  día del padre  "})

    prompt = provider.generate_calls[0]["prompt"]
    assert "<foco_del_pedido>\ndía del padre\n</foco_del_pedido>" in prompt


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # falta content_type
        {"content_type": "video"},
        {"content_type": "ugc", "count": 0},
        {"content_type": "ugc", "count": 21},
        {"content_type": "ugc", "brief": "   "},
        {"content_type": "ugc", "brief": "hola\x00"},
        {"content_type": "ugc", "brief": "x" * 1001},
    ],
)
def test_generate_ideas_invalid_arguments_go_back_to_model(data_dir, provider, arguments):
    write_client(data_dir, "acme", brand="Acme vende yunques.")

    result = run_tool("generate_ideas", "acme", arguments)

    assert result.ok is False
    assert result.metadata["error_type"] == "validation_error"
    assert provider.generate_calls == []  # un pedido mal armado no cuesta una llamada


def test_generate_ideas_without_brand_is_a_tool_error(data_dir, provider):
    write_client(data_dir, "acme", posts="Post de Acme")

    result = run_tool("generate_ideas", "acme", {"content_type": "ugc"})

    assert result.ok is False
    assert result.metadata["error_type"] == "missing_brand"
    assert provider.generate_calls == []


def test_generate_ideas_with_too_few_ideas_is_a_tool_error(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    provider.ideas_to_return = 1

    result = run_tool("generate_ideas", "acme", {"content_type": "ugc", "count": 3})

    assert result.ok is False
    assert result.metadata["error_type"] == "NotEnoughIdeasError"


def test_generate_ideas_reports_model_usage(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")

    result = run_tool("generate_ideas", "acme", {"content_type": "ugc"})

    # El StubProvider responde con 100 tokens de entrada y 50 de salida.
    assert result.metadata[MODEL_INPUT_TOKENS_KEY] == 100
    assert result.metadata[MODEL_OUTPUT_TOKENS_KEY] == 50


def test_generate_ideas_reports_usage_even_with_too_few_ideas(data_dir, provider):
    # La llamada respondió aunque no alcance: esos tokens se pagaron igual.
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    provider.ideas_to_return = 1

    result = run_tool("generate_ideas", "acme", {"content_type": "ugc", "count": 3})

    assert result.ok is False
    assert result.metadata[MODEL_INPUT_TOKENS_KEY] == 100


def test_read_tools_report_no_model_usage(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme")

    assert MODEL_INPUT_TOKENS_KEY not in run_tool("get_brand_info", "acme").metadata


def test_generate_ideas_uses_the_session_run_id(data_dir, provider, caplog):
    write_client(data_dir, "acme", brand="Acme vende yunques.")

    with caplog.at_level("INFO", logger=llm.__name__):
        run_tool("generate_ideas", "acme", {"content_type": "ugc"})

    assert "run_id=run-1" in caplog.text


# --- Desde el grafo ----------------------------------------------------------


def turn(*, text=None, calls=()):
    return TurnResult(
        message=Message(role="model", text=text, tool_calls=list(calls)),
        model="m", input_tokens=10, output_tokens=2, stop_reason="stop", truncated=False,
    )


def test_graph_runs_read_tools_and_model_sees_brand(data_dir, provider):
    write_client(data_dir, "acme", brand="Marca Acme: café de especialidad", posts="Post de Acme")
    provider.turns = [
        turn(calls=[ToolCall(id="a", tool_name="get_brand_info"), ToolCall(id="b", tool_name="get_previous_posts")]),
        turn(text="Acme vende café de especialidad."),
    ]

    graph = build_graph(make_registry())
    outcome = run_turn(graph, AgentSession(thread_id="t1", client_id="acme"), "¿qué vende la marca?")

    assert outcome.tools_called == ["get_brand_info", "get_previous_posts"]
    assert all(step.ok for step in outcome.trace if step.kind == "tool")
    tool_messages = [m for m in provider.turn_calls[1]["messages"] if m.role == "tool"]
    assert "café de especialidad" in tool_messages[0].result.content
    assert "Post de Acme" in tool_messages[1].result.content
    assert outcome.reply == "Acme vende café de especialidad."


def test_graph_stores_generated_ideas_in_state(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    session = AgentSession(thread_id="t1", client_id="acme")
    graph = build_graph(make_registry())
    config = {"configurable": {"thread_id": "t1"}}

    provider.turns = [
        turn(calls=[ToolCall(id="g1", tool_name="generate_ideas", arguments={"content_type": "ugc", "count": 2})]),
        turn(text="Acá van dos ideas."),
    ]
    provider.ideas_to_return = 2
    first = run_turn(graph, session, "dos ideas UGC")

    assert [idea.title for idea in first.ideas] == ["Idea 1", "Idea 2"]
    # La traza suma los tokens de la llamada interna de generate_ideas.
    tool_step = next(step for step in first.trace if step.kind == "tool")
    assert (tool_step.input_tokens, tool_step.output_tokens) == (100, 50)
    assert first.input_tokens == 10 + 100 + 10  # dos llamadas del agente más la de la tool
    assert first.output_tokens == 2 + 50 + 2
    state = graph.get_state(config).values
    assert [idea.id for idea in state["ideas"]] == [idea.id for idea in first.ideas]
    # Las ideas no quedan duplicadas en la metadata del historial.
    tool_message = next(m for m in state["messages"] if m.role == "tool")
    assert IDEAS_METADATA_KEY not in tool_message.result.metadata
    # Pero el modelo sí las ve, con sus ids, en el contenido.
    assert first.ideas[0].id in tool_message.result.content

    provider.turns = [
        turn(calls=[ToolCall(id="g2", tool_name="generate_ideas", arguments={"content_type": "design", "count": 1})]),
        turn(text="Una más."),
    ]
    provider.ideas_to_return = 1
    second = run_turn(graph, session, "una de diseño")

    # El turno informa solo sus ideas; la sesión acumula todas.
    assert len(second.ideas) == 1
    assert second.ideas[0].content_type == "design"
    assert len(graph.get_state(config).values["ideas"]) == 3


def test_turn_events_tell_the_whole_story(data_dir, provider, caplog):
    write_client(data_dir, "acme", brand="Acme vende yunques.", posts="Post del yunque rojo.")
    provider.turns = [
        turn(calls=[ToolCall(id="g1", tool_name="generate_ideas",
                             arguments={"content_type": "ugc", "count": 2, "brief": "dato del cliente"})]),
        turn(text="Dos ideas."),
    ]
    provider.ideas_to_return = 2

    with caplog.at_level(logging.INFO):
        outcome = run_turn(build_graph(make_registry()), AgentSession(thread_id="t1", client_id="acme"), "dos ideas")

    timeline = [(e.event, e.depth) for e in outcome.events]
    assert timeline == [
        ("turn_start", 0),
        ("llm_turn", 0),           # el modelo decide usar generate_ideas
        ("route", 0),              # → ejecutar tools
        ("tool_start", 0),
        ("client_context", 1),     # adentro de la tool: carga el material...
        ("llm_call", 1),           # ...y llama al modelo
        ("tool_call", 0),
        ("ideas_added", 0),
        ("llm_turn", 0),           # el modelo responde con texto
        ("route", 0),              # → fin del turno
        ("turn_end", 0),
    ]
    # El primero de cada tipo: hay dos llm_turn y dos route.
    events = {}
    for e in outcome.events:
        events.setdefault(e.event, e)
    assert all(e.run_id == outcome.run_id for e in outcome.events)
    # Los argumentos que eligió el modelo están en la traza...
    assert events["tool_start"].private["arguments"]["brief"] == "dato del cliente"
    assert events["llm_turn"].private["tool_calls"] == [
        {"tool": "generate_ideas", "arguments": {"content_type": "ugc", "count": 2, "brief": "dato del cliente"}}
    ]
    # ...pero nunca en el log.
    assert "dato del cliente" not in caplog.text
    assert events["ideas_added"].fields["ids"] == [idea.id for idea in outcome.ideas]
    assert [e.fields["to"] for e in outcome.events if e.event == "route"] == ["run_tools", "__end__"]


def test_invalid_tool_arguments_show_which_field_failed(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    provider.turns = [
        turn(calls=[ToolCall(id="g1", tool_name="generate_ideas", arguments={"content_type": "video"})]),
        turn(text="¿UGC o diseño?"),
    ]

    outcome = run_turn(build_graph(make_registry()), AgentSession(thread_id="t1", client_id="acme"), "ideas")

    [tool_call] = [e for e in outcome.events if e.event == "tool_call"]
    assert tool_call.fields["ok"] is False
    assert tool_call.fields["error_type"] == "validation_error"
    assert any(item.startswith("content_type:") for item in tool_call.fields["invalid_fields"])


def test_checkpoint_reads_state_types_without_unregistered_warnings(data_dir, provider, caplog):
    # Si el estado guarda un tipo que no está en CHECKPOINT_TYPES, LangGraph
    # avisa al leer el checkpoint (y en versiones futuras lo bloquea).
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    session = AgentSession(thread_id="t1", client_id="acme")
    graph = build_graph(make_registry())
    provider.turns = [
        turn(calls=[ToolCall(id="g1", tool_name="generate_ideas", arguments={"content_type": "ugc", "count": 1})]),
        turn(text="Una idea."),
        turn(text="Dale."),
    ]
    provider.ideas_to_return = 1

    with caplog.at_level("WARNING"):
        run_turn(graph, session, "una idea UGC")
        run_turn(graph, session, "gracias")  # el segundo turno lee el checkpoint del primero

    assert "unregistered type" not in caplog.text


def test_turn_without_ideas_reports_none(data_dir, provider):
    write_client(data_dir, "acme", brand="Acme vende yunques.")
    provider.turns = [turn(text="¿UGC o diseño?")]

    outcome = run_turn(build_graph(make_registry()), AgentSession(thread_id="t1", client_id="acme"), "ideas")

    assert outcome.ideas == []
