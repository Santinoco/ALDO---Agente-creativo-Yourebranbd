"""Skills del Agente Creativo: formato de los archivos, carga, uso fijo y tool load_skill."""

import pytest

from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import ExecutionContext, ToolCall
from yourebrand.creative.prompts import CONTENT_TYPE_SKILLS, build_agent_system_prompt, build_system_prompt
from yourebrand.creative.skills import (
    AGENT_SECTION_HEADING,
    InvalidSkillError,
    UnknownSkillError,
    available_skills,
    get_skill,
    load_skills,
    parse_skill,
)
from yourebrand.creative import skills as skills_module
from yourebrand.creative import tools as tools_module
from yourebrand.creative.tools import build_creative_registry

VALID = """---
name: mi-skill
description: Para probar.
status: borrador
version: 0.1
---

Criterio general.

### Para el agente

Preguntá antes de suponer.
"""


# --- Formato de un SKILL.md --------------------------------------------------


def test_parses_header_and_body():
    skill = parse_skill(VALID, folder_name="mi-skill")

    assert (skill.name, skill.description, skill.status, skill.version) == (
        "mi-skill", "Para probar.", "borrador", "0.1",
    )
    assert skill.body.startswith("Criterio general.")


def test_criteria_leaves_out_the_agent_section():
    skill = parse_skill(VALID, folder_name="mi-skill")

    assert skill.criteria == "Criterio general."
    assert AGENT_SECTION_HEADING in skill.body


def test_accepts_windows_line_endings():
    assert parse_skill(VALID.replace("\n", "\r\n"), folder_name="mi-skill").name == "mi-skill"


def test_status_defaults_to_active():
    text = "---\nname: x\ndescription: d\n---\ncuerpo"

    assert parse_skill(text, folder_name="x").status == "activa"


@pytest.mark.parametrize(
    "text",
    [
        "sin encabezado",
        "---\nname: mi-skill\n---\ncuerpo",                           # falta description
        "---\nname: mi-skill\ndescription: d\n---\n   ",              # cuerpo vacío
        "---\nname: Mi Skill\ndescription: d\n---\ncuerpo",           # nombre inválido
        "---\nname: mi-skill\ndescription: d\nstatus: lista\n---\nc",  # estado desconocido
        "---\nname: mi-skill\nlinea sin dos puntos\n---\ncuerpo",
    ],
)
def test_rejects_malformed_skills(text):
    with pytest.raises(InvalidSkillError):
        parse_skill(text, folder_name="mi-skill")


def test_name_must_match_folder():
    with pytest.raises(InvalidSkillError):
        parse_skill(VALID, folder_name="otra-carpeta")


def test_load_skills_reads_every_folder(tmp_path):
    for name in ("b-skill", "a-skill"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(VALID.replace("mi-skill", name), encoding="utf-8")
    (tmp_path / "sin-skill").mkdir()  # una carpeta sin SKILL.md no es una skill

    assert list(load_skills(tmp_path)) == ["a-skill", "b-skill"]


# --- Las skills del repo -----------------------------------------------------


def test_bundled_skills_load():
    assert list(available_skills()) == ["design-ideation", "ugc-ideation", "ugc-video-editing"]


def test_video_editing_skill_follows_the_agency_guide():
    # Validada con la guía de edición de Yourebrand (noviembre 2026).
    skill = get_skill("ugc-video-editing")

    assert skill.status == "activa"
    for rule in ("palabra por palabra", "Sin transiciones", "libre uso", "llamado a la acción"):
        assert rule in skill.body


@pytest.mark.parametrize("name", ["ugc-ideation", "design-ideation", "ugc-video-editing"])
def test_bundled_skills_do_not_mention_clients(name):
    # Una skill es criterio de la agencia: sirve para cualquier cliente.
    body = get_skill(name).body.lower()
    for client in ("tambor", "acme", "globex"):
        assert client not in body


def test_unknown_skill_fails():
    with pytest.raises(UnknownSkillError):
        get_skill("no-existe")


# --- Uso fijo: generate_ideas ------------------------------------------------


@pytest.mark.parametrize("content_type", ["ugc", "design"])
def test_generation_prompt_uses_the_skill_criteria_without_agent_section(content_type):
    skill = get_skill(CONTENT_TYPE_SKILLS[content_type])
    system = build_system_prompt(content_type)

    assert skill.criteria in system
    assert AGENT_SECTION_HEADING not in system


# --- Uso decidido por el agente: índice y load_skill -------------------------


def test_agent_prompt_lists_skills_without_their_body():
    prompt = build_agent_system_prompt()

    for skill in available_skills().values():
        assert f"- {skill.name}" in prompt
        assert skill.description in prompt
    # Solo el índice: el criterio completo lo trae load_skill.
    assert get_skill("ugc-video-editing").criteria not in prompt


def test_agent_prompt_marks_draft_skills():
    draft = parse_skill(VALID, folder_name="mi-skill")

    assert "- mi-skill (borrador): Para probar." in build_agent_system_prompt(skills=[draft])


def test_agent_prompt_without_skills_has_no_skills_section():
    assert "## Skills" not in build_agent_system_prompt(skills=[])


def run_load_skill(arguments):
    registry = build_creative_registry(generation_timeout_s=5)
    context = ExecutionContext(tenant=TenantContext("acme"), run_id="run-1")
    return registry.run(ToolCall(id="c1", tool_name="load_skill", arguments=arguments), context)


def test_load_skill_declaration_lists_names_as_enum():
    registry = build_creative_registry(generation_timeout_s=5)
    declaration = {d["name"]: d for d in registry.function_declarations()}["load_skill"]

    assert declaration["parameters"]["properties"]["name"]["enum"] == list(available_skills())
    assert declaration["parameters"]["required"] == ["name"]


def test_load_skill_returns_the_full_skill():
    result = run_load_skill({"name": "ugc-ideation"})

    assert result.ok is True
    assert result.content.startswith('<skill nombre="ugc-ideation">')
    assert get_skill("ugc-ideation").body in result.content
    assert result.metadata["status"] == "activa"


def test_load_skill_warns_when_the_skill_is_a_draft(monkeypatch):
    draft = parse_skill(VALID, folder_name="mi-skill")
    monkeypatch.setattr(skills_module, "available_skills", lambda: {"mi-skill": draft})
    monkeypatch.setattr(tools_module, "available_skills", lambda: {"mi-skill": draft})

    result = run_load_skill({"name": "mi-skill"})

    assert "borrador" in result.content.splitlines()[1]


def test_load_skill_of_an_active_skill_has_no_draft_warning():
    result = run_load_skill({"name": "ugc-video-editing"})

    assert "borrador" not in result.content.splitlines()[1]


def test_load_skill_with_unknown_name_is_a_validation_error():
    result = run_load_skill({"name": "inventada"})

    assert result.ok is False
    assert result.metadata["error_type"] == "validation_error"
