import pytest
from pydantic import ValidationError

from yourebrand.creative.schemas import Idea, IdeaList

IDEA = {
    "title": "Primera vez en el local",
    "format": "reel",
    "pillar": "Cómo funciona",
    "description": "Plano secuencia de una persona usando el local por primera vez.",
    "rationale": "La principal barrera del público es no saber cómo se usa.",
}


def test_idea_accepts_complete_data():
    idea = Idea(**IDEA)

    assert idea.title == "Primera vez en el local"
    assert idea.format == "reel"


def test_idea_strips_surrounding_whitespace():
    assert Idea(**IDEA | {"title": "  Primera vez  "}).title == "Primera vez"


@pytest.mark.parametrize("field", IDEA.keys())
def test_idea_rejects_missing_field(field):
    data = {key: value for key, value in IDEA.items() if key != field}

    with pytest.raises(ValidationError):
        Idea(**data)


@pytest.mark.parametrize("field", IDEA.keys())
def test_idea_rejects_blank_field(field):
    with pytest.raises(ValidationError):
        Idea(**IDEA | {field: "   "})


def test_idea_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Idea(**IDEA | {"copy": "Vos seguí con lo tuyo."})


def test_idea_list_parses_model_json():
    raw = IdeaList(ideas=[Idea(**IDEA)]).model_dump_json()

    assert IdeaList.model_validate_json(raw).ideas[0].pillar == "Cómo funciona"


def test_idea_list_rejects_empty_list():
    with pytest.raises(ValidationError):
        IdeaList(ideas=[])


def test_schema_forbids_additional_properties():
    """La salida estructurada de la API exige objetos cerrados."""
    schema = IdeaList.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Idea"]["additionalProperties"] is False
