import pytest

from yourebrand.creative.context import ClientContext
from yourebrand.creative.prompts import build_idea_prompt, build_system_prompt

ACME = ClientContext("acme", brand="Acme vende yunques.", posts="Post del yunque rojo.")
ACME_WITHOUT_POSTS = ClientContext("acme", brand="Acme vende yunques.", posts=None)
GLOBEX = ClientContext("globex", brand="Globex vende cohetes.", posts="Post del cohete azul.")


def test_system_prompt_changes_with_content_type():
    ugc = build_system_prompt("ugc")
    design = build_system_prompt("design")

    assert ugc != design
    assert "Contenido UGC" in ugc
    assert "Contenido de diseño" in design


@pytest.mark.parametrize("content_type", ["ugc", "design"])
def test_system_prompt_leaves_no_unfilled_placeholders(content_type):
    system = build_system_prompt(content_type)

    assert "{" not in system
    assert "}" not in system


def test_idea_prompt_wraps_brand_and_posts_in_tags():
    prompt = build_idea_prompt(ACME, "ugc", 5)

    assert "<marca>\nAcme vende yunques.\n</marca>" in prompt
    assert "<posts_anteriores>\nPost del yunque rojo.\n</posts_anteriores>" in prompt


def test_idea_prompt_omits_posts_block_when_there_are_none():
    prompt = build_idea_prompt(ACME_WITHOUT_POSTS, "ugc", 5)

    assert "<marca>" in prompt
    assert "posts_anteriores" not in prompt


@pytest.mark.parametrize(
    ("content_type", "count", "expected"),
    [
        ("ugc", 5, "Proponé 5 ideas de contenido UGC para esta marca."),
        ("design", 3, "Proponé 3 ideas de contenido de diseño para esta marca."),
        ("ugc", 1, "Proponé 1 idea de contenido UGC para esta marca."),
    ],
)
def test_idea_prompt_ends_with_the_request(content_type, count, expected):
    assert build_idea_prompt(ACME, content_type, count).endswith(expected)


def test_idea_prompt_only_contains_its_own_client():
    prompt = build_idea_prompt(ACME, "ugc", 5)

    assert GLOBEX.brand not in prompt
    assert GLOBEX.posts not in prompt
    assert "Globex" not in prompt


@pytest.mark.parametrize("content_type", ["ugc", "design"])
def test_system_prompt_does_not_depend_on_any_client(content_type):
    system = build_system_prompt(content_type)

    for word in ("Acme", "Globex", "Tambor"):
        assert word not in system
