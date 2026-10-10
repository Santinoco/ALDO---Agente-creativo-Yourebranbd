import logging
from types import SimpleNamespace

import pytest

from yourebrand.core import tenancy
from yourebrand.core.config import PROJECT_ROOT
from yourebrand.core.tenancy import TenantContext, UnknownClientError
from yourebrand.creative import context
from yourebrand.creative.context import MissingBrandError, load_client_context

ACME = TenantContext("acme")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Usa una carpeta temporal como data_dir, sin leer la configuración real."""
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    return tmp_path


def write_client(data_dir, client_id, *, brand=None, posts=None):
    folder = data_dir / client_id
    folder.mkdir()
    if brand is not None:
        (folder / "brand.md").write_text(brand, encoding="utf-8")
    if posts is not None:
        (folder / "posts.md").write_text(posts, encoding="utf-8")


def test_loads_brand_and_posts(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme\n", posts="Post de Acme\n")

    result = load_client_context(ACME)

    assert result.client_id == "acme"
    assert result.brand == "Marca Acme"
    assert result.posts == "Post de Acme"


def test_works_without_posts_file(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme")

    assert load_client_context(ACME).posts is None


def test_blank_posts_file_counts_as_no_posts(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme", posts="  \n")

    assert load_client_context(ACME).posts is None


@pytest.mark.parametrize("brand", [None, "", "  \n"])
def test_missing_or_blank_brand_fails(data_dir, brand):
    write_client(data_dir, "acme", brand=brand, posts="Post de Acme")

    with pytest.raises(MissingBrandError):
        load_client_context(ACME)


def test_unknown_client_fails(data_dir):
    with pytest.raises(UnknownClientError):
        load_client_context(ACME)


def test_reads_only_the_requested_client(data_dir):
    write_client(data_dir, "acme", brand="Marca Acme", posts="Post de Acme")
    write_client(data_dir, "globex", brand="Marca Globex", posts="Post de Globex")

    result = load_client_context(ACME)

    assert "Globex" not in result.brand
    assert "Globex" not in result.posts


def test_keeps_non_ascii_characters(data_dir):
    write_client(data_dir, "acme", brand="Lavandería: «vos seguí con lo tuyo» ☕")

    assert load_client_context(ACME).brand == "Lavandería: «vos seguí con lo tuyo» ☕"


def test_logs_size_without_content(data_dir, caplog):
    write_client(data_dir, "acme", brand="secreto de marca", posts="post confidencial")

    with caplog.at_level(logging.INFO, logger=context.__name__):
        load_client_context(ACME)

    assert "client_id=acme" in caplog.text
    assert "brand_chars=16" in caplog.text
    assert "posts_chars=17" in caplog.text
    assert "secreto de marca" not in caplog.text
    assert "post confidencial" not in caplog.text


def test_bundled_sample_client_loads(monkeypatch):
    """El cliente ficticio del repo tiene que poder cargarse tal como está."""
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=PROJECT_ROOT / "data"))

    result = load_client_context(TenantContext("tambor"))

    assert "Tambor" in result.brand
    assert result.posts is not None
