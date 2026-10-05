from types import SimpleNamespace

import pytest

from yourebrand.core import tenancy
from yourebrand.core.tenancy import (
    InvalidClientIdError,
    TenantContext,
    UnknownClientError,
    client_data_dir,
)


@pytest.mark.parametrize("client_id", ["acme", "cafe-del-sur", "cliente-2"])
def test_valid_client_id(client_id):
    assert TenantContext(client_id).client_id == client_id


@pytest.mark.parametrize(
    "client_id",
    ["", "Acme", "../acme", "acme/otro", "acme otro", "-acme", "acme-", "acme\n"],
)
def test_invalid_client_id_fails(client_id):
    with pytest.raises(InvalidClientIdError):
        TenantContext(client_id)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Usa una carpeta temporal como data_dir, sin leer la configuración real."""
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    return tmp_path


def test_client_data_dir_returns_client_folder(data_dir):
    (data_dir / "acme").mkdir()

    assert client_data_dir(TenantContext("acme")) == data_dir / "acme"


def test_client_data_dir_unknown_client_fails(data_dir):
    with pytest.raises(UnknownClientError):
        client_data_dir(TenantContext("acme"))
