"""Subida de videos: almacenamiento aislado por cliente, formato, el agente y la API."""

import io
from functools import partial
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yourebrand.core import llm, media, storage, tenancy
from yourebrand.core.providers import Message, TurnResult
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.storage import (
    InvalidVideoError,
    UnknownVideoError,
    VideoTooLargeError,
    clean_filename,
    get_video,
    save_video,
    video_file,
)
from yourebrand.core.tenancy import TenantContext, UnknownClientError
from yourebrand.core.tools import ToolRegistry
from yourebrand.api.app import create_app
from yourebrand.creative.graph import AgentSession, build_graph, run_turn

ACME = TenantContext("acme")
GLOBEX = TenantContext("globex")

# Cabeceras mínimas: tamaño de la primera caja, "ftyp" y la marca del formato.
MP4_HEADER = b"\x00\x00\x00\x18ftypisom"
MOV_HEADER = b"\x00\x00\x00\x14ftypqt  "


def video_bytes(header=MP4_HEADER, size=2048):
    return header + b"\x00" * (size - len(header))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Carpeta de clientes y de almacenamiento temporales; tope de 1 MB; sin ffprobe."""
    data_dir, storage_dir = tmp_path / "data", tmp_path / "storage"
    for client_id in ("acme", "globex"):
        (data_dir / client_id).mkdir(parents=True)
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(storage_dir=storage_dir, max_video_mb=1))
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    return SimpleNamespace(data=data_dir, storage=storage_dir)


# --- Formato -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (MP4_HEADER, "mp4"),
        (b"\x00\x00\x00\x20ftypmp42", "mp4"),
        (MOV_HEADER, "mov"),
        (b"\x00\x00\x00\x08wide\x00\x00\x00\x00", "mov"),  # .mov viejo sin ftyp
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d", None),
        (b"RIFF\x00\x00\x00\x00AVI ", None),
        (b"corto", None),
    ],
)
def test_detect_container(header, expected):
    assert media.detect_container(header) == expected


def test_probe_duration_without_ffprobe_is_none(dirs, tmp_path):
    assert media.probe_duration(tmp_path / "x.mp4") is None


# --- Almacenamiento ----------------------------------------------------------


def test_save_and_get_video(dirs):
    ref = save_video(ACME, io.BytesIO(video_bytes()), "Mi video.mp4")

    assert ref.id.startswith("vid-")
    assert (ref.filename, ref.container, ref.size_bytes, ref.duration_s) == ("Mi video.mp4", "mp4", 2048, None)
    path = video_file(ACME, ref.id)
    assert path == dirs.storage / "acme" / "videos" / ref.id / "original.mp4"
    assert path.read_bytes() == video_bytes()
    assert get_video(ACME, ref.id) == ref


def test_mov_is_saved_with_its_extension(dirs):
    ref = save_video(ACME, io.BytesIO(video_bytes(MOV_HEADER)), "clip.MOV")

    assert ref.container == "mov"
    assert video_file(ACME, ref.id).name == "original.mov"


def test_not_a_video_is_rejected_and_nothing_is_written(dirs):
    with pytest.raises(InvalidVideoError):
        save_video(ACME, io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100), "foto.mp4")

    assert not (dirs.storage / "acme" / "videos").exists() or not any((dirs.storage / "acme" / "videos").iterdir())


def test_empty_file_is_rejected(dirs):
    with pytest.raises(InvalidVideoError):
        save_video(ACME, io.BytesIO(b""), "vacio.mp4")


def test_too_large_video_is_rejected_and_cleaned_up(dirs):
    with pytest.raises(VideoTooLargeError):
        save_video(ACME, io.BytesIO(video_bytes(size=2 * 1024 * 1024)), "largo.mp4")

    assert not any((dirs.storage / "acme" / "videos").iterdir())


def test_a_client_cannot_read_another_clients_video(dirs):
    ref = save_video(ACME, io.BytesIO(video_bytes()), "de acme.mp4")

    with pytest.raises(UnknownVideoError):
        get_video(GLOBEX, ref.id)


@pytest.mark.parametrize("video_id", ["../acme", "vid-../../x", "vid-123", "VID-0123456789ab", "", "vid-0123456789ab/.."])
def test_malformed_ids_cannot_escape_the_client_folder(dirs, video_id):
    with pytest.raises(UnknownVideoError):
        get_video(ACME, video_id)


def test_unknown_client_cannot_store_videos(dirs):
    with pytest.raises(UnknownClientError):
        save_video(TenantContext("nadie"), io.BytesIO(video_bytes()), "x.mp4")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/pass<wd>.mp4", "passwd.mp4"),
        ("C:\\fakepath\\clip.mov", "clip.mov"),
        ("con\x00trol.mp4", "control.mp4"),
        ("", "video"),
        (None, "video"),
        ("a" * 300 + ".mp4", "a" * 120),
    ],
)
def test_clean_filename(raw, expected):
    assert clean_filename(raw) == expected


# --- El agente ---------------------------------------------------------------


class ScriptedProvider:
    name = "scripted"

    def __init__(self):
        self.turns = []
        self.calls = []

    def generate_turn(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.turns.pop(0)


def text_turn(text):
    return TurnResult(message=Message(role="model", text=text), model="m", input_tokens=10, output_tokens=2,
                      stop_reason="stop", truncated=False)


@pytest.fixture
def provider(monkeypatch):
    stub = ScriptedProvider()
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


SESSION = AgentSession(thread_id="t1", client_id="acme")


def test_attached_video_reaches_the_model_and_the_state(dirs, provider):
    ref = save_video(ACME, io.BytesIO(video_bytes()), "cafeteria.mp4")
    graph = build_graph(ToolRegistry())
    provider.turns = [text_turn("Recibí el video.")]

    outcome = run_turn(graph, SESSION, "Editá este video", [ref.id])

    user_message = provider.calls[0]["messages"][-1].text
    assert user_message.startswith("Editá este video")
    assert f"[Video adjunto: id {ref.id} · «cafeteria.mp4» · duración desconocida · 0.0 MB]" in user_message
    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert [video.id for video in state["videos"]] == [ref.id]
    [attached] = [e for e in outcome.events if e.event == "video_attached"]
    assert attached.fields["video_id"] == ref.id
    # El nombre del archivo no va a los campos que se loguean.
    assert "cafeteria" not in str(attached.fields)


def test_video_of_another_client_is_rejected_before_calling_the_model(dirs, provider):
    ref = save_video(GLOBEX, io.BytesIO(video_bytes()), "de globex.mp4")

    with pytest.raises(UnknownVideoError):
        run_turn(build_graph(ToolRegistry()), SESSION, "Editá este", [ref.id])
    assert provider.calls == []


def test_the_same_video_is_not_added_twice(dirs, provider):
    ref = save_video(ACME, io.BytesIO(video_bytes()), "clip.mp4")
    graph = build_graph(ToolRegistry())
    provider.turns = [text_turn("uno"), text_turn("dos")]

    run_turn(graph, SESSION, "primero", [ref.id, ref.id])
    run_turn(graph, SESSION, "de nuevo este", [ref.id])

    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert len(state["videos"]) == 1
    # Igual se le vuelve a mencionar al modelo en el segundo mensaje.
    assert ref.id in provider.calls[1]["messages"][-1].text


def test_turn_without_videos_adds_no_note(dirs, provider):
    provider.turns = [text_turn("hola")]

    run_turn(build_graph(ToolRegistry()), SESSION, "hola")

    assert provider.calls[0]["messages"][-1].text == "hola"


# --- API ---------------------------------------------------------------------


@pytest.fixture
def client(dirs, provider):
    return TestClient(create_app(build_graph(ToolRegistry())))


def open_session(client, client_id="acme"):
    return client.post("/sessions", json={"client_id": client_id}).json()["thread_id"]


def upload(client, thread_id, content, filename="clip.mp4"):
    return client.post(f"/sessions/{thread_id}/videos", files={"file": (filename, content, "video/mp4")})


def test_upload_returns_the_video_reference(client):
    response = upload(client, open_session(client), video_bytes())

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("vid-")
    assert (body["filename"], body["container"], body["size_bytes"]) == ("clip.mp4", "mp4", 2048)


def test_upload_stores_the_video_in_the_session_client(client, dirs):
    thread_id = open_session(client, "globex")
    video_id = upload(client, thread_id, video_bytes()).json()["id"]

    assert (dirs.storage / "globex" / "videos" / video_id / "original.mp4").is_file()
    assert not (dirs.storage / "acme").exists()


@pytest.mark.parametrize(
    ("content", "status"),
    [(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, 415), (video_bytes(size=2 * 1024 * 1024), 413)],
    # Nombres cortos: sin esto pytest pone los 2 MB en el nombre del test.
    ids=["no-es-video", "demasiado-grande"],
)
def test_upload_rejects_bad_files(client, content, status):
    assert upload(client, open_session(client), content).status_code == status


def test_upload_to_unknown_session_is_404(client):
    assert upload(client, "no-existe", video_bytes()).status_code == 404


def test_message_with_uploaded_video(client, provider):
    thread_id = open_session(client)
    video_id = upload(client, thread_id, video_bytes()).json()["id"]
    provider.turns = [text_turn("Lo veo.")]

    response = client.post(f"/sessions/{thread_id}/messages", json={"text": "editalo", "video_ids": [video_id]})

    assert response.status_code == 200
    assert video_id in provider.calls[0]["messages"][-1].text
    assert "video_attached" in [event["event"] for event in response.json()["events"]]


def test_message_with_video_of_another_client_is_404(client, provider):
    globex_video = upload(client, open_session(client, "globex"), video_bytes()).json()["id"]
    acme_thread = open_session(client, "acme")

    response = client.post(f"/sessions/{acme_thread}/messages", json={"text": "editalo", "video_ids": [globex_video]})

    assert response.status_code == 404
    assert response.json()["detail"] == "video desconocido"
    assert provider.calls == []
