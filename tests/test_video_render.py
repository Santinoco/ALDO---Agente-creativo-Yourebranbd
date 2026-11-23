"""Render de video: planes guardados, la tool render_video y un render real con ffmpeg."""

import io
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from yourebrand.core import storage, tenancy, video_prep
from yourebrand.core.storage import UnknownPlanError, save_video
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tools import ApprovalPreviewError, ExecutionContext, ToolCall
from yourebrand.core.transcription import Transcript, Word
from yourebrand.creative import video_render
from yourebrand.creative.schemas import VideoEditPlan
from yourebrand.creative.tools import RENDER_METADATA_KEY, build_creative_registry, render_video_preview

ACME = TenantContext("acme")
GLOBEX = TenantContext("globex")
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100
TRANSCRIPT = Transcript(language="es", words=[
    Word(start_s=0.5, end_s=0.9, text="Probé"), Word(start_s=0.9, end_s=1.2, text="el"),
    Word(start_s=1.2, end_s=1.8, text="café"), Word(start_s=3.2, end_s=3.6, text="{\\b1}rico"),
])


def make_plan(**overrides) -> VideoEditPlan:
    plan = {
        "video_type": "recomendacion",
        "hook": "Arranca con la primera probada.",
        "segments": [
            {"start_s": 3.0, "end_s": 4.0, "purpose": "gancho", "zoom": True,
             "b_roll": [{"start_s": 4.5, "end_s": 5.0, "offset_s": 0.2, "description": "la taza"}]},
            {"start_s": 0.0, "end_s": 2.0, "purpose": "cierre", "zoom": False, "b_roll": []},
        ],
        "subtitle_style": "palabra",
        "on_screen_texts": [{"text": "café", "at_s": 0.5}],
        "music": "acústica suave",
        "sound_effects": [],
        "call_to_action": "andá a probarlo",
        "target_network": "Reels",
        "warnings": [],
    }
    return VideoEditPlan.model_validate(plan | overrides)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir, storage_dir = tmp_path / "data", tmp_path / "storage"
    for client_id in ("acme", "globex"):
        (data_dir / client_id).mkdir(parents=True)
        (data_dir / client_id / "brand.md").write_text("# Marca", encoding="utf-8")
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(storage_dir=storage_dir, max_video_mb=50))
    monkeypatch.setattr(video_prep, "get_settings", lambda: SimpleNamespace(video_max_frames=3))
    return SimpleNamespace(tmp=tmp_path)


class Recorder:
    """Executor que guarda lo que se le pide en vez de correrlo."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))


class Inline:
    """Executor que corre el trabajo en el momento: el test espera el render."""

    def submit(self, fn, *args):
        fn(*args)


def registry():
    return build_creative_registry(generation_timeout_s=5, video_edit=True, video_edit_timeout_s=5,
                                   video_edit_max_calls=2)


def render_call(plan_id):
    return ToolCall(id="c1", tool_name="render_video", arguments={"plan_id": plan_id})


# --- Planes guardados y vista previa -------------------------------------------


def test_stored_plans_are_per_client(env):
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")

    assert video_render.load_plan(ACME, stored.plan_id).plan == make_plan()
    with pytest.raises(UnknownPlanError):
        video_render.load_plan(GLOBEX, stored.plan_id)
    with pytest.raises(UnknownPlanError):
        video_render.load_plan(ACME, "../acme/plans/x")


def test_preview_shows_the_stored_plan(env):
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")
    preview = render_video_preview({"plan_id": stored.plan_id}, ExecutionContext(tenant=ACME, run_id="r2"))

    assert preview["plan_id"] == stored.plan_id
    assert preview["duracion_estimada_s"] == 3.0
    assert [t["para"] for t in preview["tramos"]] == ["gancho", "cierre"]


def test_preview_of_another_clients_plan_fails(env):
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")
    with pytest.raises(ApprovalPreviewError):
        render_video_preview({"plan_id": stored.plan_id}, ExecutionContext(tenant=GLOBEX, run_id="r2"))


def test_render_tool_requires_approval_and_is_offered_only_with_video_edit():
    reg = registry()
    assert reg.requires_approval("render_video")
    assert "render_video" in [t["name"] for t in reg.function_declarations()]
    plain = build_creative_registry(generation_timeout_s=5, video_edit=False)
    assert "render_video" not in plain.tools


# --- La tool ---------------------------------------------------------------------


def test_render_tool_without_approval_does_not_start(env, monkeypatch):
    executor = Recorder()
    monkeypatch.setattr(video_render, "_EXECUTOR", executor)
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")

    result = registry().run(render_call(stored.plan_id), ExecutionContext(tenant=ACME, run_id="r2"))
    assert result.metadata["error_type"] == "approval_required"
    assert executor.submitted == []


def test_approved_render_tool_queues_and_returns_right_away(env, monkeypatch):
    executor = Recorder()
    monkeypatch.setattr(video_render, "_EXECUTOR", executor)
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")

    result = registry().run(render_call(stored.plan_id), ExecutionContext(tenant=ACME, run_id="r2", approved=True))

    assert result.ok, result.content
    content = json.loads(result.content)
    assert content["render_id"].startswith("render-")
    assert any("música" in w for w in content["advertencias"])  # la música todavía no se agrega
    assert result.metadata[RENDER_METADATA_KEY]["state"] == "queued"
    assert len(executor.submitted) == 1
    assert video_render.get_render(ACME, content["render_id"]).state == "queued"


def test_render_tool_with_unknown_plan(env):
    result = registry().run(render_call("plan-000000000000"),
                            ExecutionContext(tenant=ACME, run_id="r2", approved=True))
    assert result.metadata["error_type"] == "unknown_plan"


def test_a_render_left_from_a_previous_process_reads_as_failed(env, monkeypatch):
    monkeypatch.setattr(video_render, "_EXECUTOR", Recorder())
    stored = video_render.store_plan(ACME, "vid-000000000001", make_plan(), run_id="r1")
    status = video_render.start_render(ACME, stored.plan_id, run_id="r2", approved=True)
    video_render._ACTIVE.discard(status.render_id)  # como si la app se hubiera reiniciado

    assert video_render.get_render(ACME, status.render_id).state == "failed"


# --- Armado sin ffmpeg -----------------------------------------------------------


def test_subtitles_follow_the_final_timeline_and_cannot_inject_ass():
    plan = make_plan()
    cues = video_render.subtitle_cues(plan, TRANSCRIPT.words)

    # El primer tramo del video final es 3.0-4.0 del original: "rico" (3.2-3.6) queda en 0.2-0.6.
    assert cues[0].text == "{\\b1}rico" and cues[0].start_s == 0.2
    # El segundo tramo (0.0-2.0) empieza en 1.0 del video final.
    assert [c.text for c in cues[1:]] == ["Probé", "el", "café"]
    assert cues[1].start_s == 1.5

    ass = video_render.build_ass(cues, video_render.on_screen_cues(plan))
    assert "{\\b1}" not in ass and "(b1)rico" in ass


def test_ffmpeg_args_cut_every_segment_and_burn_subtitles():
    args = video_render.build_ffmpeg_args(make_plan(), storage.Path("src.mp4"), has_audio=True)
    graph = args[args.index("-filter_complex") + 1]

    assert args.count("-i") == 3  # dos tramos más un b-roll
    assert "concat=n=2:v=1:a=1" in graph and "ass=subs.ass" in graph
    assert args[-1] == video_render.OUTPUT_FILENAME


# --- Render real -------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg no está instalado")


@needs_ffmpeg
def test_real_render_produces_the_edited_video(env, monkeypatch):
    source = env.tmp / "crudo.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=360x640:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )
    with source.open("rb") as file:
        ref = save_video(ACME, file, "crudo.mp4")
    # Whisper real descargaría un modelo: la transcripción se simula.
    monkeypatch.setattr(video_prep, "transcribe", lambda audio: TRANSCRIPT)
    stored = video_render.store_plan(ACME, ref.id, make_plan(), run_id="r1")

    status = video_render.start_render(ACME, stored.plan_id, run_id="r2", approved=True, executor=Inline())
    final = video_render.get_render(ACME, status.render_id)

    assert final.state == "done", final.error
    assert final.duration_s == pytest.approx(3.0, abs=0.2)
    output = video_render.render_output(ACME, status.render_id)
    assert output.stat().st_size > 0
    with pytest.raises(storage.UnknownRenderError):
        video_render.render_output(GLOBEX, status.render_id)
