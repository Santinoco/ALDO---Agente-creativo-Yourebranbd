"""Etapa 2 de la edición de video: imágenes en el contrato, preparación y la tool plan_video_edit.

ffmpeg, Whisper y Claude se simulan: los tests no dependen de nada instalado.
"""

import json
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from yourebrand.core import llm, media, storage, tenancy, video_prep
from yourebrand.core.errors import BudgetExceededError, ProviderTimeoutError, RateLimitError, TransientProviderError
from yourebrand.core.llm import LLMResponse, StructuredResponse
from yourebrand.core.providers import ImageInput, ProviderResult
from yourebrand.core.providers.anthropic_provider import _user_content
from yourebrand.core.providers.gemini_provider import _prompt_contents
from yourebrand.core.storage import save_video
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.reliability import call_with_retry
from yourebrand.core.tools import MODEL_INPUT_TOKENS_KEY, ExecutionContext, Tool, ToolCall, ToolRegistry, ToolResult
from yourebrand.core.transcription import Transcript, Word
from yourebrand.creative import tools as creative_tools
from yourebrand.creative import video_edit
from yourebrand.creative.schemas import VideoEditPlan
from yourebrand.creative.tools import VIDEO_PLAN_METADATA_KEY, build_creative_registry

ACME = TenantContext("acme")
GLOBEX = TenantContext("globex")
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100
IMAGE = ImageInput(data=b"\xff\xd8jpeg", label="Fotograma en 1.0 s")


# --- Contrato: imágenes ------------------------------------------------------


def test_anthropic_sends_images_before_the_prompt_with_their_label():
    blocks = _user_content("armá el plan", [IMAGE])

    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert blocks[0]["text"] == "Fotograma en 1.0 s"
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "/9hqcGVn"}
    assert blocks[2]["text"] == "armá el plan"


def test_anthropic_without_images_sends_plain_text():
    assert _user_content("hola", []) == "hola"


def test_gemini_sends_images_as_parts():
    parts = _prompt_contents("armá el plan", [IMAGE])

    assert parts[0].text == "Fotograma en 1.0 s"
    assert parts[1].inline_data.data == b"\xff\xd8jpeg"
    assert parts[1].inline_data.mime_type == "image/jpeg"
    assert parts[2].text == "armá el plan"
    assert _prompt_contents("hola", []) == "hola"


class RecordingProvider:
    """Registra cada llamada. `failures` son errores que lanza en orden antes de responder."""

    name = "anthropic"

    def __init__(self):
        self.calls = []
        self.failures = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return ProviderResult(text='{"ok": true}', model=kwargs["model"], input_tokens=900, output_tokens=50,
                              stop_reason="end_turn", truncated=False)


class SimpleModel(BaseModel):
    ok: bool


@pytest.fixture
def provider(monkeypatch):
    stub = RecordingProvider()
    stub.settings = SimpleNamespace(
        model_generation="g", model_fast="f", model_video_edit="claude-opus-5-5",
        llm_timeout_seconds=30, llm_max_retries=2, video_edit_timeout_seconds=600,
        video_edit_max_output_tokens=16000, video_edit_max_input_tokens=60000,
    )
    monkeypatch.setattr(llm, "get_provider", lambda name=None: stub)
    monkeypatch.setattr(llm, "get_settings", lambda: stub.settings)
    monkeypatch.setattr(llm, "call_with_retry", partial(call_with_retry, sleep=lambda seconds: None))
    return stub


def plan_call(**overrides):
    kwargs = dict(system="s", prompt="p", schema=SimpleModel, tenant=ACME, run_id="r", tier="video_edit")
    return llm.complete_structured(**(kwargs | overrides))


def test_llm_passes_images_only_when_there_are(provider):
    llm.complete(system="s", prompt="p", tenant=ACME, run_id="r", tier="video_edit", max_tokens=10)
    plan_call(images=[IMAGE])

    assert "images" not in provider.calls[0]
    assert provider.calls[1]["images"] == [IMAGE]


# --- Topes de gasto de la edición de video -----------------------------------


def test_video_edit_output_is_capped_whatever_the_caller_asks(provider):
    plan_call(max_tokens=128000)

    assert provider.calls[0]["max_tokens"] == 16000


def test_other_tasks_keep_their_max_tokens_and_timeout(provider):
    llm.complete(system="s", prompt="p", tenant=ACME, run_id="r", tier="generation", max_tokens=30000)

    assert provider.calls[0]["max_tokens"] == 30000
    assert provider.calls[0]["timeout_seconds"] == 30


def test_input_over_budget_is_refused_without_calling_the_model(provider):
    provider.settings.video_edit_max_input_tokens = 5000
    images = [ImageInput(data=b"x", width=512, height=910)] * 10  # ~6.220 tokens

    with pytest.raises(BudgetExceededError):
        plan_call(images=images)
    assert provider.calls == []


def test_input_budget_counts_text_too(provider):
    provider.settings.video_edit_max_input_tokens = 1000

    with pytest.raises(BudgetExceededError):
        plan_call(prompt="palabra " * 1000)
    assert provider.calls == []


def test_image_token_estimate():
    assert llm.estimate_image_tokens(ImageInput(data=b"", width=512, height=910)) == 622
    # Claude la achica a 1568 px de lado mayor antes de contar.
    big = llm.estimate_image_tokens(ImageInput(data=b"", width=3000, height=3000))
    assert big == 1534
    # Sin tamaño, el peor caso.
    assert llm.estimate_image_tokens(ImageInput(data=b"")) >= big


def test_a_video_edit_timeout_is_not_retried(provider):
    """Cortada por tiempo, la respuesta pudo cobrarse entera: reintentar pagaría dos veces."""
    provider.failures = [ProviderTimeoutError("anthropic: timeout")]

    with pytest.raises(ProviderTimeoutError):
        plan_call()
    assert len(provider.calls) == 1
    assert provider.calls[0]["timeout_seconds"] == 600


def test_a_video_edit_rate_limit_is_retried(provider):
    """Un 429 no generó nada: reintentar no paga de más."""
    provider.failures = [RateLimitError("anthropic: 429")]

    plan_call()
    assert len(provider.calls) == 2


def test_jpeg_size_reads_the_header():
    # SOI, un APP0 de 16 bytes y un SOF0 con alto 910 y ancho 512.
    jpeg = (b"\xff\xd8" + b"\xff\xe0\x00\x10" + b"\x00" * 14
            + b"\xff\xc0\x00\x11\x08" + (910).to_bytes(2, "big") + (512).to_bytes(2, "big") + b"\x03")
    assert media.jpeg_size(jpeg) == (512, 910)
    assert media.jpeg_size(b"no es un jpeg") is None


def test_registry_caps_calls_per_run():
    registry = ToolRegistry()
    registry.register(Tool(name="cara", description="d", kind="compute",
                           fn=lambda args, ctx: ToolResult(ok=True, content="ok"), max_calls_per_run=2))
    call = ToolCall(id="c", tool_name="cara")
    first_run = ExecutionContext(tenant=ACME, run_id="run-a")

    results = [registry.run(call, first_run) for _ in range(3)]

    assert [r.ok for r in results] == [True, True, False]
    assert results[2].metadata["error_type"] == "call_limit"
    # Otro mensaje de la persona es otro run: vuelve a tener lugar.
    assert registry.run(call, ExecutionContext(tenant=ACME, run_id="run-b")).ok


# --- ffmpeg: lectura de pausas y momentos de los fotogramas -------------------


def test_parse_silences_pairs_start_and_end_and_drops_an_open_one():
    log = (
        "[silencedetect @ 0x1] silence_start: -0.01\n"
        "[silencedetect @ 0x1] silence_end: 0.8 | silence_duration: 0.81\n"
        "[silencedetect @ 0x1] silence_start: 12.4\n"
        "[silencedetect @ 0x1] silence_end: 13.25 | silence_duration: 0.85\n"
        "[silencedetect @ 0x1] silence_start: 59.9\n"
    )
    assert media.parse_silences(log) == [media.Silence(0.0, 0.8), media.Silence(12.4, 13.25)]


def test_frame_timestamps_are_spread_and_capped():
    assert media.frame_timestamps(10, 40) == [float(i) for i in range(10)] + [9.9]
    stamps = media.frame_timestamps(120, 40)
    assert len(stamps) == 40 and stamps[1] == 3.0 and stamps[-1] < 120
    assert media.frame_timestamps(0, 40) == []


def test_missing_ffmpeg_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(media.MediaToolMissingError):
        media.extract_audio(tmp_path / "v.mp4", tmp_path / "a.wav")


# --- Preparación ---------------------------------------------------------------

TRANSCRIPT = Transcript(language="es", words=[
    Word(start_s=0.5, end_s=0.9, text="Probé"), Word(start_s=0.9, end_s=1.2, text="el"),
    Word(start_s=1.2, end_s=1.8, text="café"),
])


@pytest.fixture
def video(tmp_path, monkeypatch):
    """Un video de 10 s guardado para acme, con ffmpeg y Whisper simulados."""
    data_dir, storage_dir = tmp_path / "data", tmp_path / "storage"
    for client_id in ("acme", "globex"):
        (data_dir / client_id).mkdir(parents=True)
    (data_dir / "acme" / "brand.md").write_text("# Acme\nTono cercano.", encoding="utf-8")
    monkeypatch.setattr(tenancy, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(storage_dir=storage_dir, max_video_mb=1))
    monkeypatch.setattr(video_prep, "get_settings", lambda: SimpleNamespace(video_max_frames=4))
    monkeypatch.setattr(media, "probe_duration", lambda path: 10.0)
    monkeypatch.setattr(storage, "probe_duration", lambda path: 10.0)

    calls = SimpleNamespace(frames=[], transcribe=0)

    def fake_frame(source, at, out):
        calls.frames.append(at)
        out.write_bytes(b"\xff\xd8" + str(at).encode())
        return out

    def fake_transcribe(audio):
        calls.transcribe += 1
        return TRANSCRIPT

    monkeypatch.setattr(media, "extract_frame", fake_frame)
    monkeypatch.setattr(media, "extract_audio", lambda source, out: out)
    monkeypatch.setattr(media, "detect_silences", lambda source: [media.Silence(4.0, 5.0)])
    monkeypatch.setattr(video_prep, "transcribe", fake_transcribe)

    import io
    ref = save_video(ACME, io.BytesIO(MP4), "crudo.mp4")
    return SimpleNamespace(ref=ref, calls=calls)


def test_prepare_extracts_frames_silences_and_transcript_and_caches_them(video):
    prep = video_prep.prepare_video(ACME, video.ref.id)

    assert [f.at_s for f in prep.frames] == [0.0, 2.5, 4.5, 5.0, 7.5]  # 4.5: adentro del silencio largo
    assert prep.silences[0].start_s == 4.0
    assert prep.transcript.text == "Probé el café"

    again = video_prep.prepare_video(ACME, video.ref.id)
    assert again == prep
    assert video.calls.transcribe == 1  # la segunda vez sale de disco

    images = video_prep.frame_images(ACME, prep)
    assert images[1].label == "Fotograma en 2.5 s" and images[1].data == b"\xff\xd82.5"


def test_prepare_does_not_cross_clients(video):
    with pytest.raises(storage.UnknownVideoError):
        video_prep.prepare_video(GLOBEX, video.ref.id)


# --- Plan y tool -----------------------------------------------------------------


def plan_dict(**overrides):
    plan = {
        "video_type": "recomendacion",
        "hook": "Arranca con la primera probada del café.",
        "segments": [
            {"start_s": 0.5, "end_s": 4.0, "purpose": "gancho", "zoom": True,
             "b_roll": [{"start_s": 6.0, "end_s": 8.0, "offset_s": 1.0, "description": "la taza"}]},
            {"start_s": 5.0, "end_s": 9.5, "purpose": "cierre", "zoom": False, "b_roll": []},
        ],
        "subtitle_style": "palabra",
        "on_screen_texts": [{"text": "café de especialidad", "at_s": 1.0}],
        "music": "acústica suave",
        "sound_effects": [],
        "call_to_action": "si pasás por la zona, andá",
        "target_network": "Reels",
        "warnings": ["Red no indicada: se asume Reels."],
    }
    return plan | overrides


def test_check_plan_rejects_times_outside_the_video():
    ok = VideoEditPlan.model_validate(plan_dict())
    assert video_edit.check_plan(ok, 10.0) == []
    assert ok.duration_s == 8.0

    bad = VideoEditPlan.model_validate(plan_dict(segments=[
        {"start_s": 8.0, "end_s": 30.0, "purpose": "x", "zoom": False, "b_roll": []},
        {"start_s": 3.0, "end_s": 2.0, "purpose": "y", "zoom": False, "b_roll": []},
    ]))
    problems = video_edit.check_plan(bad, 10.0)
    assert len(problems) == 2 and "tramo 1" in problems[0]


@pytest.fixture
def claude(monkeypatch):
    """Simula la llamada estructurada a Claude y registra qué recibió."""
    received = {}

    def fake_complete_structured(**kwargs):
        received.update(kwargs)
        usage = LLMResponse(text="", provider="anthropic", model="claude-opus-5-5", input_tokens=5000,
                            output_tokens=700, latency_ms=1.0, stop_reason="end_turn", run_id=kwargs["run_id"],
                            attempts=1)
        return StructuredResponse(data=VideoEditPlan.model_validate(received.get("plan", plan_dict())),
                                  response=usage)

    monkeypatch.setattr(video_edit, "complete_structured", fake_complete_structured)
    return received


def make_registry():
    return build_creative_registry(generation_timeout_s=5, video_edit=True, video_edit_timeout_s=5,
                                   video_edit_max_calls=2)


def run_tool(arguments, tenant=ACME, registry=None):
    registry = registry or make_registry()
    return registry.run(ToolCall(id="c1", tool_name="plan_video_edit", arguments=arguments),
                        ExecutionContext(tenant=tenant, run_id="run-1"))


def test_plan_tool_stops_after_the_per_turn_limit(video, monkeypatch):
    """Si el modelo entra en un bucle de reintentos, Opus se llama como mucho dos veces por mensaje."""
    calls = []

    def failing(**kwargs):
        calls.append(kwargs)
        raise TransientProviderError("anthropic: falla temporal")

    monkeypatch.setattr(video_edit, "complete_structured", failing)
    registry = make_registry()
    results = [run_tool({"video_id": video.ref.id}, registry=registry) for _ in range(4)]

    assert len(calls) == 2
    assert [r.metadata["error_type"] for r in results[2:]] == ["call_limit", "call_limit"]


def test_plan_tool_tells_the_model_not_to_retry_an_over_budget_video(video, monkeypatch):
    def over_budget(**kwargs):
        raise BudgetExceededError("supera el tope")

    monkeypatch.setattr(video_edit, "complete_structured", over_budget)
    result = run_tool({"video_id": video.ref.id})

    assert not result.ok and "No lo reintentes" in result.content


def test_plan_tool_returns_a_summary_and_keeps_the_full_plan_in_metadata(video, claude):
    result = run_tool({"video_id": video.ref.id, "notes": "es para TikTok"})

    assert result.ok, result.content
    summary = json.loads(result.content)
    assert summary["tipo"] == "recomendacion" and summary["duracion_estimada_s"] == 9.3
    assert summary["tramos"][0]["b_roll"] == 1
    assert result.metadata[VIDEO_PLAN_METADATA_KEY]["segments"][0]["b_roll"][0]["description"] == "la taza"
    assert result.metadata[MODEL_INPUT_TOKENS_KEY] == 5000
    # El plan queda guardado: render_video lo lee por su id, no por lo que diga el modelo.
    from yourebrand.creative.video_render import load_plan
    assert summary["plan_id"] == result.metadata["plan_id"]
    stored = load_plan(ACME, summary["plan_id"])
    assert stored.video_id == video.ref.id and stored.plan.duration_s == 9.3

    # A Claude le llegan la tarea video_edit, los fotogramas, la skill, la marca y la transcripción.
    assert claude["tier"] == "video_edit"
    assert len(claude["images"]) == 5
    assert "Edición de videos UGC" in claude["system"] and "Tono cercano" in claude["system"]
    assert "[0.50-0.90]Probé" in claude["prompt"] and "es para TikTok" in claude["prompt"]
    assert "4.00 a 5.00 s" in claude["prompt"]


def test_plan_tool_reports_an_impossible_plan_to_the_model(video, claude):
    claude["plan"] = plan_dict(segments=[{"start_s": 0, "end_s": 50, "purpose": "x", "zoom": False, "b_roll": []}])

    result = run_tool({"video_id": video.ref.id})

    assert not result.ok
    assert result.metadata["error_type"] == "InvalidPlanError"
    assert result.metadata[MODEL_INPUT_TOKENS_KEY] == 5000  # los tokens se gastaron igual


def test_plan_tool_does_not_see_other_clients_videos(video, claude):
    result = run_tool({"video_id": video.ref.id}, tenant=GLOBEX)

    assert not result.ok and result.metadata["error_type"] == "unknown_video"
    assert claude == {}


def test_plan_tool_rejects_a_malformed_id_before_running(video, claude):
    result = run_tool({"video_id": "../acme/videos"})

    assert not result.ok
    assert claude == {}


def test_plan_tool_explains_missing_ffmpeg(video, claude, monkeypatch):
    def missing(source):
        raise media.MediaToolMissingError("ffmpeg no está instalado")

    monkeypatch.setattr(media, "detect_silences", missing)
    result = run_tool({"video_id": video.ref.id})

    assert not result.ok and "ffmpeg" in result.content


def test_plan_tool_handles_provider_failure(video, monkeypatch):
    def failing(**kwargs):
        raise TransientProviderError("anthropic: falla temporal")

    monkeypatch.setattr(video_edit, "complete_structured", failing)
    result = run_tool({"video_id": video.ref.id})

    assert not result.ok and result.metadata["error_type"] == "TransientProviderError"


def test_registry_only_offers_the_tool_when_video_edit_is_enabled():
    names = lambda reg: [d["name"] for d in reg.function_declarations()]  # noqa: E731
    assert "plan_video_edit" in names(build_creative_registry(generation_timeout_s=5, video_edit=True))
    assert "plan_video_edit" not in names(build_creative_registry(generation_timeout_s=5, video_edit=False))


# --- Protección del habla: solo se corta en silencios largos ---------------------


def speech_prep(silences, words=(), duration=17.0):
    return video_prep.VideoPreparation(
        video_id="vid-000000000001", duration_s=duration, frames=[],
        silences=[video_prep.SilenceSpan(start_s=a, end_s=b) for a, b in silences],
        transcript=Transcript(language="es", words=[Word(start_s=a, end_s=b, text=t) for a, b, t in words]),
    )


def plan_with(*spans, **overrides):
    segments = [{"start_s": a, "end_s": b, "purpose": "x", "zoom": bool(i % 2), "b_roll": []}
                for i, (a, b) in enumerate(spans)]
    return VideoEditPlan.model_validate(plan_dict(segments=segments, on_screen_texts=[], warnings=[]) | overrides)


def spans(plan):
    return [(s.start_s, s.end_s) for s in plan.segments]


def test_a_cut_between_words_is_undone_and_the_end_is_kept():
    # El caso real: el modelo sacó "hola", una palabra del medio y todo el final.
    prep = speech_prep([(0.0, 0.8), (4.0, 4.5), (7.0, 9.0)])
    result = video_edit.protect_speech(plan_with((1.0, 2.0), (2.3, 5.6), (9.3, 11.0)), prep)

    assert spans(result) == [(0.65, 2.3), (2.3, 7.15), (8.85, 17.0)]


def test_a_short_pause_is_never_a_cut():
    prep = speech_prep([(4.0, 4.9)], duration=10.0)
    assert spans(video_edit.protect_speech(plan_with((0.0, 4.0), (4.9, 10.0)), prep)) == [(0.0, 4.9), (4.9, 10.0)]


def test_a_segment_across_a_long_silence_keeps_it():
    # Si en el silencio se muestra algo, el modelo lo conserva con un solo tramo.
    prep = speech_prep([(4.0, 7.0)], duration=10.0)
    assert spans(video_edit.protect_speech(plan_with((1.0, 9.0)), prep)) == [(0.0, 10.0)]


def test_contiguous_segments_only_change_the_zoom():
    prep = speech_prep([], duration=10.0)
    result = video_edit.protect_speech(plan_with((0.0, 3.0), (3.0, 6.0), (6.0, 10.0)), prep)
    assert spans(result) == [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    assert [s.zoom for s in result.segments] == [False, True, False]


def test_a_whole_block_left_out_is_reported():
    prep = speech_prep([(4.0, 6.0)], words=[(1.0, 2.0, "primera"), (7.0, 8.0, "repetida")], duration=10.0)
    result = video_edit.protect_speech(plan_with((1.0, 3.0)), prep)

    assert spans(result) == [(0.0, 4.15)]
    assert "repetida" in result.warnings[0]


def test_a_block_can_move_to_the_front_whole():
    prep = speech_prep([(4.0, 6.0)], duration=10.0)
    result = video_edit.protect_speech(plan_with((7.0, 8.0), (1.0, 2.0)), prep)
    assert spans(result) == [(5.85, 10.0), (0.0, 4.15)]


def test_on_screen_text_follows_its_moment():
    prep = speech_prep([(0.0, 2.0)], duration=10.0)
    text = {"text": "dato", "at_s": 1.0}
    result = video_edit.protect_speech(plan_with((3.0, 6.0), on_screen_texts=[text]), prep)
    # Era el segundo 4.0 del original: el video ahora arranca en 1.85.
    assert result.on_screen_texts[0].at_s == pytest.approx(2.15)


def test_a_long_result_warns_instead_of_cutting():
    prep = speech_prep([], duration=100.0)
    result = video_edit.protect_speech(plan_with((0.0, 50.0)), prep)
    assert spans(result) == [(0.0, 100.0)] and "100 s" in result.warnings[0]
