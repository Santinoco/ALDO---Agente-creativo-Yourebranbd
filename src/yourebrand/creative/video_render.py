"""Render de un video UGC a partir de un plan aprobado, sin modelo de lenguaje.

Tres partes:
- Planes guardados: plan_video_edit deja el plan en disco con un id, y el
  estado de la conversación guarda solo ese id. render_video lo lee de ahí.
- Armado: de un plan y la transcripción salen los subtítulos (archivo ASS) y
  los argumentos de ffmpeg. Son funciones puras, se prueban sin ffmpeg.
- Trabajo en segundo plano: el render tarda minutos, así que corre en un hilo
  aparte y deja su estado en disco. La tool vuelve enseguida con el id del
  render; la pantalla consulta el estado.

Lo que el render hace con cada parte del plan:
- Tramos: se cortan del original y se concatenan en el orden del plan, con
  cortes limpios (la skill no admite transiciones).
- Zoom: un acercamiento fijo del tramo (punch-in), el recurso típico de UGC
  para dar movimiento a un plano de alguien hablando.
- B-roll: el tramo del original que indica, encima de la imagen, desde
  `offset_s`. La voz sigue siendo la del tramo.
- Subtítulos: palabra por palabra o en frases cortas, con los tiempos de la
  transcripción, un poco por debajo del medio de la pantalla.
- Textos en pantalla: arriba, resaltados, durante unos segundos.
- Música y efectos: todavía no. Falta la librería propia de temas de libre
  uso; el render lo avisa en sus advertencias.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yourebrand.core import media
from yourebrand.core.storage import (
    UnknownPlanError,
    new_plan_id,
    new_render_id,
    plan_path,
    render_dir,
    video_file,
)
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.trace import emit
from yourebrand.core.transcription import Word
from yourebrand.core.video_prep import prepare_video
from yourebrand.creative.schemas import VideoEditPlan

logger = logging.getLogger(__name__)

WIDTH, HEIGHT, FPS = 1080, 1920, 30  # 9:16, el formato técnico de la skill
ZOOM_FACTOR = 1.12
SUBTITLES_FILENAME = "subs.ass"
OUTPUT_FILENAME = "output.mp4"
STATUS_FILENAME = "status.json"
ON_SCREEN_TEXT_S = 2.0
PHRASE_MAX_WORDS = 4
PHRASE_GAP_S = 0.4  # una pausa más larga corta la frase
WORD_HOLD_S = 0.3  # una palabra sigue en pantalla hasta la siguiente si la pausa es menor


# --- Planes guardados --------------------------------------------------------


class StoredPlan(BaseModel):
    """Un plan de edición guardado. render_video solo recibe su id."""

    plan_id: str
    video_id: str
    run_id: str  # el turno en que se armó, para la traza
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    plan: VideoEditPlan


def store_plan(tenant: TenantContext, video_id: str, plan: VideoEditPlan, *, run_id: str) -> StoredPlan:
    stored = StoredPlan(plan_id=new_plan_id(), video_id=video_id, run_id=run_id, plan=plan)
    plan_path(tenant, stored.plan_id).write_text(stored.model_dump_json(indent=2), encoding="utf-8")
    return stored


def load_plan(tenant: TenantContext, plan_id: str) -> StoredPlan:
    """Plan del cliente. Lanza UnknownPlanError si no existe para este cliente."""
    path = plan_path(tenant, plan_id)
    if not path.is_file():
        raise UnknownPlanError(f"no hay un plan {plan_id!r} para este cliente")
    return StoredPlan.model_validate_json(path.read_text(encoding="utf-8"))


# --- Línea de tiempo y subtítulos ---------------------------------------------


@dataclass(frozen=True)
class Cue:
    """Un texto en pantalla, con tiempos del video final."""

    start_s: float
    end_s: float
    text: str


def segment_starts(plan: VideoEditPlan) -> list[float]:
    """Dónde empieza cada tramo en el video final: la suma de los anteriores."""
    starts, elapsed = [], 0.0
    for segment in plan.segments:
        starts.append(round(elapsed, 3))
        elapsed += segment.end_s - segment.start_s
    return starts


def subtitle_cues(plan: VideoEditPlan, words: list[Word]) -> list[Cue]:
    """Subtítulos en tiempos del video final.

    Cada palabra cuyo centro cae dentro de un tramo se traslada al lugar que
    ocupa ese tramo en el video final. Si un tramo se repite, sus palabras
    también.
    """
    placed: list[Cue] = []
    for segment, final_start in zip(plan.segments, segment_starts(plan)):
        for word in words:
            middle = (word.start_s + word.end_s) / 2
            text = word.text.strip()
            if not text or not (segment.start_s <= middle < segment.end_s):
                continue
            start = final_start + max(word.start_s, segment.start_s) - segment.start_s
            end = final_start + min(word.end_s, segment.end_s) - segment.start_s
            placed.append(Cue(round(start, 3), round(max(end, start + 0.05), 3), text))

    if plan.subtitle_style == "frase":
        return _group_phrases(placed)
    # Palabra por palabra: cada una se queda hasta la siguiente si la pausa es
    # corta, así el texto no parpadea entre palabras.
    held = []
    for current, following in zip(placed, [*placed[1:], None]):
        end = current.end_s
        if following is not None and 0 <= following.start_s - end < WORD_HOLD_S:
            end = following.start_s
        held.append(Cue(current.start_s, end, current.text))
    return held


def _group_phrases(words: list[Cue]) -> list[Cue]:
    phrases: list[list[Cue]] = []
    for word in words:
        if (
            phrases
            and len(phrases[-1]) < PHRASE_MAX_WORDS
            and word.start_s - phrases[-1][-1].end_s < PHRASE_GAP_S
            and not phrases[-1][-1].text.endswith((".", "?", "!"))
        ):
            phrases[-1].append(word)
        else:
            phrases.append([word])
    return [Cue(group[0].start_s, group[-1].end_s, " ".join(w.text for w in group)) for group in phrases]


def on_screen_cues(plan: VideoEditPlan) -> list[Cue]:
    total = plan.duration_s
    return [
        Cue(round(item.at_s, 3), round(min(item.at_s + ON_SCREEN_TEXT_S, total), 3), item.text)
        for item in plan.on_screen_texts
        if 0 <= item.at_s < total
    ]


def _ass_time(seconds: float) -> str:
    centis = round(max(seconds, 0) * 100)
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    # Llaves y barra invertida son comandos en ASS: un texto de la transcripción
    # no puede cambiar el estilo ni la posición.
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(subtitles: list[Cue], highlights: list[Cue]) -> str:
    """Archivo de subtítulos ASS: estilos de la skill y un evento por texto.

    Sub: blanco con borde negro, centrado, con el borde inferior en el 63 % de
    la altura (un poco por debajo del medio). Highlight: textos resaltados,
    arriba, para no pisar los subtítulos.
    """
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\nPlayResY: {HEIGHT}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Sub,Arial,96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,"
        "90,90,710,1\n"
        "Style: Highlight,Arial,78,&H0000D7FF,&H0000D7FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,8,"
        "90,90,330,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{_ass_time(c.start_s)},{_ass_time(c.end_s)},{style},,0,0,0,,{_ass_text(c.text)}"
        for style, cues in (("Sub", subtitles), ("Highlight", highlights))
        for c in cues
    ]
    return header + "\n".join(lines) + "\n"


# --- Argumentos de ffmpeg ------------------------------------------------------


def _frame_filter(zoom: bool) -> str:
    """Lleva cualquier encuadre a 1080x1920 sin bordes: escala hasta cubrir y recorta al centro.

    Con zoom escala un poco más antes de recortar: el acercamiento fijo.
    """
    factor = ZOOM_FACTOR if zoom else 1.0
    w, h = round(WIDTH * factor / 2) * 2, round(HEIGHT * factor / 2) * 2
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},"
            f"setsar=1,fps={FPS},format=yuv420p")


def build_ffmpeg_args(plan: VideoEditPlan, source: Path, *, has_audio: bool) -> list[str]:
    """Argumentos de ffmpeg para renderizar el plan (sin el ejecutable ni -y).

    Cada tramo y cada b-roll entra como una entrada propia con -ss/-t: ffmpeg
    salta directo a ese momento en vez de decodificar el video entero cada vez.
    Se corre desde la carpeta del render, donde está el archivo de subtítulos.
    """
    inputs: list[str] = []
    filters: list[str] = []
    concat: list[str] = []
    index = 0
    for i, segment in enumerate(plan.segments):
        length = round(segment.end_s - segment.start_s, 3)
        inputs += ["-ss", f"{segment.start_s:.3f}", "-t", f"{length:.3f}", "-i", str(source)]
        seg_input = index
        index += 1
        filters.append(f"[{seg_input}:v]{_frame_filter(segment.zoom)},trim=duration={length},"
                       f"setpts=PTS-STARTPTS[v{i}]")
        current = f"v{i}"
        for j, broll in enumerate(segment.b_roll):
            visible = round(min(broll.end_s - broll.start_s, length - broll.offset_s), 3)
            if visible <= 0:
                continue
            inputs += ["-ss", f"{broll.start_s:.3f}", "-t", f"{visible:.3f}", "-i", str(source)]
            filters.append(f"[{index}:v]{_frame_filter(False)},trim=duration={visible},"
                           f"setpts=PTS-STARTPTS+{broll.offset_s:.3f}/TB[b{i}_{j}]")
            end = round(broll.offset_s + visible, 3)
            filters.append(f"[{current}][b{i}_{j}]overlay=eof_action=pass:"
                           f"enable='between(t,{broll.offset_s:.3f},{end})'[v{i}_{j}]")
            current = f"v{i}_{j}"
            index += 1
        # El audio de cada tramo dura exactamente lo mismo que su imagen: si no,
        # el desfase se acumula de un tramo al siguiente.
        if has_audio:
            filters.append(f"[{seg_input}:a]aresample=48000,aformat=channel_layouts=stereo,"
                           f"apad=whole_dur={length},atrim=duration={length},asetpts=PTS-STARTPTS[a{i}]")
        else:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={length}[a{i}]")
        concat += [f"[{current}]", f"[a{i}]"]

    filters.append(f"{''.join(concat)}concat=n={len(plan.segments)}:v=1:a=1[vc][ac]")
    filters.append(f"[vc]ass={SUBTITLES_FILENAME}[vout]")
    return [
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[ac]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        OUTPUT_FILENAME,
    ]


def render_warnings(plan: VideoEditPlan) -> list[str]:
    """Lo que el plan pide y el render todavía no hace. Se muestra junto al video."""
    warnings = [f"La música («{plan.music}») no se agregó: todavía no hay una librería de temas de libre "
                "uso cargada. Sumala en la edición final."]
    if plan.sound_effects:
        warnings.append("Los efectos de sonido del plan no se agregaron: " + ", ".join(plan.sound_effects) + ".")
    return warnings


# --- Trabajo en segundo plano --------------------------------------------------

RenderState = Literal["queued", "running", "done", "failed"]


class RenderStatus(BaseModel):
    """Estado de un render. Vive en disco, junto al video que produce."""

    render_id: str
    plan_id: str
    video_id: str
    run_id: str
    state: RenderState
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    duration_s: float | None = None  # del video final
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


class PlanNotApprovedError(PermissionError):
    """Se pidió renderizar sin la aprobación de una persona."""


class RenderNotReadyError(RuntimeError):
    """El render todavía no terminó o falló: no hay video para entregar."""


# Un render a la vez: ffmpeg ya usa todos los núcleos, y dos en paralelo
# tardarían lo mismo que uno detrás del otro.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
# Renders vivos en este proceso. Un estado "queued" o "running" que no está acá
# quedó de un proceso anterior que se cortó: se informa como fallido.
_ACTIVE: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


def _write_status(tenant: TenantContext, status: RenderStatus) -> None:
    path = render_dir(tenant, status.render_id) / STATUS_FILENAME
    partial = path.with_suffix(".part")
    partial.write_text(status.model_dump_json(indent=2), encoding="utf-8")
    partial.replace(path)  # nadie lee un estado escrito a medias


def start_render(
    tenant: TenantContext,
    plan_id: str,
    *,
    run_id: str,
    approved: bool,
    executor: Executor | None = None,
) -> RenderStatus:
    """Encola el render de un plan aprobado y devuelve su estado inicial sin esperar.

    `approved` lo pasa la tool desde el contexto que arma el grafo: es la
    segunda barrera, por si algo llamara a esta función sin pasar por la pausa.
    """
    if not approved:
        raise PlanNotApprovedError("el plan no está aprobado")
    stored = load_plan(tenant, plan_id)  # UnknownPlanError si no es de este cliente
    render_id = new_render_id()
    render_dir(tenant, render_id, create=True)
    status = RenderStatus(render_id=render_id, plan_id=plan_id, video_id=stored.video_id, run_id=run_id,
                          state="queued", warnings=render_warnings(stored.plan))
    _write_status(tenant, status)
    with _ACTIVE_LOCK:
        _ACTIVE.add(render_id)
    emit(logger, "render_queued", run_id=run_id, client_id=tenant.client_id, render_id=render_id,
         plan_id=plan_id, video_id=stored.video_id, segments=len(stored.plan.segments))
    (executor or _EXECUTOR).submit(_run_render, tenant, stored, status)
    return status


def _run_render(tenant: TenantContext, stored: StoredPlan, status: RenderStatus,
                runner: Callable[..., None] | None = None) -> RenderStatus:
    """Hace el render y deja el estado final en disco. Nunca lanza: el error queda en el estado."""
    started = time.perf_counter()
    folder = render_dir(tenant, status.render_id)
    status = status.model_copy(update={"state": "running"})
    _write_status(tenant, status)
    try:
        # La transcripción ya está en disco desde el plan: no vuelve a correr Whisper.
        prep = prepare_video(tenant, stored.video_id)
        source = video_file(tenant, stored.video_id)
        ass = build_ass(subtitle_cues(stored.plan, prep.transcript.words), on_screen_cues(stored.plan))
        (folder / SUBTITLES_FILENAME).write_text(ass, encoding="utf-8")
        args = build_ffmpeg_args(stored.plan, source, has_audio=media.has_audio(source))
        (runner or media.run_render)(args, cwd=folder)
        output = folder / OUTPUT_FILENAME
        if not output.is_file():
            raise media.MediaProcessingError("ffmpeg terminó sin generar el video")
        status = status.model_copy(update={"state": "done", "duration_s": media.probe_duration(output),
                                           "finished_at": datetime.now(UTC)})
    except Exception as exc:  # noqa: BLE001 - corre en un hilo: si lanzara, nadie lo vería
        status = status.model_copy(update={"state": "failed", "error": str(exc) or type(exc).__name__,
                                           "finished_at": datetime.now(UTC)})
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(status.render_id)
    _write_status(tenant, status)
    emit(logger, "render_finished", level=logging.INFO if status.state == "done" else logging.ERROR,
         run_id=status.run_id, client_id=tenant.client_id, render_id=status.render_id, state=status.state,
         error=status.error, latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return status


def get_render(tenant: TenantContext, render_id: str) -> RenderStatus:
    """Estado de un render del cliente. Lanza UnknownRenderError si no es de este cliente."""
    path = render_dir(tenant, render_id) / STATUS_FILENAME
    status = RenderStatus.model_validate_json(path.read_text(encoding="utf-8"))
    if status.state in ("queued", "running"):
        with _ACTIVE_LOCK:
            alive = render_id in _ACTIVE
        if not alive:
            status = status.model_copy(update={"state": "failed", "finished_at": datetime.now(UTC),
                                               "error": "el render se interrumpió (la app se reinició)"})
            _write_status(tenant, status)
    return status


def render_output(tenant: TenantContext, render_id: str) -> Path:
    """Ruta del video final. Lanza RenderNotReadyError si todavía no está."""
    status = get_render(tenant, render_id)
    output = render_dir(tenant, render_id) / OUTPUT_FILENAME
    if status.state != "done" or not output.is_file():
        raise RenderNotReadyError(f"el render está en estado {status.state}")
    return output
