"""Preparación de un video para el plan de edición, sin modelo de lenguaje.

Del archivo original saca lo que el modelo del plan puede leer (no recibe video
ni audio): fotogramas como imágenes, la transcripción con tiempos por palabra y
las pausas. El resultado queda en disco junto al video: preparar es lento
(Whisper tarda) y el plan se puede pedir más de una vez.
"""

import logging
import time
from pathlib import Path

from pydantic import BaseModel

from yourebrand.core import media
from yourebrand.core.config import get_settings
from yourebrand.core.providers import ImageInput
from yourebrand.core.storage import get_video, prep_dir, video_file
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.trace import emit
from yourebrand.core.transcription import Transcript, transcribe

logger = logging.getLogger(__name__)

PREP_FILENAME = "prep.json"
# Sube cuando cambia cómo se prepara: invalida lo guardado con la versión anterior.
PREP_VERSION = 3  # 2: transcripción con filtro de voz (VAD). 3: fotogramas dentro de los silencios largos
# Con un fotograma en cada silencio largo, el modelo puede ver si ahí se muestra
# algo antes de decidir cortarlo. Con tope, porque cada imagen suma costo.
SILENCE_FRAME_MIN_S = 1.0
MAX_SILENCE_FRAMES = 10


class Frame(BaseModel):
    at_s: float
    filename: str  # dentro de la carpeta prep/ del video


class SilenceSpan(BaseModel):
    start_s: float
    end_s: float


class VideoPreparation(BaseModel):
    """Lo que se sabe de un video para planear su edición. Las imágenes quedan en disco."""

    version: int = PREP_VERSION
    video_id: str
    duration_s: float
    frames: list[Frame]
    silences: list[SilenceSpan]
    transcript: Transcript


def prepare_video(tenant: TenantContext, video_id: str, *, max_frames: int | None = None) -> VideoPreparation:
    """Prepara el video (o devuelve la preparación guardada) del cliente `tenant`.

    Lanza UnknownVideoError si el video no es de este cliente, MediaError si
    ffmpeg falta o falla, y los errores de transcripción si Whisper no está.
    """
    ref = get_video(tenant, video_id)
    folder = prep_dir(tenant, video_id)
    cached = folder / PREP_FILENAME
    if cached.is_file():
        prep = VideoPreparation.model_validate_json(cached.read_text(encoding="utf-8"))
        if prep.version == PREP_VERSION and all((folder / f.filename).is_file() for f in prep.frames):
            return prep

    started = time.perf_counter()
    source = video_file(tenant, video_id)
    duration = ref.duration_s or media.probe_duration(source)
    if not duration:
        raise media.MediaProcessingError("no se pudo leer la duración del video (¿falta ffprobe?)")

    limit = max_frames or get_settings().video_max_frames
    silences = [SilenceSpan(start_s=s.start_s, end_s=s.end_s) for s in media.detect_silences(source)]
    long_silences = [s for s in silences if s.end_s - s.start_s >= SILENCE_FRAME_MIN_S][:MAX_SILENCE_FRAMES]
    moments = {*media.frame_timestamps(duration, limit), *(round((s.start_s + s.end_s) / 2, 2) for s in long_silences)}
    frames = []
    for i, at in enumerate(sorted(moments)):
        name = f"frame_{i:03d}.jpg"
        media.extract_frame(source, at, folder / name)
        frames.append(Frame(at_s=at, filename=name))

    audio = media.extract_audio(source, folder / "audio.wav")
    transcript = transcribe(audio)

    prep = VideoPreparation(video_id=video_id, duration_s=duration, frames=frames, silences=silences,
                            transcript=transcript)
    cached.write_text(prep.model_dump_json(indent=2), encoding="utf-8")
    # Solo cantidades: la transcripción es contenido del cliente.
    emit(logger, "video_prepared", client_id=tenant.client_id, video_id=video_id, frames=len(frames),
         silences=len(silences), words=len(transcript.words),
         latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return prep


def frame_images(tenant: TenantContext, prep: VideoPreparation) -> list[ImageInput]:
    """Los fotogramas como imágenes para el modelo, cada una rotulada con su momento."""
    folder: Path = prep_dir(tenant, prep.video_id)
    images = []
    for f in prep.frames:
        data = (folder / f.filename).read_bytes()
        width, height = media.jpeg_size(data) or (None, None)
        images.append(ImageInput(data=data, media_type="image/jpeg", label=f"Fotograma en {f.at_s:.1f} s",
                                 width=width, height=height))
    return images
