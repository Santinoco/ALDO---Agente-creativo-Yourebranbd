"""Lectura de archivos de video: qué formato son y cuánto duran.

Edita solo a pedido del render (`run_render`), que le pasa los argumentos ya
armados. Sin dependencias de Python: la firma del
archivo se lee a mano; la duración, los fotogramas, el audio y las pausas salen
de ffprobe y ffmpeg, que se instalan aparte. Este módulo no arma rutas: las
recibe de core/storage.py.
"""

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

VideoContainer = Literal["mp4", "mov"]

# Bytes que hacen falta del principio del archivo para reconocer el formato.
HEADER_BYTES = 12
FFPROBE_TIMEOUT_S = 30

# MP4 y MOV comparten estructura (cajas de ISO/QuickTime): a partir del byte 4
# va el tipo de la primera caja. Lo normal es "ftyp", seguido de la marca del
# formato; "qt  " es QuickTime (los .mov del iPhone).
_QUICKTIME_BRAND = b"qt  "
# Algunos .mov viejos arrancan directo con otra caja de QuickTime, sin "ftyp".
_QUICKTIME_FIRST_BOXES = {b"moov", b"mdat", b"wide", b"free"}


def detect_container(header: bytes) -> VideoContainer | None:
    """Formato del video según sus primeros bytes, o None si no es un MP4 ni un MOV.

    Se mira el contenido y no la extensión ni el tipo que manda el navegador:
    los dos los elige quien sube el archivo.
    """
    if len(header) < HEADER_BYTES:
        return None
    box_type = header[4:8]
    if box_type == b"ftyp":
        return "mov" if header[8:12] == _QUICKTIME_BRAND else "mp4"
    if box_type in _QUICKTIME_FIRST_BOXES:
        return "mov"
    return None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe_duration(path: Path) -> float | None:
    """Duración en segundos, o None si ffprobe no está instalado o no la pudo leer.

    No lanza: la duración es un dato útil, no una condición para guardar el video.
    """
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True,
            timeout=FFPROBE_TIMEOUT_S,
            check=True,
        )
        return round(float(json.loads(completed.stdout)["format"]["duration"]), 2)
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
        logger.warning("ffprobe_failed error=%s", type(exc).__name__)
        return None


# --- Preparación para el plan de edición (sin modelo) -----------------------
#
# Todo pasa por ffmpeg como proceso aparte: no hay bindings de Python que
# instalar y es la misma herramienta que usará el render.

FFMPEG_TIMEOUT_S = 300
# Umbral de silencio: por debajo de -30 dB durante medio segundo cuenta como
# pausa. Es el corte que marca el creador en un video largo (ver la skill).
SILENCE_NOISE_DB = -30
SILENCE_MIN_S = 0.5
FRAME_WIDTH_PX = 512  # alcanza para ver encuadre, caras y producto; ahorra tokens

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)")


class MediaError(Exception):
    """Base de los errores al procesar un video."""


class MediaToolMissingError(MediaError):
    """ffmpeg no está instalado en la máquina."""


class MediaProcessingError(MediaError):
    """ffmpeg no pudo procesar el archivo."""


@dataclass(frozen=True)
class Silence:
    start_s: float
    end_s: float


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MediaToolMissingError("ffmpeg no está instalado: hace falta para preparar el video")
    return executable


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run([_ffmpeg(), "-hide_banner", *args], capture_output=True,
                              timeout=FFMPEG_TIMEOUT_S, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        # El detalle de ffmpeg puede incluir rutas del cliente: no va al mensaje.
        logger.warning("ffmpeg_failed error=%s", type(exc).__name__)
        raise MediaProcessingError(f"ffmpeg no pudo procesar el video ({type(exc).__name__})") from exc


def frame_timestamps(duration_s: float, max_frames: int, min_interval_s: float = 1.0) -> list[float]:
    """Momentos parejos a lo largo del video: uno cada `min_interval_s` como mínimo, hasta `max_frames`."""
    if duration_s <= 0 or max_frames <= 0:
        return []
    interval = max(min_interval_s, duration_s / max_frames)
    count = min(max_frames, int(duration_s / interval) + 1)
    # El último fotograma queda un poco antes del final: justo en el final no siempre hay imagen.
    return [round(min(i * interval, max(duration_s - 0.1, 0)), 2) for i in range(count)]


def extract_frame(video: Path, at_s: float, out: Path) -> Path:
    """Guarda en `out` (JPEG) el fotograma del segundo `at_s`, achicado a FRAME_WIDTH_PX de ancho."""
    _run_ffmpeg(["-v", "error", "-y", "-ss", f"{at_s:.2f}", "-i", str(video), "-frames:v", "1",
                 "-vf", f"scale={FRAME_WIDTH_PX}:-2", "-q:v", "4", str(out)])
    if not out.is_file():
        raise MediaProcessingError(f"no hay fotograma en {at_s:.2f} s")
    return out


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Ancho y alto de un JPEG leyendo su encabezado, o None si no se encuentra.

    Recorre los segmentos hasta el marcador SOF, que es donde el formato guarda
    el tamaño. Evita sumar una librería de imágenes solo para esto.
    """
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        # SOF0 a SOF15, salvo DHT (C4), JPG (C8) y DAC (CC), que no son de tamaño.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        i += 2 + length
    return None


def extract_audio(video: Path, out: Path) -> Path:
    """Audio en WAV mono a 16 kHz: el formato que espera Whisper."""
    _run_ffmpeg(["-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out)])
    return out


def detect_silences(video: Path) -> list[Silence]:
    """Pausas del audio: en un video largo, son los cortes que marcó el creador."""
    completed = _run_ffmpeg(["-i", str(video), "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_S}",
                             "-f", "null", "-"])
    return parse_silences(completed.stderr.decode("utf-8", errors="replace"))


def parse_silences(log: str) -> list[Silence]:
    """Lee la salida de silencedetect. Un silencio sin final (llega hasta el cierre) se descarta."""
    silences: list[Silence] = []
    start: float | None = None
    for line in log.splitlines():
        if match := _SILENCE_START.search(line):
            start = max(0.0, float(match.group(1)))
        elif (match := _SILENCE_END.search(line)) and start is not None:
            silences.append(Silence(round(start, 2), round(float(match.group(1)), 2)))
            start = None
    return silences


def has_audio(path: Path) -> bool:
    """Si el video tiene pista de audio. Sin ffprobe, se asume que sí (lo normal en un UGC)."""
    executable = shutil.which("ffprobe")
    if executable is None:
        return True
    try:
        completed = subprocess.run(
            [executable, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(path)],
            capture_output=True, timeout=FFPROBE_TIMEOUT_S, check=True,
        )
        return bool(completed.stdout.strip())
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffprobe_failed error=%s", type(exc).__name__)
        return True


# --- Render -----------------------------------------------------------------

RENDER_TIMEOUT_S = 1800  # un video de 1:20 en 1080x1920 en CPU tarda minutos, no media hora


def run_render(args: list[str], *, cwd: Path, timeout_s: float = RENDER_TIMEOUT_S) -> None:
    """Corre ffmpeg con los argumentos del render, desde `cwd`.

    `cwd` es la carpeta del render: el filtro de subtítulos recibe el archivo
    por nombre relativo, porque escapar una ruta de Windows ("C:\\...") adentro
    de un filtro de ffmpeg es frágil.
    """
    try:
        subprocess.run([_ffmpeg(), "-hide_banner", "-v", "error", "-y", *args], cwd=cwd,
                       capture_output=True, timeout=timeout_s, check=True)
    except subprocess.CalledProcessError as exc:
        # La última línea de ffmpeg dice qué falló; va al log, que no lleva
        # contenido del cliente (son nombres de filtros y códecs).
        tail = exc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:] if exc.stderr else []
        logger.warning("ffmpeg_render_failed returncode=%s detail=%s", exc.returncode, tail)
        raise MediaProcessingError(f"ffmpeg no pudo renderizar el video (código {exc.returncode})") from exc
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg_render_failed error=%s", type(exc).__name__)
        raise MediaProcessingError(f"ffmpeg no pudo renderizar el video ({type(exc).__name__})") from exc
