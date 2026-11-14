"""Lectura de archivos de video: qué formato son y cuánto duran.

No edita nada: eso lo hará el render. Sin dependencias de Python: la firma del
archivo se lee a mano y la duración, con ffprobe si está instalado.
"""

import json
import logging
import shutil
import subprocess
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
