"""Transcripción del audio con tiempos por palabra, en la misma máquina.

Usa faster-whisper (Whisper de OpenAI reimplementado sobre CTranslate2): corre
en CPU, no manda el audio a ningún servicio y da el tiempo de cada palabra,
que es lo que necesitan el plan (dónde cortar) y los subtítulos palabra por
palabra. Es una dependencia opcional (`pip install -e ".[video]"`): sin ella
la app funciona, solo que sin edición de video.
"""

import logging
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from yourebrand.core.config import get_settings

logger = logging.getLogger(__name__)


class TranscriptionUnavailableError(RuntimeError):
    """faster-whisper no está instalado o no se pudo cargar el modelo."""


class TranscriptionFailedError(RuntimeError):
    """El modelo no pudo transcribir el audio."""


class Word(BaseModel):
    start_s: float
    end_s: float
    text: str


class Transcript(BaseModel):
    language: str | None
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


@lru_cache
def _load_model(name: str) -> Any:
    """Carga el modelo una sola vez por proceso: tarda segundos y ocupa memoria."""
    try:
        from faster_whisper import WhisperModel  # dependencia opcional
    except ImportError as exc:
        raise TranscriptionUnavailableError(
            'falta faster-whisper: instalalo con pip install -e ".[video]"'
        ) from exc
    try:
        # int8 en CPU: anda en cualquier máquina sin GPU, con poca pérdida de calidad.
        return WhisperModel(name, device="cpu", compute_type="int8")
    except Exception as exc:  # la librería no documenta qué lanza al cargar
        raise TranscriptionUnavailableError(f"no se pudo cargar el modelo de Whisper ({type(exc).__name__})") from exc


def _read_wav(path: Path) -> Any:
    """Muestras de un WAV PCM de 16 bits como float32 entre -1 y 1, como las espera Whisper."""
    import numpy as np  # viene con faster-whisper

    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise TranscriptionFailedError("el audio no es WAV mono de 16 bits")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio: Path, *, language: str | None = "es") -> Transcript:
    """Transcribe un WAV y devuelve cada palabra con su inicio y fin en segundos."""
    settings = get_settings()
    model = _load_model(settings.whisper_model)
    try:
        # Se le pasan las muestras y no la ruta: con la ruta, faster-whisper
        # decodifica con PyAV, y la versión 1.2 choca con PyAV 16. El WAV ya
        # viene en el formato que espera (mono, 16 kHz) desde core/media.py.
        # vad_filter: transcribe solo donde detecta voz. Sin él, sobre música o
        # ruido Whisper inventa frases (en la prueba con un video sin voz devolvió
        # "¡Suscríbete!"), y el plan cortaría sobre palabras que nadie dijo.
        segments, info = model.transcribe(
            _read_wav(audio), language=language, word_timestamps=True, vad_filter=True
        )
        # segments es un generador: la transcripción ocurre al recorrerlo.
        words = [
            Word(start_s=round(w.start, 2), end_s=round(w.end, 2), text=w.word.strip())
            for segment in segments
            for w in (segment.words or [])
            if w.word.strip()
        ]
    except Exception as exc:
        raise TranscriptionFailedError(f"no se pudo transcribir el audio ({type(exc).__name__})") from exc
    return Transcript(language=getattr(info, "language", language), words=words)
