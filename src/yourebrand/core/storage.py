"""Almacenamiento de videos, aislado por cliente. Único módulo que arma rutas de videos.

Los videos viven en la máquina donde corre la app:

    <storage_dir>/<client_id>/videos/<video_id>/
        original.mp4    el archivo que subió la persona
        meta.json       sus datos (VideoRef)
        prep/           lo que se deriva para el plan: fotogramas, audio,
                        transcripción y pausas (ver core/video_prep.py)

Toda ruta sale de un TenantContext validado y de un video_id con formato fijo:
un id armado a mano no puede salir de la carpeta del cliente. Si más adelante
los videos pasan a la nube, cambia este módulo y nada más.
"""

import logging
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field

from yourebrand.core.config import get_settings
from yourebrand.core.media import HEADER_BYTES, VideoContainer, detect_container, probe_duration
from yourebrand.core.tenancy import TenantContext, client_data_dir
from yourebrand.core.trace import emit

logger = logging.getLogger(__name__)

VIDEO_ID_PATTERN = re.compile(r"vid-[0-9a-f]{12}")
META_FILENAME = "meta.json"
_CHUNK_BYTES = 1024 * 1024
_MAX_FILENAME_CHARS = 120
_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*\\/]")


class VideoStorageError(Exception):
    """Base de los errores de este módulo."""


class InvalidVideoError(VideoStorageError):
    """El archivo no es un video que se pueda usar (formato no admitido o vacío)."""


class VideoTooLargeError(VideoStorageError):
    """El archivo supera el tamaño máximo configurado."""


class UnknownVideoError(VideoStorageError, LookupError):
    """No hay un video con ese id para este cliente."""


class VideoRef(BaseModel):
    """Lo que se sabe de un video guardado. Es lo que guarda el estado, nunca el archivo."""

    id: str
    filename: str  # nombre con el que se subió, limpio: solo para mostrarlo
    container: VideoContainer
    size_bytes: int
    duration_s: float | None = None  # None si no hay ffprobe instalado
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _videos_root(tenant: TenantContext) -> Path:
    # El client_id ya está validado por TenantContext (solo minúsculas,
    # números y guiones), así que no puede subir de carpeta.
    return get_settings().storage_dir / tenant.client_id / "videos"


def _video_dir(tenant: TenantContext, video_id: str) -> Path:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise UnknownVideoError("id de video inválido")
    return _videos_root(tenant) / video_id


def clean_filename(filename: str | None) -> str:
    """Nombre para mostrar: sin carpetas, sin caracteres raros y con largo acotado.

    El nombre lo elige quien sube el archivo y termina en el mensaje que lee el
    modelo, así que se reduce a algo inofensivo.
    """
    name = Path(filename or "").name
    name = _UNSAFE_FILENAME_CHARS.sub("", name).strip()
    return name[:_MAX_FILENAME_CHARS] or "video"


def save_video(tenant: TenantContext, source: BinaryIO, filename: str | None) -> VideoRef:
    """Guarda un video subido y devuelve su referencia.

    Lee el archivo de a partes: un video de cientos de MB nunca entra entero en
    memoria. Si algo falla, no deja archivos a medio escribir.
    """
    client_data_dir(tenant)  # lanza UnknownClientError si el cliente no existe
    max_bytes = get_settings().max_video_mb * 1024 * 1024

    header = source.read(HEADER_BYTES)
    container = detect_container(header)
    if container is None:
        raise InvalidVideoError("el archivo no es un video MP4 ni MOV")

    video_id = f"vid-{uuid.uuid4().hex[:12]}"
    folder = _video_dir(tenant, video_id)
    folder.mkdir(parents=True)
    partial = folder / f"original.{container}.part"
    size = 0
    try:
        with partial.open("wb") as out:
            out.write(header)
            size = len(header)
            while chunk := source.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise VideoTooLargeError(f"el video supera los {get_settings().max_video_mb} MB")
                out.write(chunk)
        final = folder / f"original.{container}"
        partial.rename(final)
        ref = VideoRef(
            id=video_id,
            filename=clean_filename(filename),
            container=container,
            size_bytes=size,
            duration_s=probe_duration(final),
        )
        (folder / META_FILENAME).write_text(ref.model_dump_json(indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    # Se registran datos del archivo, nunca su nombre (puede decir algo del cliente).
    emit(logger, "video_saved", client_id=tenant.client_id, video_id=video_id, container=container,
         size_bytes=size, duration_s=ref.duration_s)
    return ref


def get_video(tenant: TenantContext, video_id: str) -> VideoRef:
    """Referencia de un video del cliente. Falla si el id no existe para ese cliente."""
    meta = _video_dir(tenant, video_id) / META_FILENAME
    if not meta.is_file():
        raise UnknownVideoError(f"no hay un video {video_id!r} para este cliente")
    return VideoRef.model_validate_json(meta.read_text(encoding="utf-8"))


def video_file(tenant: TenantContext, video_id: str) -> Path:
    """Ruta del archivo original, para quien lo procese (fotogramas, audio, render)."""
    ref = get_video(tenant, video_id)
    return _video_dir(tenant, video_id) / f"original.{ref.container}"


def prep_dir(tenant: TenantContext, video_id: str) -> Path:
    """Carpeta de lo derivado de un video (fotogramas, audio, transcripción). La crea si falta."""
    get_video(tenant, video_id)  # falla si el video no es de este cliente
    folder = _video_dir(tenant, video_id) / "prep"
    folder.mkdir(exist_ok=True)
    return folder


# --- Planes de edición y renders ---------------------------------------------
#
#     <storage_dir>/<client_id>/plans/<plan_id>.json     plan de edición guardado
#     <storage_dir>/<client_id>/renders/<render_id>/     un render: estado,
#                                                        subtítulos y video final
#
# Los ids tienen formato fijo, igual que los de video: no se puede armar uno que
# salga de la carpeta del cliente, y el de otro cliente no existe acá.

PLAN_ID_PATTERN = re.compile(r"plan-[0-9a-f]{12}")
RENDER_ID_PATTERN = re.compile(r"render-[0-9a-f]{12}")


class UnknownPlanError(VideoStorageError, LookupError):
    """No hay un plan de edición con ese id para este cliente."""


class UnknownRenderError(VideoStorageError, LookupError):
    """No hay un render con ese id para este cliente."""


def new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:12]}"


def new_render_id() -> str:
    return f"render-{uuid.uuid4().hex[:12]}"


def plan_path(tenant: TenantContext, plan_id: str) -> Path:
    """Ruta del archivo de un plan. La carpeta se crea si falta; el archivo puede no existir."""
    if not PLAN_ID_PATTERN.fullmatch(plan_id):
        raise UnknownPlanError("id de plan inválido")
    folder = get_settings().storage_dir / tenant.client_id / "plans"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{plan_id}.json"


def render_dir(tenant: TenantContext, render_id: str, *, create: bool = False) -> Path:
    """Carpeta de un render. Con `create`, la crea; si no, falla si no existe."""
    if not RENDER_ID_PATTERN.fullmatch(render_id):
        raise UnknownRenderError("id de render inválido")
    folder = get_settings().storage_dir / tenant.client_id / "renders" / render_id
    if create:
        folder.mkdir(parents=True, exist_ok=False)
    elif not folder.is_dir():
        raise UnknownRenderError(f"no hay un render {render_id!r} para este cliente")
    return folder
