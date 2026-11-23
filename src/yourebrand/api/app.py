"""API del Agente Creativo y la pantalla de demo.

Solo transporte: traduce HTTP a llamadas a `creative/graph.py` y errores a
códigos HTTP. La lógica vive en el agente, que se usa y se prueba sin levantar
FastAPI.

Para levantarla:
    uvicorn yourebrand.api.app:app --reload
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from yourebrand.core.reliability import InvalidUserInputError
from yourebrand.core.storage import (
    InvalidVideoError,
    UnknownRenderError,
    UnknownVideoError,
    VideoRef,
    VideoTooLargeError,
    save_video,
)
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.tenancy import InvalidClientIdError, UnknownClientError, list_client_ids
from yourebrand.core.trace import TraceEvent
from yourebrand.creative.graph import (
    AgentSession,
    NoPendingApprovalError,
    PendingApprovalError,
    SessionClientMismatchError,
    StepTrace,
    TurnOutcome,
    build_graph,
    open_session,
    resume_turn,
    run_turn,
)
from yourebrand.creative.schemas import GeneratedIdea
from yourebrand.creative.tools import build_creative_registry
from yourebrand.creative.video_render import RenderNotReadyError, RenderStatus, get_render, render_output

STATIC_DIR = Path(__file__).parent / "static"


class SessionRequest(BaseModel):
    client_id: str


class SessionResponse(BaseModel):
    thread_id: str
    client_id: str


class MessageRequest(BaseModel):
    text: str
    # Videos ya subidos con POST /sessions/{thread_id}/videos que se adjuntan a este mensaje.
    video_ids: list[str] = Field(default_factory=list, max_length=10)


class TurnResponse(BaseModel):
    reply: str
    run_id: str
    tools_called: list[str]
    input_tokens: int  # del turno completo: agente más tools que llaman a un modelo
    output_tokens: int
    trace: list[StepTrace]
    events: list[TraceEvent]  # el detalle de todo lo que pasó en el turno, en orden
    ideas: list[GeneratedIdea]
    # Si el turno quedó en pausa: lo que hay que aprobar o rechazar con POST .../approval.
    pending_approval: list[dict[str, Any]] | None = None
    # Renders que arrancaron en el turno: se siguen con GET .../renders/{render_id}.
    renders: list[RenderStatus] = Field(default_factory=list)

    @classmethod
    def from_outcome(cls, outcome: TurnOutcome) -> "TurnResponse":
        return cls(
            reply=outcome.reply,
            run_id=outcome.run_id,
            tools_called=outcome.tools_called,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            trace=outcome.trace,
            events=outcome.events,
            ideas=outcome.ideas,
            pending_approval=outcome.pending_approval,
            renders=[RenderStatus.model_validate(render) for render in outcome.renders],
        )


class ApprovalRequest(BaseModel):
    approved: bool
    # Al rechazar, qué cambiar: el agente vuelve a planear con esto.
    comment: str | None = None


class SessionStore:
    """Sesiones abiertas: qué cliente corresponde a cada thread_id.

    El cliente lo fija el servidor al abrir la sesión: el navegador solo manda
    el thread_id, así una petición no puede cambiar de cliente. Vive en memoria,
    como el checkpointer; en la iteración 3 se reemplaza por lo que guarda el
    checkpointer SQLite, para retomar sesiones después de reiniciar.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.Lock()  # FastAPI atiende peticiones en varios hilos

    def add(self, session: AgentSession) -> None:
        with self._lock:
            self._sessions[session.thread_id] = session

    def get(self, thread_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(thread_id)


def create_app(graph=None) -> FastAPI:
    """Arma la app. Sin `graph`, lo construye al arrancar con las tools reales.

    Los tests pasan un grafo propio y no dependen del `.env`.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.graph is None:
            app.state.graph = build_graph(build_creative_registry())
        yield

    app = FastAPI(title="Yourebrand: Agente Creativo", lifespan=lifespan)
    app.state.graph = graph
    app.state.sessions = SessionStore()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/clients")
    def clients() -> list[str]:
        return list_client_ids()

    @app.post("/sessions")
    def create_session(request: SessionRequest) -> SessionResponse:
        try:
            session = open_session(request.client_id)
        except InvalidClientIdError:
            raise HTTPException(status_code=422, detail="client_id inválido") from None
        except UnknownClientError:
            raise HTTPException(status_code=404, detail="cliente desconocido") from None
        app.state.sessions.add(session)
        return SessionResponse(thread_id=session.thread_id, client_id=session.client_id)

    # Subida de un video. El cliente sale de la sesión abierta en el servidor,
    # nunca de la petición: así no se puede guardar un video en otro cliente.
    # `def`: copiar cientos de MB bloquea, y FastAPI lo corre en otro hilo.
    @app.post("/sessions/{thread_id}/videos")
    def upload_video(thread_id: str, file: UploadFile = File(...)) -> VideoRef:
        session = app.state.sessions.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail="sesión desconocida")
        try:
            return save_video(TenantContext(session.client_id), file.file, file.filename)
        except VideoTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        except InvalidVideoError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from None
        finally:
            file.file.close()

    # `def` y no `async def`: run_turn bloquea mientras el modelo responde, y
    # FastAPI corre las funciones comunes en un hilo aparte sin frenar al resto.
    @app.post("/sessions/{thread_id}/messages")
    def send_message(thread_id: str, request: MessageRequest) -> TurnResponse:
        session = app.state.sessions.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail="sesión desconocida")
        try:
            outcome = run_turn(app.state.graph, session, request.text, request.video_ids)
        except InvalidUserInputError as exc:
            # El mensaje describe la forma del texto, nunca lo cita.
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except UnknownVideoError:
            raise HTTPException(status_code=404, detail="video desconocido") from None
        except SessionClientMismatchError:
            raise HTTPException(status_code=409, detail="la sesión pertenece a otro cliente") from None
        except PendingApprovalError:
            raise HTTPException(status_code=409, detail="primero aprobá o rechazá lo pendiente") from None
        return TurnResponse.from_outcome(outcome)

    # Resuelve la pausa del turno: aprobar o rechazar (con un comentario). El
    # turno sigue donde se cortó y la respuesta es la misma que la de un mensaje.
    @app.post("/sessions/{thread_id}/approval")
    def resolve_approval(thread_id: str, request: ApprovalRequest) -> TurnResponse:
        session = _session(thread_id)
        try:
            outcome = resume_turn(app.state.graph, session, request.approved, request.comment)
        except InvalidUserInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except NoPendingApprovalError:
            raise HTTPException(status_code=409, detail="no hay nada pendiente de aprobación") from None
        except SessionClientMismatchError:
            raise HTTPException(status_code=409, detail="la sesión pertenece a otro cliente") from None
        return TurnResponse.from_outcome(outcome)

    # Los renders cuelgan de la sesión: el cliente sale de ella, así un
    # render_id de otro cliente no existe acá.
    @app.get("/sessions/{thread_id}/renders/{render_id}")
    def render_status(thread_id: str, render_id: str) -> RenderStatus:
        session = _session(thread_id)
        try:
            return get_render(TenantContext(session.client_id), render_id)
        except UnknownRenderError:
            raise HTTPException(status_code=404, detail="render desconocido") from None

    @app.get("/sessions/{thread_id}/renders/{render_id}/video")
    def render_video_file(thread_id: str, render_id: str) -> FileResponse:
        session = _session(thread_id)
        try:
            path = render_output(TenantContext(session.client_id), render_id)
        except UnknownRenderError:
            raise HTTPException(status_code=404, detail="render desconocido") from None
        except RenderNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return FileResponse(path, media_type="video/mp4", filename=f"{render_id}.mp4",
                            content_disposition_type="inline")

    def _session(thread_id: str) -> AgentSession:
        session = app.state.sessions.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail="sesión desconocida")
        return session

    return app


app = create_app()
