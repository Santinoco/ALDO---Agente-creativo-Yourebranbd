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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from yourebrand.core.reliability import InvalidUserInputError
from yourebrand.core.tenancy import InvalidClientIdError, UnknownClientError, list_client_ids
from yourebrand.core.trace import TraceEvent
from yourebrand.creative.graph import (
    AgentSession,
    SessionClientMismatchError,
    StepTrace,
    build_graph,
    open_session,
    run_turn,
)
from yourebrand.creative.schemas import GeneratedIdea
from yourebrand.creative.tools import build_creative_registry

STATIC_DIR = Path(__file__).parent / "static"


class SessionRequest(BaseModel):
    client_id: str


class SessionResponse(BaseModel):
    thread_id: str
    client_id: str


class MessageRequest(BaseModel):
    text: str


class TurnResponse(BaseModel):
    reply: str
    run_id: str
    tools_called: list[str]
    input_tokens: int  # del turno completo: agente más tools que llaman a un modelo
    output_tokens: int
    trace: list[StepTrace]
    events: list[TraceEvent]  # el detalle de todo lo que pasó en el turno, en orden
    ideas: list[GeneratedIdea]


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

    # `def` y no `async def`: run_turn bloquea mientras el modelo responde, y
    # FastAPI corre las funciones comunes en un hilo aparte sin frenar al resto.
    @app.post("/sessions/{thread_id}/messages")
    def send_message(thread_id: str, request: MessageRequest) -> TurnResponse:
        session = app.state.sessions.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail="sesión desconocida")
        try:
            outcome = run_turn(app.state.graph, session, request.text)
        except InvalidUserInputError as exc:
            # El mensaje describe la forma del texto, nunca lo cita.
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except SessionClientMismatchError:
            raise HTTPException(status_code=409, detail="la sesión pertenece a otro cliente") from None
        return TurnResponse(
            reply=outcome.reply,
            run_id=outcome.run_id,
            tools_called=outcome.tools_called,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            trace=outcome.trace,
            events=outcome.events,
            ideas=outcome.ideas,
        )

    return app


app = create_app()
