"""Runtime de tools: registro, validación, ejecución con plazo y errores controlados.

Sigue el runtime del curso, con tres cambios:
- El contexto lleva el cliente y el run_id, no roles: los permisos por
  empleado están fuera del TFI.
- El handler recibe el contexto: de ahí sale el client_id, nunca del modelo.
- Cada ToolCall tiene id, para emparejar cada resultado con su pedido.
"""

import logging
from collections.abc import Callable
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import perf_counter, sleep
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationError

from yourebrand.core.reliability import DeadlineExceeded, run_with_deadline
from yourebrand.core.tenancy import TenantContext
from yourebrand.core.trace import emit, nested

# Un logger por módulo: su nombre es "yourebrand.core.tools", así en los tests
# o en la configuración de logs se puede filtrar solo lo de este archivo.
logger = logging.getLogger(__name__)

# Literal: el tipo solo admite estos textos exactos. El editor avisa si escribís
# "reed" en vez de "read", pero en ejecución Python no lo controla.
ToolKind = Literal["read", "compute", "local_write", "draft", "external_effect", "write"]
# Por eso se arma un set con los mismos valores para chequear en ejecución.
# get_args(ToolKind) devuelve la tupla ("read", "compute", ...): así la lista
# se escribe una sola vez. frozenset es un set que no se puede modificar.
VALID_TOOL_KINDS: frozenset[str] = frozenset(get_args(ToolKind))


# frozen=True: después de crearlo no se puede cambiar ningún campo
# (context.run_id = "otro" da error). Nadie puede cambiar de cliente a mitad
# de una ejecución. Es un dataclass y no un BaseModel porque no hace falta
# validarlo: lo arma el código, y TenantContext ya se valida al crearse.
@dataclass(frozen=True)
class ExecutionContext:
    """Contexto confiable que arma la aplicación al abrir la sesión, nunca el modelo."""

    tenant: TenantContext
    run_id: str
    # Lo pone el grafo, nunca el modelo: True solo cuando una persona aprobó
    # este pedido en la pausa. Una tool con requires_approval no corre sin esto.
    approved: bool = False


class ToolCall(BaseModel):
    """Pedido del modelo: qué tool usar y con qué argumentos."""

    id: str
    tool_name: str
    # default_factory=dict crea un dict nuevo por cada ToolCall. Con
    # `arguments: dict = {}` todos compartirían el mismo dict.
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Salida estable de una tool: vuelve al modelo como resultado."""

    ok: bool
    content: str  # lo que lee el modelo; los datos estructurados van como JSON en texto
    metadata: dict[str, Any] = Field(default_factory=dict)  # lo que lee la aplicación


# Convención para tools que llaman a un modelo por dentro (generar, refinar,
# comparar): informan los tokens de esas llamadas en la metadata con estas
# claves, y quien registra la traza los suma al turno. Sin esto, el costo de
# esas llamadas no aparece en la traza, porque no pasan por el nodo del modelo.
MODEL_INPUT_TOKENS_KEY = "model_input_tokens"
MODEL_OUTPUT_TOKENS_KEY = "model_output_tokens"


def model_usage_metadata(input_tokens: int, output_tokens: int) -> dict[str, int]:
    """Metadata de uso de modelo de una tool, con las claves de la convención."""
    return {MODEL_INPUT_TOKENS_KEY: input_tokens, MODEL_OUTPUT_TOKENS_KEY: output_tokens}


class RetryableToolError(RuntimeError):
    """Falla transitoria de una tool: es la única que habilita reintentar."""


# Forma que tiene que tener un handler: recibe (args, contexto) y devuelve un
# ToolResult. Es solo un alias para que las firmas se lean mejor.
ToolHandler = Callable[[dict[str, Any], ExecutionContext], ToolResult]


def _validation_summary(details: list[dict[str, Any]]) -> list[str]:
    """Qué campo falló y por qué, sin el valor recibido (puede traer texto del cliente)."""
    return [f"{'.'.join(str(part) for part in item.get('loc', ())) or '(argumentos)'}: {item.get('msg')}"
            for item in details]


class Tool:
    """Una capacidad del agente: lo que ve el modelo y los controles del runtime."""

    def __init__(
        self,
        name: str,
        description: str,
        kind: ToolKind,
        fn: ToolHandler,
        args_model: type[BaseModel] | None = None,
        timeout_s: float = 3.0,
        retries: int = 0,
        *,  # lo que sigue solo se puede pasar por nombre: backoff_s=0.1
        backoff_s: float = 0.05,
        idempotent: bool = True,
        max_calls_per_run: int | None = None,
        requires_approval: bool = False,
    ) -> None:
        # Se valida al crear la tool, cuando arranca el programa, y no al
        # ejecutarla: una tool mal configurada falla enseguida y no en medio
        # de una conversación.
        if kind not in VALID_TOOL_KINDS:
            raise ValueError(f"{name}: kind desconocido: {kind}")
        if timeout_s <= 0:
            raise ValueError(f"{name}: timeout_s debe ser mayor que 0")
        if retries < 0:
            raise ValueError(f"{name}: retries no puede ser negativo")
        if backoff_s < 0:
            raise ValueError(f"{name}: backoff_s no puede ser negativo")
        # Reintentar algo que escribe (por ejemplo, en Notion) podría duplicarlo.
        if retries > 0 and not idempotent:
            raise ValueError(f"{name}: una tool no idempotente no puede reintentarse sola")

        self.name = name
        self.description = description
        self.kind = kind
        self.fn = fn
        self.args_model = args_model
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.idempotent = idempotent
        # Tope de ejecuciones en un mismo run (un mensaje de la persona). Para
        # tools caras: si el modelo entra en un bucle, el costo queda acotado.
        if max_calls_per_run is not None and max_calls_per_run < 1:
            raise ValueError(f"{name}: max_calls_per_run debe ser al menos 1")
        self.max_calls_per_run = max_calls_per_run
        # Human-in-the-loop: el grafo pausa antes de ejecutarla y espera la
        # decisión de una persona. El runtime la bloquea si llega sin aprobación.
        self.requires_approval = requires_approval

    def function_declaration(self) -> dict[str, Any]:
        """Lo que ve el modelo: nombre, descripción y schema de argumentos.

        El handler y los controles no se exponen: son de la aplicación.
        """
        # `parameters` es un JSON Schema: describe qué argumentos acepta la
        # tool, de qué tipo es cada uno y cuáles son obligatorios. El modelo lo
        # lee para saber cómo armar su pedido.
        #
        # model_json_schema() lo genera a partir del modelo Pydantic. Para
        # EchoArgs(text: str) devuelve un dict así:
        #   {"title": "EchoArgs", "type": "object",
        #    "properties": {"text": {"title": "Text", "type": "string", ...}},
        #    "required": ["text"]}
        #
        # Sin args_model, la tool no recibe argumentos: un objeto sin propiedades.
        parameters = (
            self.args_model.model_json_schema()
            if self.args_model
            else {"type": "object", "properties": {}}
        )
        # Pydantic agrega "title" al schema y a cada propiedad. Son nombres
        # repetidos ("EchoArgs", "Text") que no le dicen nada nuevo al modelo y
        # suman tokens, que se pagan en cada llamada.
        # pop(clave, None) borra la clave si existe; el None evita el error
        # cuando no está.
        parameters.pop("title", None)
        # .get("properties", {}) devuelve {} si no hay propiedades, así el for
        # no falla. .values() recorre el schema de cada argumento, y pop lo
        # modifica en el lugar, adentro de `parameters`.
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": parameters}


class ToolRegistry:
    """Registro de tools: disponibilidad, validación, ejecución con plazo y reintentos."""

    # Sin aclarar nada, el modelo solo ve tools que no tienen efectos afuera.
    # Mostrar las demás (por ejemplo external_effect) es una decisión explícita.
    DEFAULT_ALLOWED_KINDS: frozenset[str] = frozenset({"read", "compute", "local_write"})

    def __init__(self) -> None:
        # Indexadas por nombre: el modelo pide "generate_ideas" y se encuentra
        # la tool en un paso, sin recorrer una lista.
        self.tools: dict[str, Tool] = {}
        # Ejecuciones por (run_id, tool), para max_calls_per_run. Acotado: solo
        # importan los runs recientes. El lock cubre tools que corren en paralelo.
        self._calls: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._calls_lock = Lock()

    def requires_approval(self, tool_name: str) -> bool:
        """Si el pedido a esta tool tiene que pausar para que una persona lo apruebe."""
        tool = self.tools.get(tool_name)
        return tool is not None and tool.requires_approval

    def register(self, tool: Tool) -> None:
        # Dos tools con el mismo nombre: la segunda pisaría a la primera sin avisar.
        if tool.name in self.tools:
            raise ValueError(f"tool duplicada: {tool.name}")
        self.tools[tool.name] = tool

    def available_tools(self, allowed_kinds: set[str] | None = None) -> list[Tool]:
        """Tools que se exponen al modelo. Un set vacío significa ninguna."""
        # Se compara con `is None` y no con `or`: un set vacío es "falso" en
        # Python, y `allowed_kinds or DEFAULT` convertiría "ninguna" en "las de
        # siempre".
        allowed = self.DEFAULT_ALLOWED_KINDS if allowed_kinds is None else allowed_kinds
        return [tool for tool in self.tools.values() if tool.kind in allowed]

    def function_declarations(self, allowed_kinds: set[str] | None = None) -> list[dict[str, Any]]:
        # Una declaración por tool visible: es la lista que se le manda al modelo.
        return [tool.function_declaration() for tool in self.available_tools(allowed_kinds)]

    _MAX_TRACKED_CALLS = 1000

    def _take_call(self, run_id: str, tool: "Tool") -> bool:
        """Suma una ejecución de `tool` en `run_id`. False si ya llegó al tope."""
        key = (run_id, tool.name)
        with self._calls_lock:
            used = self._calls.get(key, 0)
            if used >= (tool.max_calls_per_run or 0):
                return False
            self._calls[key] = used + 1
            self._calls.move_to_end(key)
            while len(self._calls) > self._MAX_TRACKED_CALLS:
                self._calls.popitem(last=False)
            return True

    def run(
        self,
        call: ToolCall,
        context: ExecutionContext,
        allowed_kinds: set[str] | None = None,
    ) -> ToolResult:
        """Ejecuta un pedido del modelo. Los errores esperables vuelven como ToolResult(ok=False)."""
        # perf_counter() es un reloj para medir duraciones: la resta de dos
        # lecturas da los segundos que pasaron.
        started = perf_counter()
        # Todo el trabajo está en _run, que puede terminar por siete caminos
        # distintos. Envolviéndolo, el log se escribe una sola vez acá, salga
        # por donde salga.
        result = self._run(call, context, allowed_kinds)
        # Trazabilidad: qué tool, para qué cliente y run, y cómo terminó.
        # Al log no van argumentos ni contenido: pueden traer datos del cliente.
        # El mensaje de error va solo a la traza de la pantalla (private).
        fields: dict[str, Any] = {
            "run_id": context.run_id,
            "client_id": context.tenant.client_id,
            "tool": call.tool_name,
            "ok": result.ok,
            # .get devuelve None si la tool salió bien y no hay error_type.
            "error_type": result.metadata.get("error_type"),
            "latency_ms": (perf_counter() - started) * 1000,  # segundos a milisegundos
        }
        if "details" in result.metadata:
            fields["invalid_fields"] = _validation_summary(result.metadata["details"])
        emit(logger, "tool_call", private={} if result.ok else {"error_message": result.content}, **fields)
        return result

    def _run(
        self,
        call: ToolCall,
        context: ExecutionContext,
        allowed_kinds: set[str] | None,
    ) -> ToolResult:
        # Orden de controles: registro → disponibilidad → validación → ejecución.
        # Cada control corta apenas falla y devuelve un error que el modelo puede
        # leer: así nunca se ejecuta algo que no pasó los controles anteriores.

        # 1. Registro: ¿existe? .get devuelve None en vez de lanzar KeyError.
        tool = self.tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                ok=False,
                content=f"Tool desconocida: {call.tool_name}",
                metadata={"error_type": "unknown_tool"},
            )

        # 2. Disponibilidad: el modelo podría pedir por nombre una tool que no
        # se le mostró (por ejemplo, si la inventa). Que no esté en la lista
        # visible no alcanza: hay que bloquearla acá también.
        allowed = self.DEFAULT_ALLOWED_KINDS if allowed_kinds is None else allowed_kinds
        if tool.kind not in allowed:
            return ToolResult(
                ok=False,
                content=f"Tool no disponible: {tool.name}",
                metadata={"error_type": "forbidden_tool", "kind": tool.kind},
            )

        # 2b. Aprobación: si la tool la exige y el contexto no la trae, no corre.
        # El grafo ya pausa antes; esto cubre cualquier otro camino de ejecución.
        if tool.requires_approval and not context.approved:
            return ToolResult(
                ok=False,
                content=f"{tool.name} necesita la aprobación de una persona antes de ejecutarse.",
                metadata={"error_type": "approval_required"},
            )

        # 3. Validación: lo que manda el modelo es un dict sin garantías.
        # model_validate lo controla contra el modelo Pydantic (tipos,
        # obligatorios, largos) y model_dump lo vuelve a convertir en dict, ya
        # limpio y con los valores por defecto completos. El handler recibe
        # siempre datos válidos.
        try:
            args = (
                tool.args_model.model_validate(call.arguments).model_dump()
                if tool.args_model
                else dict(call.arguments)  # copia: el handler no toca el original
            )
        except ValidationError as exc:
            # exc.errors() es una lista con qué campo falló y por qué. Va en la
            # metadata para que el modelo pueda corregir su pedido.
            # include_url=False saca el link a la documentación de Pydantic.
            return ToolResult(
                ok=False,
                content="Argumentos inválidos",
                metadata={"error_type": "validation_error", "details": exc.errors(include_url=False)},
            )

        # 3b. Tope por run: se cuenta antes de ejecutar, así un pedido que
        # falla también consume su lugar (puede haber gastado igual).
        if tool.max_calls_per_run is not None and not self._take_call(context.run_id, tool):
            return ToolResult(
                ok=False,
                content=f"Ya se usó {tool.name} {tool.max_calls_per_run} veces en este mensaje: no se vuelve "
                "a ejecutar. Contale a la persona qué pasó y esperá su próximo mensaje.",
                metadata={"error_type": "call_limit", "max_calls_per_run": tool.max_calls_per_run},
            )

        # 4. Ejecución. Solo RetryableToolError habilita otro intento.
        last_retryable: RetryableToolError | None = None
        # range(1, retries + 2) numera los intentos desde 1: con retries=2 da
        # 1, 2 y 3, o sea el primer intento más dos reintentos.
        for attempt in range(1, tool.retries + 2):
            if attempt > 1:
                # Backoff exponencial: antes del intento 2 espera backoff_s, antes
                # del 3 el doble, antes del 4 el cuádruple. Le da tiempo a una
                # falla temporal para que se resuelva.
                sleep(tool.backoff_s * 2 ** (attempt - 2))
            started = perf_counter()
            try:
                # run_with_deadline recibe una función sin argumentos. La lambda
                # la arma: cuando se llame, ejecuta tool.fn(args, context).
                # nested(): lo que la tool emita por dentro (por ejemplo, su
                # llamada al modelo) queda marcado como parte de este paso.
                with nested():
                    raw = run_with_deadline(lambda: tool.fn(args, context), tool.timeout_s)
            except DeadlineExceeded:
                # Un timeout no se reintenta: el hilo anterior puede seguir
                # corriendo, y otro intento lo duplicaría.
                return ToolResult(
                    ok=False,
                    content=f"Timeout ejecutando {tool.name}",
                    metadata={"error_type": "timeout", "timeout_s": tool.timeout_s, "attempt": attempt},
                )
            except RetryableToolError as exc:
                # Se guarda el error y `continue` salta al próximo intento. Si
                # era el último, el for termina y se sigue después del bucle.
                last_retryable = exc
                continue
            except Exception as exc:
                # Va después de los anteriores porque Python usa el primer except
                # que coincide, y Exception coincide con casi todo. Un error no
                # marcado como transitorio no se repite: puede ser determinístico
                # o un bug del handler.
                return ToolResult(
                    ok=False,
                    content=f"Error controlado: {exc}",
                    metadata={"error_type": type(exc).__name__, "attempt": attempt, "retryable": False},
                )

            # La salida también se valida. Si el handler devolvió algo que no es
            # un ToolResult (un texto, un dict con otra forma), se corta acá y
            # no llega al modelo.
            try:
                result = ToolResult.model_validate(raw)
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content=f"Salida inválida de {tool.name}",
                    metadata={
                        "error_type": "output_validation_error",
                        "details": exc.errors(include_url=False),
                    },
                )
            # {**a, "x": 1} arma un dict nuevo con todo lo de `a` más las claves
            # nuevas. Se conserva lo que puso el handler y se suman los datos
            # del runtime.
            result.metadata = {
                **result.metadata,
                "tool": tool.name,
                "kind": tool.kind,
                "attempt": attempt,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }
            return result

        # Se llega acá solo si todos los intentos terminaron en RetryableToolError:
        # cualquier otro camino ya hizo return adentro del bucle.
        return ToolResult(
            ok=False,
            content=f"Se agotaron los reintentos: {last_retryable}",
            metadata={"error_type": "retry_exhausted", "attempts": tool.retries + 1},
        )
