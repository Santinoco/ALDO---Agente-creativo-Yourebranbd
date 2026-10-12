# Yourebrand: sistema agéntico

Sistema de agentes de IA para Yourebrand, agencia de marketing digital que gestiona contenido de múltiples clientes (empresas). Objetivo: potenciar la creatividad y proactividad del equipo y mejorar la supervisión del contenido publicado.

El proyecto es a la vez una herramienta real para Yourebrand y el Trabajo Final Integrador (TFI) de la certificación AI Agent Developer del ITBA. El autor quiere aprender a diseñar sistemas de IA de punta a punta.

**Prioridad actual: entregar el TFI el domingo 8 de noviembre de 2026, 23:59 h.** Hasta esa fecha, todo lo que no sume a la entrega queda fuera de alcance (ver "Fuera del TFI").

## Cómo trabajar conmigo

- Actuá como mentor técnico, no solo como ejecutor: explicá el porqué de cada decisión, para poder aplicar el criterio en otros proyectos.
- Antes de proponer algo nuevo, verificá que no contradiga las decisiones de este archivo ni saque al proyecto del alcance del TFI.
- Priorizá simplicidad: no sugieras infraestructura nueva hasta que el problema que resuelve sea real y concreto. La consigna lo pide igual: "una solución acotada, estable y bien evaluada vale más que una arquitectura ambiciosa pero incompleta".
- Si recomendás algo distinto a lo que enseña el curso, decímelo igual y explicá el trade-off.
- Si una pregunta toca algo que cambia rápido (modelos, APIs, precios), buscá información actual en vez de asumir.
- Respondé en español.

## TFI: qué exige la consigna

Arquitectura elegida: **agente único robusto** (opción A de la consigna). No multiagente: no hay un problema que lo justifique y la consigna no valora sumar agentes por complejidad.

Los ocho componentes obligatorios y cómo los cubre este proyecto:

| # | Requisito | Cómo se cubre |
|---|---|---|
| 1 | Capacidad agéntica: 2+ tools, selección dinámica, validación y manejo de errores. No alcanza una secuencia fija | El Agente Creativo elige qué tool usar según lo que pide el empleado (generar, refinar, consultar la marca, guardar) |
| 2 | Skill reutilizable del dominio | Una skill por tipo de contenido: ideación UGC e ideación de diseño |
| 3 | Estado y memoria con persistencia básica | Estado de la sesión en LangGraph, persistido con checkpointer SQLite |
| 4 | Human-in-the-loop: una acción se pausa y el usuario aprueba, modifica o rechaza | Pausa antes de escribir en Notion |
| 5 | Integración externa con valor real | Notion: las ideas aprobadas se guardan en la base del equipo |
| 6 | Interfaz usable sin tocar código | FastAPI más una pantalla mínima |
| 7 | Ejecución reproducible | `README.md`, `.env.example`, `pyproject.toml`, Docker Compose |
| 8 | Evidencia de calidad | Tests de tools y del recorrido principal, set de evaluación, resultados en `artifacts/`, trazas con `run_id` |

Rúbrica: funcionamiento end-to-end 30%, diseño agéntico 25%, evaluación y trazabilidad 20%, arquitectura y decisiones técnicas 15%, presentación y documentación 10%.

RAG es opcional en la consigna y este proyecto no lo usa: el material de un cliente entra entero en el contexto. Hay que justificarlo en el README.

El `README.md` es el documento principal de la entrega y tiene 13 secciones obligatorias: nombre, integrantes, problema, usuario, descripción, arquitectura (con diagrama), decisiones técnicas, instalación, configuración, ejecución, uso, evaluación y limitaciones. El diagrama tiene que coincidir con lo implementado.

Recorrido de la demo: entrada del usuario, decisión del agente, uso de tools, resultado, intervención humana, respuesta final. Se recomienda un video de respaldo de 3 a 5 minutos y un tag `v1.0-tfi` en el repositorio.

## Componentes

1. **Agente Creativo** (`src/yourebrand/creative/`): es el TFI. Lo usan los empleados de la agencia en una conversación: piden ideas de contenido para un cliente (UGC o diseño), las refinan con comentarios y guardan en Notion las que eligen. El modelo decide qué tool usar en cada turno; no hay una secuencia fija.
   - Usa como contexto la información de la empresa y, si hay, sus posts anteriores, para que las ideas sigan la línea de la marca y del feed.
   - Entrega ideas en texto; no produce la pieza final. La generación de imágenes está postergada.
   - Tools previstas (el diseño fino se cierra al implementarlas): `get_brand_info`, `get_previous_posts`, `generate_ideas`, `refine_idea`, `save_ideas_to_notion`.
   - Skills: `ugc-ideation` y `design-ideation`. Contienen el criterio de cada tipo de contenido, que hoy está en `creative/prompts.py`.
   - Human-in-the-loop: `save_ideas_to_notion` nunca escribe sin aprobación. El agente propone qué guardar, se pausa, y la persona aprueba, modifica o rechaza.
2. **Agente Supervisor** (`src/yourebrand/supervisor/`): fuera del TFI.
3. **Extractor de tareas** (`src/yourebrand/tasks/`): fuera del TFI.

## Decisiones ya tomadas (no rediscutir)

- **Multi-tenant desde el día uno**: cada cliente está aislado por `client_id`. Es innegociable: no proponer diseños que lo salteen "para simplificar". El filtro por cliente vive en un solo lugar (`core/`), nunca repetido a mano.
- **El `client_id` no lo elige el modelo**: el cliente se fija al abrir la sesión y vive en el estado. Las tools lo leen de ahí y no lo reciben como argumento, así una respuesta del modelo no puede cruzar de un cliente a otro.
- **El Agente Creativo es un agente con tools, no un pipeline fijo.** Reemplaza la decisión anterior ("pipeline simple, no un framework de agentes"), que era razonable como producto pero no cumple la consigna del TFI.
- **LangGraph para el grafo, la pausa y la persistencia** (`interrupt` y checkpointer). Se adelanta desde el Supervisor porque el motivo que lo justificaba, human-in-the-loop, ahora es un requisito.
- **LangGraph sin los modelos de chat de LangChain**: los nodos son funciones propias que llaman a `core/llm.py`. Se mantiene el contrato `LLMProvider` y el registro de tokens, latencia y `run_id`. El costo es escribir a mano el nodo que ejecuta las tools en vez de usar los prearmados (`create_react_agent`, `ToolNode`).
- **Persistencia en SQLite** para el TFI: sin servicios que levantar, así otra persona puede reproducir el proyecto. Supabase queda para después.
- **Stack**: Python 3.12, FastAPI, LangGraph con checkpointer SQLite, API de Claude (Sonnet para generación, Haiku para tareas de volumen), Notion API.
- **Nada se escribe en un sistema externo sin aprobación humana.**
- **Un solo paquete, un solo venv, un solo `pyproject.toml`.**

## Estructura

```
src/yourebrand/
  core/         infraestructura común (config, tenancy, llm, providers, errors, reliability)
  creative/     Agente Creativo: grafo, tools, skills, prompts, schemas
  api/          FastAPI: routers que llaman al agente
  supervisor/   fuera del TFI
  tasks/        fuera del TFI
tests/          pruebas
data/           clientes ficticios (nunca datos reales)
evals/          set de evaluación: pedidos fijos por cliente
artifacts/      resultados de evaluación y salidas de experimentos
docs/           arquitectura, decisiones técnicas, plan
```

La lógica de negocio no vive en `api/`: el agente tiene que poder usarse y probarse sin levantar FastAPI.

## Convenciones

- La configuración se lee siempre con `get_settings()` (`core/config.py`, pydantic-settings). Nunca `os.environ` ni `.env` directo en otros módulos.
- Las claves van como `SecretStr`. No imprimirlas ni loguearlas. Nunca credenciales reales en git: la consigna lo penaliza.
- Las llamadas a modelos pasan por `core/llm.py`. Los SDKs de proveedores solo se importan en `core/providers.py`, detrás del contrato `LLMProvider`; el proveedor se elige con `LLM_PROVIDER`. Los nombres de modelo vienen de variables de entorno (`MODEL_GENERATION`, `MODEL_FAST`), no se hardcodean.
- No se da por hecho nada que un SDK haga por su cuenta. El contrato `LLMProvider` dice qué garantiza cada proveedor y `core/` se hace cargo del resto:
  - Los proveedores lanzan solo errores de `core/errors.py`; ningún otro módulo captura excepciones de un SDK.
  - Los reintentos viven en `core/reliability.py` (`call_with_retry`) y los aplica `core/llm.py`. Los reintentos propios de cada SDK se apagan.
  - El timeout es parte del contrato (`timeout_seconds`). Se configura con `LLM_TIMEOUT_SECONDS` y `LLM_MAX_RETRIES`.
- Todo texto libre que escribe un usuario pasa por `validate_user_text` (`core/reliability.py`) antes de llegar a un modelo. Valida la forma, no frena una inyección de prompt: eso lo resuelven el `client_id` fuera del alcance del modelo y la aprobación humana.
- Cada tool valida sus entradas y salidas con Pydantic. Un error de una tool vuelve al modelo como resultado de la tool, para que pueda corregir o avisar; no corta la conversación.
- Toda ejecución lleva `run_id` y registra qué tools se llamaron, tokens y latencia. Es la evidencia de trazabilidad que pide la consigna.
- Datos reales de clientes: nunca en `data/` ni en git.
- Docstrings y comentarios en español, código en inglés.

## Estado actual

Hecho:

- `core/`: `config.py`, `tenancy.py`, `providers.py`, `llm.py` (`complete` y `complete_structured`), `errors.py` y `reliability.py` (validación de texto, reintentos), con tests.
- `creative/`: `context.py`, `prompts.py`, `schemas.py` e `ideas.py` (`generate_ideas` como pipeline fijo), con tests.
- Cliente ficticio `data/tambor/` con `brand.md` y `posts.md`.

Lo ya escrito no se tira: `generate_ideas` pasa a ser el interior de una tool, `context.py` el de las tools de lectura, y las guías de `prompts.py` se convierten en skills.

Plan hasta la entrega, en orden. El detalle de cada iteración (qué implementar, qué no y cuándo está lista) está en `docs/iteraciones.md`. Antes de implementar algo, verificar a qué iteración pertenece; no se empieza una sin cerrar la anterior.

- **Iteración 2, SIGUIENTE: agente con tools (hasta el 18/10).** Soporte de tools en `core/`, grafo de LangGraph, tools de lectura, `generate_ideas` y `refine_idea`, las dos skills, chat desde la terminal.
- **Iteración 3: estado, HITL y Notion (hasta el 25/10).** Checkpointer SQLite, `save_ideas_to_notion` con pausa y aprobación, segundo cliente ficticio y test de aislamiento.
- **Iteración 4: interfaz y reproducibilidad (hasta el 1/11).** Endpoints de FastAPI, pantalla mínima, Docker Compose, set de evaluación en `evals/`.
- **Iteración 5: evaluación y entrega (hasta el 8/11).** Resultados y métricas en `artifacts/`, `README.md` con las 13 secciones y el diagrama, video de respaldo, tag `v1.0-tfi`.

Los posts anteriores entran como texto (copy y descripción breve de la pieza), solo para dar contexto de lo que se viene subiendo.

## Fuera del TFI

No construir nada de esto antes del 8 de noviembre. Se retoma después de la entrega.

- **Agente Supervisor**: corrobora plazos, coherencia de marca y contenido sin subir, para el área de supervisión.
- **Extractor de tareas**: detecta el cliente de una reunión de Fathom, la destila con un modelo, extrae tareas y las crea como borrador en Notion. Las reuniones no se indexan crudas y las tareas quedan como "sugeridas" hasta aprobación humana.
- **Qdrant y retrieval**: solo cuando el material de un cliente no entre en el contexto. Toda búsqueda deberá filtrar por `client_id` desde `core/`.
- **Supabase y datos reales de clientes**: datos relacionales y permisos por empleado.
- **Evaluación automática con un modelo como juez**, salvo que sobre tiempo en la iteración 5.

`core/db.py` y `core/vectorstore.py` existen vacíos para estos componentes; no se tocan durante el TFI.

## Postergado: generación de imágenes

Se dejó para más adelante porque la tecnología todavía no está madura para el uso de la agencia. No construir nada de esto hasta que se retome. Lo ya averiguado (octubre 2026), para no repetir la investigación:

- Claude no genera imágenes (solo las lee), así que hará falta otro proveedor y un contrato propio, distinto de `LLMProvider`.
- Flujo previsto: Claude piensa la idea y redacta el prompt; el modelo de imágenes genera la pieza usando imágenes del cliente como referencia, para seguir su línea de diseño y el feed. Requiere un almacenamiento de imágenes por cliente, aislado por `client_id`.
- Requisito para elegir proveedor: que acepte imágenes de referencia. Candidatos: Nano Banana 2 (`gemini-3.1-flash-image`, Google) y Recraft V4 (estilo reutilizable por marca).
- Antes de elegir, probar con material real de un cliente: mismas ideas con y sin referencias, puntuadas a ciegas por alguien de diseño. Precios y modelos cambian rápido: volver a verificar al retomar.

## Referencia: curso ITBA

Punto de partida de patrones y convenciones: el repo `Agent-Developer-ITBA` (carpeta `ai_agent_project/`, que usa `src/ai_agent_course/`, `data/`, `artifacts/`, `tests/`). Ruta local: `c:/Users/santinoco/Desktop/ai_agent_project/`.
Es una preferencia, no una restricción: si hay una opción mejor para este caso, proponerla.

La consigna del TFI está en `TFI_AI_Agent_Developer_ITBA_Instrucciones_Entrega.pdf` (carpeta Descargas del autor, fuera del repositorio).

## Comandos

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
pytest
```
