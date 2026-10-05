# Yourebrand: sistema agéntico

Sistema de agentes de IA para Yourebrand, agencia de marketing digital que gestiona contenido de múltiples clientes (empresas). Objetivo: potenciar la creatividad y proactividad del equipo y mejorar la supervisión del contenido publicado.

El proyecto es a la vez una herramienta real para Yourebrand y un ejercicio de aprendizaje: el autor cursa el AI Agent Developer del ITBA y quiere aprender a diseñar sistemas de IA de punta a punta.

## Cómo trabajar conmigo

- Actuá como mentor técnico, no solo como ejecutor: explicá el porqué de cada decisión, para poder aplicar el criterio en otros proyectos.
- Antes de proponer algo nuevo, verificá que no contradiga las decisiones de este archivo.
- Priorizá simplicidad: no sugieras infraestructura nueva hasta que el problema que resuelve sea real y concreto.
- Si recomendás algo distinto a lo que enseña el curso, decímelo igual y explicá el trade-off.
- Si una pregunta toca algo que cambia rápido (modelos, APIs, precios), buscá información actual en vez de asumir.
- Respondé en español.

## Componentes

1. **Agente Creativo** (`src/yourebrand/creative/`): lo usan los empleados. Su labor principal es **generar ideas de contenido** para un cliente, de dos tipos: contenido UGC y contenido de diseño (piezas gráficas). Usa como contexto la información de la empresa y, si hay disponibles, sus posts anteriores, para que las ideas sigan la línea de la marca y del feed. Es un pipeline simple: contexto, generación, refinamiento. NO es un framework de agentes autónomos.
   - El agente entrega ideas en texto; no produce la pieza final. La generación de imágenes está postergada (ver "Postergado").
2. **Agente Supervisor** (`src/yourebrand/supervisor/`, fase 2): lo usa el área de supervisión. Corrobora plazos, coherencia de marca y contenido sin subir. LangGraph se reserva para cuando necesite human-in-the-loop.
3. **Extractor de tareas** (`src/yourebrand/tasks/`): componente compartido, no pertenece a ninguno de los dos agentes. Detecta el cliente de una nueva reunión de Fathom, la destila con un modelo, extrae tareas y las crea como borrador en Notion.

## Decisiones ya tomadas (no rediscutir)

- **Multi-tenant desde el día uno**: cada cliente está aislado por `client_id`. Es innegociable: no proponer diseños que lo salteen "para simplificar". Toda consulta a Qdrant o Supabase debe filtrar por `client_id`, y ese filtro vive en un solo lugar (`core/`), nunca repetido a mano en cada agente.
- **Stack**: Python 3.12 + FastAPI, Qdrant (vectores), Supabase (datos relacionales y permisos), API de Claude (Sonnet para generación, Haiku para clasificación y tareas de volumen), Notion API (tareas).
- **Las reuniones de Fathom no se indexan crudas**: se destilan primero con un modelo. Las tareas extraídas quedan como "sugeridas" hasta aprobación humana y nunca se confirman solas.
- **Un solo paquete, un solo venv, un solo `pyproject.toml`**. No separar en paquetes por agente hasta que haya un motivo real de despliegue.

## Estructura

```
src/yourebrand/
  core/         infraestructura común (config, tenancy, llm, vectorstore, db)
  creative/     Agente Creativo
  supervisor/   Agente Supervisor (fase 2)
  tasks/        Extractor de tareas (Fathom -> Notion)
  api/          FastAPI: routers que llaman a los componentes
tests/          pruebas
data/           datos de prueba con clientes ficticios (nunca datos reales)
artifacts/      reportes de evaluación y salidas de experimentos
docs/           propuesta, decisiones de arquitectura, notas
```

La lógica de negocio no vive en `api/`: los agentes deben poder usarse y probarse sin levantar FastAPI.

## Convenciones

- La configuración se lee siempre con `get_settings()` (`core/config.py`, pydantic-settings). Nunca `os.environ` ni `.env` directo en otros módulos.
- Las claves van como `SecretStr`. No imprimirlas ni loguearlas.
- Las llamadas a modelos pasan por `core/llm.py` (`complete`). Los SDKs de proveedores solo se importan en `core/providers.py`, detrás del contrato `LLMProvider`; el proveedor se elige con `LLM_PROVIDER`. Los nombres de modelo vienen de variables de entorno (`MODEL_GENERATION`, `MODEL_FAST`), no se hardcodean.
- Datos reales de clientes: nunca en `data/` ni en git. Viven en Qdrant y Supabase.
- Docstrings y comentarios en español, código en inglés.

## Estado actual

El plan detallado por iteraciones (qué implementar, qué no y cuándo está lista cada una) está en `docs/iteraciones.md`. Antes de implementar algo, verificar a qué iteración pertenece.

Orden de construcción previsto:

1. HECHO: `core/config.py`, `core/tenancy.py`, `core/providers.py`, `core/llm.py`, con tests. Falta probar una llamada real a Claude (requiere `.env` con la clave).
2. SIGUIENTE: Agente creativo mínimo SIN Qdrant. Genera ideas (UGC o diseño) para un cliente ficticio; la información de la empresa y los posts anteriores salen de `data/<client_id>/` y van enteros al prompt. Los posts anteriores son opcionales: tiene que funcionar sin ellos.
3. Test de aislamiento: el contenido del cliente A nunca contiene información del cliente B.
4. Embeddings y retrieval con Qdrant, solo cuando haya más material por cliente del que entra en el prompt.
5. Extractor de tareas, y recién al final el supervisor.

El retrieval no va al final por ser "lo difícil": se construye primero la versión simple de punta a punta y luego se mide si el retrieval la mejora.

Los posts anteriores entran como texto (copy y descripción breve de la pieza), solo para dar contexto de lo que se viene subiendo. Pasarlos como imágenes queda atado a la generación de imágenes, que está postergada.

## Postergado: generación de imágenes

Se dejó para más adelante porque la tecnología todavía no está madura para el uso de la agencia. No construir nada de esto hasta que se retome. Lo ya averiguado (octubre 2026), para no repetir la investigación:

- Claude no genera imágenes (solo las lee), así que hará falta otro proveedor y un contrato propio, distinto de `LLMProvider`.
- Flujo previsto: Claude piensa la idea y redacta el prompt; el modelo de imágenes genera la pieza usando imágenes del cliente como referencia, para seguir su línea de diseño y el feed. Requiere un almacenamiento de imágenes por cliente, aislado por `client_id`.
- Requisito para elegir proveedor: que acepte imágenes de referencia. Candidatos: Nano Banana 2 (`gemini-3.1-flash-image`, Google) y Recraft V4 (estilo reutilizable por marca).
- Antes de elegir, probar con material real de un cliente: mismas ideas con y sin referencias, puntuadas a ciegas por alguien de diseño. Precios y modelos cambian rápido: volver a verificar al retomar.

## Referencia: curso ITBA

Punto de partida de patrones y convenciones: el repo `Agent-Developer-ITBA` (carpeta `ai_agent_project/`, que usa `src/ai_agent_course/`, `data/`, `artifacts/`, `tests/`). Ruta local: `c:/Users/santinoco/Desktop/ai_agent_project/`.
Es una preferencia, no una restricción: si hay una opción mejor para este caso, proponerla.

## Comandos

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
pytest
```
