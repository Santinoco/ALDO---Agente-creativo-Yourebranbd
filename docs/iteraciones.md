# Plan de desarrollo por iteraciones

Plan hasta la entrega del Trabajo Final Integrador (TFI): **domingo 8 de noviembre de 2026, 23:59 h**. El alcance y las decisiones de arquitectura están en `CLAUDE.md`; este archivo detalla qué se construye en cada iteración.

Cada iteración termina con algo que funciona y se puede mostrar. No se empieza una iteración hasta cumplir el "Listo cuando" de la anterior.

Si aparece una idea que no está en la iteración actual, se anota en la iteración donde corresponde y se sigue con lo que toca. Si no entra en ninguna, va a "Después del TFI".

| # | Iteración | Fecha objetivo | Requisitos del TFI que cierra | Estado |
|---|---|---|---|---|
| 0 | Base común (`core/`) | | | Hecha |
| 1 | Creativo mínimo: ideas desde archivos | | | Hecha |
| 2 | Agente con tools | 18/10 | Capacidad agéntica, skill | **Siguiente** |
| 3 | Estado, HITL y Notion | 25/10 | Estado y memoria, human-in-the-loop, integración externa | Pendiente |
| 4 | Interfaz y reproducibilidad | 1/11 | Interfaz accesible, ejecución reproducible | Pendiente |
| 5 | Evaluación y entrega | 8/11 | Evidencia de calidad, README, demo | Pendiente |

---

## Iteración 0: Base común (hecha)

**Objetivo:** tener la infraestructura que usan todos los componentes.

**Implementado:**
- `core/config.py`: configuración con `get_settings()`.
- `core/tenancy.py`: `TenantContext` y `client_data_dir`.
- `core/providers.py`: contrato `LLMProvider` y `AnthropicProvider`.
- `core/llm.py`: `complete`, con registro de tokens, latencia y `run_id`.
- Tests de los cuatro módulos.

---

## Iteración 1: Creativo mínimo (hecha)

**Objetivo:** pedir ideas para un cliente ficticio y recibir una lista de ideas alineadas a su marca.

**Implementado:**
- Cliente ficticio en `data/tambor/`: `brand.md` y `posts.md`.
- `creative/context.py`: lee marca y posts del cliente. Funciona aunque no haya posts.
- `creative/prompts.py`: arma las instrucciones y el pedido.
- `creative/schemas.py`: forma validada de una idea.
- `core/llm.py`: `complete_structured`, salida validada con Pydantic.
- `creative/ideas.py`: `generate_ideas(tenant, content_type, count)` como secuencia fija.
- Tests con el proveedor simulado.

**Quedó pendiente:** confirmar una llamada real a Claude con el `.env` y una forma de ejecutarlo desde la terminal. Las dos cosas se resuelven al empezar la iteración 2.

---

## Iteración 2: Agente con tools

**Objetivo:** que el modelo decida qué hacer en cada turno de una conversación, en vez de seguir una secuencia fija.

Es la iteración más pesada y la que más pesa en la nota: sin ella no hay diseño agéntico (25%) ni recorrido de punta a punta (30%).

**Implementar:**
- Una llamada real a Claude con el `.env`, antes que nada.
- Soporte de tools en `core/providers.py` y `core/llm.py`: mandar las definiciones de las tools, recibir los pedidos de uso del modelo y devolverle los resultados. Mantiene el registro de tokens, latencia y `run_id`.
- Estado del grafo: mensajes de la conversación, `client_id` de la sesión, ideas generadas y `run_id`.
- Grafo de LangGraph con dos nodos propios: el que llama al modelo y el que ejecuta las tools. Sin los prearmados (`create_react_agent`, `ToolNode`).
- Tools, cada una con entradas y salidas validadas con Pydantic:
  - `get_brand_info`: información de la marca.
  - `get_previous_posts`: posts anteriores, si hay.
  - `generate_ideas`: usa por dentro el `generate_ideas` de la iteración 1.
  - `refine_idea`: ajusta una idea ya generada según un comentario.
- Las tools leen el `client_id` del estado; no lo reciben como argumento.
- Manejo de errores: si una tool falla, el error vuelve al modelo como resultado de la tool.
- Skills `ugc-ideation` y `design-ideation`, armadas a partir de las guías de `creative/prompts.py`.
- Chat desde la terminal, que registre qué tool se eligió en cada turno.
- Tests de cada tool, del nodo que las ejecuta (incluido el caso de error) y del grafo con el proveedor simulado.

**No implementar todavía:**
- Persistencia: la conversación vive mientras dura el proceso.
- Notion y cualquier escritura en sistemas externos.
- FastAPI.
- Qdrant, Supabase, un segundo agente.

**Listo cuando:** desde la terminal, en una misma conversación sobre `tambor`, se piden 5 ideas UGC, se refina una con un comentario y se hace una pregunta sobre la marca, y el registro muestra que el modelo eligió tools distintas en cada turno.

---

## Iteración 3: Estado, HITL y Notion

**Objetivo:** que la sesión sobreviva a un reinicio y que el agente guarde ideas en Notion solo con aprobación de una persona.

**Implementar:**
- Checkpointer SQLite: cada sesión se identifica y se puede retomar.
- Cliente de Notion en `core/`, único lugar donde se importa su SDK. Variables nuevas en `core/config.py` y `.env.example`: clave de Notion e id de la base de datos destino.
- Tool `save_ideas_to_notion` con pausa (`interrupt`): el agente propone qué ideas guardar y se detiene. La persona puede:
  - aprobar: se guardan como están;
  - modificar: se guardan con los cambios;
  - rechazar: no se guarda nada y el agente sigue la conversación.
- Segundo cliente ficticio, de un rubro bien distinto a `tambor`.
- Test de aislamiento: lo generado para el cliente A no contiene información del cliente B, y un `client_id` inexistente o inválido falla.
- Test del recorrido principal completo, con Notion simulado: pedido, tools, pausa, aprobación, guardado.
- El chat de terminal pasa a soportar la pausa y a retomar una sesión existente.

**No implementar todavía:**
- FastAPI y la pantalla.
- Memoria compartida entre sesiones o entre clientes.
- Usuarios y permisos por empleado.
- Supabase.

**Listo cuando:** desde la terminal se generan ideas, se pide guardar dos, el agente se pausa, se cierra el proceso, se vuelve a abrir la misma sesión, se aprueba modificando una, y las dos aparecen en Notion. El test de aislamiento pasa.

---

## Iteración 4: Interfaz y reproducibilidad

**Objetivo:** que alguien que no escribió el código pueda instalar el proyecto y usarlo desde el navegador.

**Implementar:**
- `api/` con endpoints de FastAPI, sin lógica de negocio:
  - listar clientes disponibles;
  - abrir una sesión para un cliente;
  - enviar un mensaje;
  - resolver una pausa (aprobar, modificar o rechazar).
- Pantalla mínima: elegir cliente, conversar, ver las ideas y resolver la aprobación.
- `Dockerfile` y `docker-compose.yml`.
- `.env.example` completo y sin secretos.
- Set de evaluación en `evals/`: pedidos fijos por cliente, cada uno con la tool que se espera que el agente elija.
- Script que corre el set y guarda las salidas en `artifacts/`.
- Prueba de instalación desde cero: clonar en otra carpeta y seguir los pasos escritos.

**No implementar todavía:**
- Autenticación y permisos.
- Despliegue en la nube.
- Respuestas en streaming.
- Funciones que no estén en el recorrido de la demo.

**Si la iteración se atrasa:** la pantalla se reemplaza por la documentación interactiva de FastAPI (`/docs`) y Docker por un procedimiento de instalación verificable. La consigna acepta las dos cosas.

**Listo cuando:** con un clon nuevo del repositorio y siguiendo solo las instrucciones escritas, se levanta la aplicación y se completa el recorrido de la demo desde el navegador.

---

## Iteración 5: Evaluación y entrega

**Objetivo:** medir el agente, documentarlo y entregar.

A partir de acá no se agregan funciones: solo se corrige lo que la evaluación muestre.

**Implementar:**
- Correr el set de evaluación y guardar los resultados en `artifacts/` con sus métricas: éxito de la tarea, tool elegida contra tool esperada, cantidad de pasos, latencia, costo y errores.
- Rúbrica simple para puntuar a mano la calidad de las ideas (encaja con la marca, es original, es realizable) y la puntuación obtenida.
- Corrección de los problemas que aparezcan, volviendo a medir después de cada cambio.
- `README.md` con las 13 secciones de la consigna: nombre, integrantes, problema, usuario, descripción, arquitectura, decisiones técnicas, instalación, configuración, ejecución, uso, evaluación y limitaciones.
- Diagrama de arquitectura en `docs/`, que coincida con lo implementado: modelo, tools, skills, estado, HITL e integración con Notion.
- Decisiones técnicas justificadas: agente único, LangGraph con nodos propios, sin RAG, SQLite, dónde va la pausa.
- Video de respaldo de 3 a 5 minutos: el problema, el sistema funcionando, una decisión del agente, una tool, la pausa y el resultado.
- Revisión de que no haya credenciales en el repositorio ni en su historial.
- Tag `v1.0-tfi`.
- Datos para el espacio de entrega: nombre, integrantes, URL del repositorio, URL del video y descripción de hasta 5 líneas.

**No implementar:**
- Funciones nuevas.
- Evaluación automática con un modelo como juez, salvo que todo lo anterior esté cerrado.

**Listo cuando:** el checklist de la sección 9 de la consigna está completo y el tag `v1.0-tfi` está subido.

---

## Después del TFI

Nada de esto se construye antes del 8 de noviembre. El detalle de cada punto está en `CLAUDE.md`, secciones "Fuera del TFI" y "Postergado: generación de imágenes".

- Datos reales de clientes en Supabase, con permisos por empleado.
- Retrieval con Qdrant, solo si el material de un cliente deja de entrar en el contexto.
- Extractor de tareas (Fathom a Notion).
- Agente Supervisor.
- Generación de imágenes.
