# Plan de desarrollo por iteraciones

Plan hasta la entrega del Trabajo Final Integrador (TFI): **miércoles 9 de diciembre de 2026** (la fecha original era el 8 de noviembre; se corrió porque se atrasó el curso). El alcance y las decisiones de arquitectura están en `CLAUDE.md`; este archivo detalla qué se construye en cada iteración.

Cada iteración termina con algo que funciona y se puede mostrar. No se empieza una iteración hasta cumplir el "Listo cuando" de la anterior.

Si aparece una idea que no está en la iteración actual, se anota en la iteración donde corresponde y se sigue con lo que toca. Si no entra en ninguna, va a "Después del TFI".

El set de evaluación (`evals/`) se crea en la iteración 2 y crece en cada iteración, como pidió el profesor en su devolución (resumida en `CLAUDE.md`). Mide cuatro criterios: respeta el estilo de cada marca, evita repetir ideas, usa bien las tools (incluido preguntar cuando el pedido no está claro) y pide aprobación antes de guardar.

| # | Iteración | Fecha objetivo | Requisitos del TFI que cierra | Estado |
|---|---|---|---|---|
| 0 | Base común (`core/`) | | | Hecha |
| 1 | Creativo mínimo: ideas desde archivos | | | Hecha |
| 2 | Agente con tools | 2/12 | Capacidad agéntica, skill, evaluación inicial | Hecha (26/11) |
| 2b | Edición de video de punta a punta | 25/11 | Human-in-the-loop (la pausa genérica), funcionamiento end-to-end | Hecha (25/11) |
| 3 | Estado, HITL y Notion | 2/12 | Estado y memoria, human-in-the-loop, integración externa | **Siguiente** |
| 4 | Interfaz y reproducibilidad | 5/12 | Interfaz accesible, ejecución reproducible | Pendiente |
| 5 | Evaluación y entrega | 9/12 | Evidencia de calidad, README, demo | Pendiente |

La 2b se hizo antes de cerrar la 2, como excepción a la regla de arriba: era el avance que se mostraba al profesor y al cliente. Las dos quedaron cerradas antes de empezar la 3.

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

**Agregado antes de la iteración 2 (capa de confiabilidad):**
- `core/errors.py`: errores propios de la capa de modelos. Cada proveedor traduce los de su SDK.
- `core/reliability.py`: `validate_user_text` para texto libre de usuarios y `call_with_retry` con backoff.
- `core/providers.py`: el contrato exige `timeout_seconds`, un solo intento y errores propios. Reintentos del SDK apagados.
- `core/llm.py`: reintenta fallas temporales y registra cada reintento y la cantidad de intentos.
- `LLM_TIMEOUT_SECONDS` y `LLM_MAX_RETRIES` en la configuración.

**Quedó pendiente:** confirmar una llamada real a Claude con el `.env` y una forma de ejecutarlo desde la terminal. Las dos cosas se resuelven al empezar la iteración 2. Con la llamada real, ajustar `LLM_TIMEOUT_SECONDS` según lo que tarde una lista de 20 ideas.

---

## Iteración 2: Agente con tools

**Objetivo:** que el modelo decida qué hacer en cada turno de una conversación, en vez de seguir una secuencia fija.

Es la iteración más pesada y la que más pesa en la nota: sin ella no hay diseño agéntico (25%) ni recorrido de punta a punta (30%).

**Implementar:**
- Una llamada real a Claude con el `.env`, antes que nada.
- Soporte de tools en `core/providers/` y `core/llm.py`: mandar las definiciones de las tools, recibir los pedidos de uso del modelo y devolverle los resultados. Mantiene el registro de tokens, latencia y `run_id`.
- Estado del grafo: mensajes de la conversación, `client_id` de la sesión, ideas generadas y `run_id`.
- Grafo de LangGraph con dos nodos propios: el que llama al modelo y el que ejecuta las tools. Sin los prearmados (`create_react_agent`, `ToolNode`).
- Tools, cada una con entradas y salidas validadas con Pydantic:
  - `get_brand_info`: información de la marca.
  - `get_previous_posts`: posts anteriores, si hay.
  - `generate_ideas`: usa por dentro el `generate_ideas` de la iteración 1.
  - `refine_idea`: ajusta una idea ya generada según un comentario.
  - `check_similarity`: compara una o varias ideas contra los posts anteriores y devuelve cuáles se parecen demasiado, a qué post y por qué. Usa el modelo rápido (`MODEL_FAST`) con los posts enteros en el contexto, sin embeddings.
- Las tools leen el `client_id` del estado; no lo reciben como argumento.
- Pedido de aclaraciones: si al pedido le falta algo para generar (por ejemplo, el tipo de contenido), el agente pregunta en vez de suponer. No es una tool: es el camino en el que el modelo termina el turno sin llamar a ninguna.
- Manejo de errores: si una tool falla, el error vuelve al modelo como resultado de la tool.
- Tope de pasos por turno, para que el loop de tools no pueda girar indefinidamente.
- El mensaje del usuario y el comentario de `refine_idea` pasan por `validate_user_text`.
- Skills `ugc-ideation` y `design-ideation`, armadas a partir de las guías de `creative/prompts.py`. Incluyen cuándo pedir aclaraciones y cuándo verificar similitud antes de proponer.
- Set de evaluación inicial en `evals/`: pedidos fijos sobre `tambor`, cada uno con lo que se espera del agente (qué tool elige o si pregunta). Incluye pedidos claros, pedidos ambiguos (se espera que pregunte) y pedidos que llevan a ideas ya publicadas (se espera `check_similarity`).
- Script que corre el set contra el modelo real y guarda en `artifacts/` las salidas, la tool elegida contra la esperada, los pasos, los tokens y la latencia de cada caso.
- Pantalla de demo (adelantada de la iteración 4, en lugar del chat de terminal): FastAPI con los endpoints de clientes, sesiones y mensajes, y una pantalla con el chat y, al costado, la traza de cada turno (llamadas al modelo, tools, tokens y tiempos). Se eligió una sola interfaz en vez de dos: la misma pantalla es la de la entrega y la de la demo.
- Tests de cada tool, del nodo que las ejecuta (incluido el caso de error) y del grafo con el proveedor simulado.

**No implementar todavía:**
- Persistencia: la conversación vive mientras dura el proceso.
- Notion y cualquier escritura en sistemas externos.
- Qdrant, Supabase, un segundo agente.

**Listo cuando:** desde la pantalla, en una misma conversación sobre `tambor`, se piden 5 ideas UGC, se refina una con un comentario y se hace una pregunta sobre la marca, y la traza muestra que el modelo eligió tools distintas en cada turno. Además, un pedido ambiguo hace que el agente pregunte sin generar, una idea parecida a un post de `tambor` queda marcada, y el set de evaluación corre y deja sus resultados en `artifacts/`.

**Resultado (26/11):**
- `refine_idea` devuelve la idea ajustada con un id nuevo y `refined_from`, sin pisar la original. `check_similarity` compara contra los posts con el modelo rápido y verifica que la respuesta cite posts que existen.
- Se sumó al prompt del agente una sección de alcance: se niega a pedidos fuera del contenido de la marca (código, tareas personales) y sigue ayudando con lo que sí entra, como un copy.
- Set de evaluación en `evals/cases.json` (14 casos sobre `tambor` y `thomyytripodi`) y su script, `python -m yourebrand.creative.evaluation`. Las verificaciones son deterministas: miran la traza del turno, no la opinión de otro modelo.
- Primera corrida: 9/11. Fallaban dos casos de repetición: el agente no verificaba una idea repetida si no se lo pedían, y el juez marcaba como parecido un episodio nuevo de una serie. Se ajustaron la descripción de `check_similarity` y el criterio del juez, y se sumaron tres casos de control que no se usaron para ajustar.
- Segunda corrida (`artifacts/evals_20261126_2050.json`): **13/14**. Tools 4/4, repetición 5/6, alcance 4/4.

**Limitación conocida (va al README):** el caso de control `repeticion-espontanea-carrusel` falla. Si una idea repite un post suelto pero se presenta como "segunda tanda", el juez la toma como otro episodio de una serie y la marca como distinta. Corrección posible: que una serie cuente solo si lo ya publicado la muestra (episodios numerados o una serie con nombre). No se aplicó para no convertir ese caso de control en uno de ajuste.

---

## Iteración 2b: Edición de video de punta a punta

**Objetivo:** que la persona suba un video crudo, apruebe el plan de edición y reciba el video editado desde la pantalla.

**Ya hecho (iteración 2):** subida de videos, preparación con ffmpeg y faster-whisper, `plan_video_edit` con Opus 5.5 y la skill `ugc-video-editing`, y la base del render en `creative/video_render.py` (subtítulos ASS, argumentos de ffmpeg, render en segundo plano con estado en disco).

**Implementar:**
- Pausa genérica en el grafo: si el modelo pide una tool con `requires_approval`, el nodo que ejecuta tools se detiene con `interrupt` antes de correrla. La persona aprueba o rechaza con un comentario. Al rechazar, la tool no corre y al modelo le vuelve el rechazo como resultado de la tool, así el historial queda válido.
- Tool `render_video(plan_id)` con `requires_approval`: con la aprobación arranca el render en segundo plano y vuelve enseguida con el id del render.
- Endpoints, todos colgados de la sesión: resolver la pausa, consultar el estado de un render y descargar el video terminado.
- Pantalla: tarjeta de aprobación con el resumen del plan, estado del render mientras corre y reproductor del video final.
- Tests: la pausa (aprobar y rechazar) con el proveedor simulado, la tool, los endpoints, y un render real de pocos segundos que se saltea si no hay ffmpeg.

**No implementar todavía:**
- Música y efectos: falta la librería propia de temas de libre uso.
- Editar el plan a mano en la pantalla: se modifica rechazando con un comentario.
- Cancelar un turno o un render en curso.
- Persistencia SQLite: la pausa vive en el checkpointer en memoria hasta la iteración 3.

**Listo cuando:** desde la pantalla se sube un video de `tambor`, el agente arma el plan, se pausa, se rechaza una vez con un comentario y el agente vuelve a planear, se aprueba, y el video editado se reproduce en la pantalla. La traza muestra la pausa y la decisión.

**Resultado (25/11):** cumplido con un video de `thomyytripodi`. Al probarlo aparecieron y se corrigieron dos problemas: un segundo rechazo chocaba con el límite de pasos del turno (ahora cada decisión de la persona reinicia los pasos y los topes por tool), y la pantalla mezclaba conversaciones si se cambiaba de sesión mientras el agente respondía. Los cortes solo se hacen en silencios de 1 s o más (`protect_speech`). Pedidos que el render no sabe hacer, como una placa final, el plan los avisa en vez de inventarlos.

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
- `check_similarity` también compara contra las ideas ya guardadas en Notion. Es una lectura, así que no requiere aprobación.
- Segundo cliente ficticio, de un rubro bien distinto a `tambor`.
- Casos nuevos en el set de evaluación: pedidos para el segundo cliente (respeta su estilo y no mezcla marcas) y pedidos de guardar (se espera la pausa antes de escribir).
- Test de aislamiento: lo generado para el cliente A no contiene información del cliente B, y un `client_id` inexistente o inválido falla.
- Test del recorrido principal completo, con Notion simulado: pedido, tools, pausa, aprobación, guardado.
- La pantalla pasa a soportar la pausa (aprobar, modificar o rechazar) y a retomar una sesión existente: el endpoint para resolver la pausa se adelanta de la iteración 4.

**No implementar todavía:**
- Memoria compartida entre sesiones o entre clientes.
- Usuarios y permisos por empleado.
- Supabase.

**Listo cuando:** desde la pantalla se generan ideas, se pide guardar dos, el agente se pausa, se cierra el proceso, se vuelve a abrir la misma sesión, se aprueba modificando una, y las dos aparecen en Notion. El test de aislamiento pasa.

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
- Casos nuevos en el set de evaluación para lo que falte cubrir del recorrido de la demo.
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
- Correr el set de evaluación completo y guardar los resultados en `artifacts/` con sus métricas: éxito de la tarea, tool elegida contra tool esperada, cantidad de pasos, latencia, costo y errores, agrupadas según los cuatro criterios (estilo de la marca, ideas no repetidas, uso de tools, aprobación antes de guardar).
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

Nada de esto se construye antes del 9 de diciembre. El detalle de cada punto está en `CLAUDE.md`, secciones "Fuera del TFI" y "Postergado: generación de imágenes".

- Datos reales de clientes en Supabase, con permisos por empleado.
- Retrieval con Qdrant, solo si el material de un cliente deja de entrar en el contexto.
- Extractor de tareas (Fathom a Notion).
- Agente Supervisor.
- Generación de imágenes.
