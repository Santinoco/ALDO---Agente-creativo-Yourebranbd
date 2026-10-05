# Plan de desarrollo por iteraciones

Cada iteración termina con algo que funciona y se puede mostrar. No se empieza una iteración hasta cumplir el "Listo cuando" de la anterior.

Si aparece una idea que no está en la iteración actual, se anota en la iteración donde corresponde y se sigue con lo que toca.

| # | Iteración | Estado |
|---|---|---|
| 0 | Base común (`core/`) | Hecha |
| 1 | Creativo mínimo: ideas desde archivos | **Siguiente** |
| 2 | Aislamiento y evaluación | Pendiente |
| 3 | Refinamiento de ideas | Pendiente |
| 4 | Datos reales en Supabase | Pendiente |
| 5 | API y uso por el equipo | Pendiente |
| 6 | Retrieval con Qdrant (condicional) | Pendiente |
| 7 | Extractor de tareas (Fathom a Notion) | Pendiente |
| 8 | Agente Supervisor | Pendiente |
| - | Generación de imágenes | Postergada, sin fecha |

---

## Iteración 0: Base común (hecha)

**Objetivo:** tener la infraestructura que usan todos los componentes.

**Implementado:**
- `core/config.py`: configuración con `get_settings()`.
- `core/tenancy.py`: `TenantContext` y `client_data_dir`.
- `core/providers.py`: contrato `LLMProvider` y `AnthropicProvider`.
- `core/llm.py`: `complete`, con registro de tokens, latencia y `run_id`.
- Tests de los cuatro módulos.

**Quedó pendiente:** probar una llamada real a Claude. Se resuelve al empezar la iteración 1.

---

## Iteración 1: Creativo mínimo

**Objetivo:** pedir ideas para un cliente ficticio y recibir una lista de ideas alineadas a su marca.

**Implementar:**
- `.env` con `ANTHROPIC_API_KEY` y una llamada real de prueba.
- Cliente ficticio en `data/acme/`: `brand.md` (información de la empresa) y `posts.md` (posts anteriores en texto, opcional).
- `creative/context.py`: lee marca y posts del cliente. Funciona aunque no haya posts.
- `creative/prompts.py`: arma las instrucciones y el pedido.
- `core/llm.py`: función para salida estructurada (que el modelo devuelva una lista validada con Pydantic).
- `creative/ideas.py`: `generate_ideas(tenant, content_type, count)`, con `content_type` igual a `"ugc"` o `"design"`.
- Una forma de ejecutarlo desde la terminal, para ver las ideas sin levantar un servidor.
- Tests con el proveedor simulado.

**No implementar todavía:**
- Qdrant, Supabase, FastAPI.
- Refinamiento de ideas o conversación de ida y vuelta.
- Copys, guiones o imágenes: solo ideas.
- Guardar las ideas generadas en algún lado.

**Listo cuando:** desde la terminal se piden 5 ideas UGC y 5 de diseño para `acme`, y las ideas mencionan cosas propias de esa marca.

---

## Iteración 2: Aislamiento y evaluación

**Objetivo:** demostrar que los clientes no se mezclan y tener una medida de calidad contra la cual comparar cambios futuros.

**Implementar:**
- Segundo cliente ficticio, de un rubro bien distinto al primero.
- Test de aislamiento: lo generado para el cliente A no contiene información del cliente B, y un `client_id` inexistente o inválido falla.
- Set de evaluación: una lista fija de pedidos por cliente.
- Script que corre el set y guarda las ideas en `artifacts/`.
- Rúbrica simple para puntuar a mano (encaja con la marca, es original, es realizable) y la primera puntuación como línea base.

**No implementar todavía:**
- Evaluación automática con otro modelo como juez.
- Dashboards o herramientas de observabilidad.

**Listo cuando:** el test de aislamiento pasa y hay un reporte en `artifacts/` con la línea base puntuada.

---

## Iteración 3: Refinamiento de ideas

**Objetivo:** que el empleado pueda mejorar una idea en vez de pedir todo de nuevo.

**Implementar:**
- `refine_idea(tenant, idea, feedback)`: devuelve la idea ajustada según el comentario.
- Que las ideas nuevas no repitan lo que ya está en los posts anteriores.
- Ajustes de prompt guiados por la evaluación de la iteración 2, volviendo a medir después de cada cambio.

**No implementar todavía:**
- Memoria de conversaciones entre sesiones.
- LangGraph o agentes que deciden pasos por su cuenta.

**Listo cuando:** se puede refinar una idea con un comentario y la puntuación del set de evaluación no empeora respecto de la línea base.

---

## Iteración 4: Datos reales en Supabase

**Objetivo:** dejar de leer de `data/` y trabajar con clientes reales.

**Implementar:**
- Tablas en Supabase: clientes, información de marca, posts anteriores.
- `core/db.py`: toda consulta exige un `TenantContext` y filtra por `client_id` en un solo lugar.
- `creative/context.py` pasa a leer de Supabase; `data/` queda solo para tests.
- Carga de uno o dos clientes reales.
- Guardar las ideas generadas, con su `run_id`.
- Test de aislamiento repetido contra la base.

**No implementar todavía:**
- Usuarios y permisos por empleado.
- Carga automática de posts desde las redes.

**Listo cuando:** se generan ideas para un cliente real leyendo de Supabase y el test de aislamiento pasa contra la base.

---

## Iteración 5: API y uso por el equipo

**Objetivo:** que un empleado de la agencia use el agente sin tocar código.

**Decidir antes de empezar:** por dónde lo va a usar el equipo (una web simple, Slack, Notion u otra cosa).

**Implementar:**
- `api/`: endpoints de FastAPI para generar y refinar ideas. Sin lógica de negocio, solo llaman a `creative/`.
- Autenticación básica y qué empleado puede ver qué cliente.
- La interfaz elegida, en su versión más simple.
- Prueba con dos o tres personas del equipo y registro de lo que piden cambiar.

**No implementar todavía:**
- Despliegue con alta disponibilidad o escalado.
- Funciones que nadie del equipo pidió.

**Listo cuando:** una persona de la agencia genera ideas para un cliente real por su cuenta.

---

## Iteración 6: Retrieval con Qdrant (condicional)

**Solo se hace si** el material de algún cliente ya no entra en el prompt, o si la calidad cae por exceso de contexto. Si no pasa, se saltea.

**Objetivo:** mandar al modelo solo el material relevante para cada pedido.

**Implementar:**
- `core/vectorstore.py`: búsqueda que exige `TenantContext` y filtra por `client_id`.
- Indexado de marca y posts por cliente.
- `creative/context.py` usa la búsqueda en vez de mandar todo.
- Comparación con el set de evaluación: con retrieval contra sin retrieval.

**No implementar todavía:**
- Reranking, búsqueda híbrida u otras mejoras, salvo que la medición las pida.

**Listo cuando:** la medición muestra que con retrieval la calidad es igual o mejor y el test de aislamiento pasa contra Qdrant.

---

## Iteración 7: Extractor de tareas

**Objetivo:** convertir una reunión de Fathom en tareas sugeridas en Notion.

**Implementar:**
- Lectura de una reunión de Fathom.
- Detección del cliente al que corresponde.
- Destilado de la reunión con el modelo rápido (no se guarda cruda).
- Extracción de tareas con salida estructurada.
- Creación en Notion como "sugeridas".

**No implementar todavía:**
- Confirmación automática de tareas: siempre las aprueba una persona.
- Asignación automática de responsables o fechas sin revisión.

**Listo cuando:** una reunión de prueba genera tareas sugeridas en Notion, asociadas al cliente correcto.

---

## Iteración 8: Agente Supervisor

**Objetivo:** ayudar al área de supervisión a controlar plazos, coherencia de marca y contenido sin subir.

**Implementar:**
- Registro de qué se publicó y qué está planificado.
- Chequeos de plazos y de contenido sin subir.
- Chequeo de coherencia de marca sobre contenido ya escrito.
- LangGraph para los puntos donde una persona aprueba o rechaza.

**No implementar todavía:**
- Acciones automáticas sobre el contenido: el supervisor avisa, no corrige solo.

**Listo cuando:** el área de supervisión recibe un reporte útil de un cliente real y puede aprobar o rechazar lo que el agente marca.

---

## Postergado: generación de imágenes

No tiene iteración asignada. Se retoma cuando la tecnología esté madura para el uso de la agencia. Lo ya investigado está en `CLAUDE.md`, sección "Postergado: generación de imágenes".
