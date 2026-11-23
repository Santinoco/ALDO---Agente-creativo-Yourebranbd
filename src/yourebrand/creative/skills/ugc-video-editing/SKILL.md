---
name: ugc-video-editing
description: Criterio de Yourebrand para convertir un video UGC crudo en una pieza para Reels, TikTok o historias sin perder lo que la persona dijo o mostró, con gancho, ritmo y zooms. Usala cuando la persona sube un video para editar o pide cómo editar una idea UGC.
status: activa
version: 2.1
---

# Edición de videos UGC

Editar es **darle ritmo a lo que la persona grabó, sin quitarle contenido**. El resultado tiene que sentirse dinámico, limpio y hecho por una persona, y decir todo lo que el crudo decía.

## Las cinco reglas que más pesan

1. **No se saca contenido por criterio propio.** Si alguien lo dijo o lo mostró, se queda. Un video más largo y completo siempre es mejor que uno corto e incompleto.
2. **Nunca se corta entre palabras ni en medio de una frase.** Solo se corta en un silencio largo.
3. **Un silencio puede ser importante.** Si en ese momento se muestra algo, se queda.
4. **Dos tramos seguidos del mismo plano nunca tienen el mismo zoom.** Es lo que convierte un salto feo en un corte con ritmo.
5. **Ningún plano queda quieto más de 4 segundos.** Se parte en dos tramos contiguos y cambia el zoom, sin tocar el audio.

## Qué se puede sacar

Solo esto, y siempre como un bloque entero entre dos silencios largos:

- **Una toma repetida o fallida:** la persona dice lo mismo otra vez. Quedate con la mejor, casi siempre la última.
- **Lo que no es parte de la pieza:** indicaciones fuera de cámara, "¿ya está grabando?", "lo digo de nuevo".
- **Un silencio largo en el que no pasa nada.**

No se sacan: saludos, despedidas, agradecimientos, muletillas sueltas, frases que parecen poco importantes ni el final del video. **Ante la duda, se deja.** Si creés que algo convendría sacarlo, no lo saques: proponelo en advertencias y que la persona lo decida.

## Cómo se aplican los cortes

La app protege el habla: **solo corta en los silencios largos listados en `<silencios_largos>`**. Cada zona de video entre dos de esos silencios es un bloque que entra entero. Si marcás un tramo que toca un bloque, entra todo el bloque, aunque pongas otros tiempos.

Por eso, tu trabajo con los tiempos es:

- **Elegir qué bloques entran y en qué orden.** Por defecto, todos y en el orden original.
- **Decidir cada silencio largo:** mirá los fotogramas de ese momento.
  - Si no pasa nada, dejá que se corte: un tramo termina antes del silencio y el siguiente empieza después.
  - **Si se muestra algo** (un producto, un gesto, una acción, un cambio de lugar), conservalo: usá **un solo tramo que cruce el silencio completo**.
  - Si no podés saberlo, conservalo.
- **Partir los bloques para dar ritmo:** tramos contiguos, con el fin de uno exactamente igual al inicio del siguiente. Esa división solo cambia el zoom: la voz sigue continua.

## Cómo trabajar, en este orden

### 1. Leé todo y entendé qué hay

Recorré la transcripción completa y los fotogramas. Marcá tomas repetidas, indicaciones fuera de cámara y qué pasa en cada silencio largo. Los tiempos de las palabras son aproximados: una palabra puede figurar más larga de lo que suena.

### 2. Elegí el orden

- **Por defecto, el orden original.**
- Mové un bloque al principio solo si es **claramente** un mejor gancho y la historia se sigue entendiendo.
- Si no hay un gancho fuerte, no lo fuerces: dejá el arranque como está y proponé en advertencias cómo mejorarlo.

Según el tipo de video:

- **Recomendación:** qué es, qué probó, qué le gustó, si vuelve.
- **Venta / testimonio:** problema, cómo lo resolvió, resultado.

### 3. Dale ritmo con los zooms

El render tiene dos encuadres: **abierto** (`zoom: false`) y **cerrado** (`zoom: true`, un acercamiento fijo).

- **Alterná siempre** entre tramos seguidos del mismo plano: abierto, cerrado, abierto.
- **El cerrado va en lo importante:** el gancho, el dato, la palabra fuerte, el remate.
- **Duración de cada plano:** entre **1,5 y 3,5 s** como referencia. Variá: un ritmo idéntico también aburre.
- **Sin transiciones:** solo cortes limpios. Nada de fundidos, barridos ni efectos entre planos.

### 4. Usá el b-roll si existe

- El b-roll sale **del mismo video**: solo planos realmente distintos a la persona hablando (producto, lugar, manos, detalle). Revisá los fotogramas.
- Mostrá **lo que se está diciendo en ese momento**, de 1 a 2,5 s.
- Si todo es un único plano de alguien hablando, **no fuerces b-roll**: dejá la lista vacía y resolvé el movimiento con zooms.

## Textos en pantalla

Los subtítulos palabra por palabra los genera el render a partir de la transcripción: no los escribas. `on_screen_texts` es otra cosa, un **título o dato arriba del video**, durante 2 segundos.

- **Título en el segundo 0** solo cuando sume: de 2 a 5 palabras sobre lo que se dice en el video.
- Después, **como mucho uno cada 10 o 15 segundos**, para un precio, un número o una idea clave.
- **No repitas lo que ya dicen los subtítulos en ese momento.**
- `at_s` es el momento **del video final**, no del original.
- **Datos solo si se dicen en el video o están en la marca.** Nunca inventes precios, plazos ni promociones.

Elegí `subtitle_style: "palabra"` salvo que haya un motivo claro.

## Cierre

- Conservá el final tal como lo grabó la persona.
- **Siempre tiene que haber un llamado a la acción.** Si está en el material, señalalo en `call_to_action`. Si no está, escribí ahí qué conviene sumar y avisalo: no lo inventes en el video.
- No sumes logo ni placa final.

## Sonido

- En `music` describí en una línea el tipo de música de libre uso que le va. Hoy el render **no la agrega**: queda como indicación para el retoque final.
- Dejá `sound_effects` vacío, salvo un motivo muy claro.

## Duración

La duración la define el material, no un objetivo. La referencia es de hasta **1:20**: si el video queda más largo, no saques contenido para llegar, y avisalo en advertencias.

## La marca

- De la marca salen el tono y los límites de "qué evitar". Esos límites son firmes: si el material cruza uno, marcalo en advertencias.
- **Si el video no parece de esta marca, editalo igual** con este criterio. Avisalo en **una** advertencia y no le exijas lo que la marca pide para su contenido habitual (precio, ubicación, formato).

## Advertencias

Son para la persona que aprueba: **pocas, cortas y accionables**. Máximo 4, primero la más importante.

- Sí: una palabra clave mal transcripta (los subtítulos la van a mostrar así), un dato sensible a confirmar, un bloque que propondrías sacar, un gancho débil.
- No: explicar cada decisión ni repetir el plan.

## Antes de devolver el plan, verificá

- [ ] Entran todos los bloques con voz, salvo tomas repetidas o indicaciones fuera de cámara.
- [ ] El video conserva su final.
- [ ] Revisaste qué se ve en cada silencio largo, y los que muestran algo cruzan dentro de un tramo.
- [ ] No hay dos tomas de la misma frase.
- [ ] No hay dos tramos seguidos del mismo plano con el mismo zoom.
- [ ] Ningún plano queda quieto más de 4 s.
- [ ] Solo cortes limpios, sin transiciones.
