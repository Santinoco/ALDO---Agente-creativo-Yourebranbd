"""Forma de las ideas que devuelve el Agente Creativo.

Estos modelos se le pasan al modelo de lenguaje como schema de salida: las
descripciones de cada campo son parte de las instrucciones que recibe.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ContentType = Literal["ugc", "design"]

# Texto obligatorio: sin espacios sobrantes y nunca vacío.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Idea(BaseModel):
    """Una idea de contenido: describe la pieza, no es la pieza final."""

    # Un campo que no está en el schema es un error, no un dato que se ignora.
    model_config = ConfigDict(extra="forbid")

    title: NonEmptyStr = Field(description="Nombre corto de la idea, para reconocerla en una lista.")
    format: NonEmptyStr = Field(
        description="Formato de la pieza, por ejemplo: reel, carrusel, placa, historia, foto."
    )
    pillar: NonEmptyStr = Field(
        description="Pilar de contenido de la marca al que responde la idea, tal como figura en su información."
    )
    description: NonEmptyStr = Field(
        description="Qué se ve y qué pasa en la pieza, en dos o tres oraciones. Sin pasos de producción."
    )
    rationale: NonEmptyStr = Field(
        description="Por qué la idea encaja con esta marca en particular, en una oración: "
        "qué rasgo, público u objetivo suyo aprovecha."
    )


class GeneratedIdea(Idea):
    """Idea ya generada dentro de una sesión: suma lo que pone el sistema, no el modelo.

    El id permite que el agente y la persona se refieran a una idea ("la idea-3f2a1c")
    sin reescribirla, y que las tools siguientes (refinar, guardar) la encuentren.
    """

    id: NonEmptyStr
    content_type: ContentType
    # Si es un ajuste de refine_idea, el id de la idea de la que salió. La
    # original no se pisa: la persona elige después cuál guardar.
    refined_from: str | None = None


class IdeaList(BaseModel):
    """Respuesta completa del modelo a un pedido de ideas."""

    model_config = ConfigDict(extra="forbid")

    ideas: list[Idea] = Field(min_length=1, description="Ideas de contenido, cada una distinta de las demás.")

class SimilarityVerdict(BaseModel):
    """Juicio sobre una idea frente a lo ya publicado."""

    model_config = ConfigDict(extra="forbid")

    idea_id: NonEmptyStr = Field(description="Id de la idea juzgada, tal como figura en el material.")
    # El motivo va antes del veredicto: el modelo razona primero y concluye después.
    reason: NonEmptyStr = Field(description="En una oración, qué coincide con lo publicado o qué cambia.")
    verdict: Literal["repetida", "parecida", "distinta"] = Field(description="Conclusión de la comparación.")
    reference_ids: list[str] = Field(description="Ids de las referencias parecidas. Vacío si es 'distinta'.")


class SimilarityReport(BaseModel):
    """Respuesta completa del modelo al comparar ideas."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[SimilarityVerdict] = Field(description="Un veredicto por cada idea nueva.")



# --- Plan de edición de video UGC --------------------------------------------
#
# Sigue las secciones de "Qué devolvés" de la skill ugc-video-editing. Los
# tiempos son segundos del video original: el render los usa tal cual.


class TimeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(description="Inicio, en segundos del video original.")
    end_s: float = Field(description="Fin, en segundos del video original. Mayor que el inicio.")


class BRoll(TimeSpan):
    """Material de apoyo: un tramo del video original que se ve encima de la voz.

    start_s y end_s dicen de dónde sale la imagen; offset_s, cuándo aparece
    dentro del tramo. El audio sigue siendo el del tramo.
    """

    offset_s: float = Field(
        ge=0,
        description="Cuándo aparece, en segundos desde el inicio del tramo al que pertenece. "
        "Tiene que caer dentro del tramo.",
    )
    description: NonEmptyStr = Field(description="Qué muestra este material de apoyo y por qué va acá.")


class EditSegment(TimeSpan):
    purpose: NonEmptyStr = Field(description="Para qué está el tramo: gancho, problema, prueba, cierre, etc.")
    zoom: bool = Field(description="Si lleva zoom para dar movimiento a un plano fijo de alguien hablando.")
    b_roll: list[BRoll] = Field(
        description="Material de apoyo que va encima de la voz durante este tramo. Vacío si no lleva."
    )


class OnScreenText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: NonEmptyStr = Field(description="Palabras que se resaltan. Sin datos que no estén en la marca.")
    at_s: float = Field(description="Momento del video final, en segundos, en que aparece.")


class VideoEditPlan(BaseModel):
    """Plan de edición de un video UGC, para que una persona lo apruebe antes del render."""

    model_config = ConfigDict(extra="forbid")

    video_type: Literal["recomendacion", "venta"] = Field(
        description="Recomendación de usuario o venta / testimonio. Define el gancho y el cierre."
    )
    hook: NonEmptyStr = Field(description="Qué momento abre el video y por qué funciona como gancho.")
    segments: list[EditSegment] = Field(
        min_length=1, description="Tramos en el orden del video final, que no tiene por qué ser el de grabación."
    )
    subtitle_style: Literal["palabra", "frase"] = Field(
        description="Subtítulos palabra por palabra (preferido) o frase por frase con frases cortas."
    )
    on_screen_texts: list[OnScreenText] = Field(description="Textos resaltados. Pocos: no recargar la pantalla.")
    music: NonEmptyStr = Field(description="Qué tipo de música de libre uso acompaña y por qué, según el tono.")
    sound_effects: list[str] = Field(description="Efectos de sonido, muy pocos. Vacío si no lleva.")
    call_to_action: NonEmptyStr = Field(description="Cierre con llamado a la acción, natural para el tipo de video.")
    target_network: NonEmptyStr = Field(description="Red de destino. Si no se sabe, Reels, y decilo en advertencias.")
    warnings: list[str] = Field(
        description="Lo que no se pudo resolver con el material: audio malo, falta b-roll, un dato a confirmar."
    )

    @property
    def duration_s(self) -> float:
        """Duración final estimada: la suma de los tramos (el b-roll va encima, no suma)."""
        return round(sum(segment.end_s - segment.start_s for segment in self.segments), 2)
