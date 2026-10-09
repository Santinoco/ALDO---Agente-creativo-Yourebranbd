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
        description="Qué se ve y qué pasa en la pieza, con el detalle suficiente para que alguien pueda producirla."
    )
    rationale: NonEmptyStr = Field(
        description="Por qué la idea encaja con esta marca en particular: qué rasgo, público u objetivo suyo aprovecha."
    )


class IdeaList(BaseModel):
    """Respuesta completa del modelo a un pedido de ideas."""

    model_config = ConfigDict(extra="forbid")

    ideas: list[Idea] = Field(min_length=1, description="Ideas de contenido, cada una distinta de las demás.")
