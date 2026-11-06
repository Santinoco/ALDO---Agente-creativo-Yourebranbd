"""Skills del Agente Creativo: criterio del dominio guardado en archivos, no en código.

Cada skill es una carpeta en `creative/skills/<nombre>/` con un `SKILL.md`:
un encabezado entre líneas `---` (nombre, descripción, estado, versión) y el
criterio en Markdown. Están en archivos para que alguien del equipo creativo
pueda ajustarlas sin tocar Python, y cada cambio quede versionado en git.

Se usan de dos formas:
- fija: `generate_ideas` aplica la skill del tipo de contenido pedido;
- decidida por el agente: su prompt lista las skills (nombre y descripción) y
  la tool `load_skill` le trae el criterio completo cuando lo necesita.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

SKILLS_DIR = Path(__file__).parent / "skills"
SKILL_FILENAME = "SKILL.md"
# Desde este título hasta el final, la skill le habla al agente y no a quien
# aplica el criterio (ver Skill.criteria).
AGENT_SECTION_HEADING = "### Para el agente"

# Mismo formato que un client_id: minúsculas, números y guiones.
_NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_FRONTMATTER = re.compile(r"\A---\n(?P<header>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)


class InvalidSkillError(ValueError):
    """Un SKILL.md no tiene el formato esperado."""


class UnknownSkillError(LookupError):
    """No hay ninguna skill con ese nombre."""


class Skill(BaseModel):
    """Una skill ya leída y validada."""

    name: str
    description: str = Field(min_length=1)
    status: Literal["borrador", "activa"] = "activa"
    version: str = "1.0"
    body: str = Field(min_length=1)  # el criterio, en Markdown

    @property
    def criteria(self) -> str:
        """El criterio sin la sección para el agente.

        Es lo que recibe una llamada que solo aplica el criterio (por ejemplo,
        la que genera ideas): las indicaciones sobre cuándo preguntar o cómo
        presentar son del agente y ahí solo confundirían.
        """
        criteria, _, _ = self.body.partition(AGENT_SECTION_HEADING)
        return criteria.strip()

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_PATTERN.fullmatch(value):
            raise ValueError(f"nombre de skill inválido: {value!r}")
        return value


def parse_skill(text: str, *, folder_name: str) -> Skill:
    """Lee el contenido de un SKILL.md. El nombre tiene que coincidir con la carpeta."""
    # Los archivos editados en Windows pueden traer \r\n: se normalizan antes de parsear.
    match = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    if match is None:
        raise InvalidSkillError(f"{folder_name}: falta el encabezado entre líneas ---")

    # Encabezado simple, una clave por línea ("clave: valor"). No hace falta
    # YAML completo, y así no se suma una dependencia.
    header: dict[str, str] = {}
    for line in match["header"].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise InvalidSkillError(f"{folder_name}: línea de encabezado sin ':': {line!r}")
        header[key.strip()] = value.strip()

    try:
        skill = Skill(**header, body=match["body"].strip())
    except ValidationError as exc:
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        raise InvalidSkillError(f"{folder_name}: encabezado inválido en {fields}") from None
    if skill.name != folder_name:
        raise InvalidSkillError(f"{folder_name}: el nombre del encabezado es {skill.name!r}")
    return skill


def load_skills(directory: Path = SKILLS_DIR) -> dict[str, Skill]:
    """Todas las skills de la carpeta, por nombre y en orden alfabético.

    Una skill mal armada hace fallar la carga entera: mejor enterarse al
    arrancar que cuando el agente la necesita a mitad de una conversación.
    """
    skills: dict[str, Skill] = {}
    for path in sorted(directory.glob(f"*/{SKILL_FILENAME}")):
        skill = parse_skill(path.read_text(encoding="utf-8"), folder_name=path.parent.name)
        skills[skill.name] = skill
    return skills


@lru_cache
def available_skills() -> dict[str, Skill]:
    """Las skills del agente, leídas una sola vez por proceso."""
    return load_skills()


def get_skill(name: str) -> Skill:
    try:
        return available_skills()[name]
    except KeyError:
        raise UnknownSkillError(f"no hay ninguna skill llamada {name!r}") from None
