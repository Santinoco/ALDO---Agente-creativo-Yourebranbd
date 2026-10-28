"""Entrada de terminal del Agente Creativo.

Uso: python -m yourebrand.creative.cli tambor ugc 5

Hace una llamada real al modelo configurado en `.env`. Hoy pide una lista de
ideas; en la iteración 2 pasa a ser el chat con el agente.
"""

import argparse
import logging
import sys
from typing import get_args

from yourebrand.core.errors import LLMError
from yourebrand.core.llm import new_run_id
from yourebrand.core.tenancy import InvalidClientIdError, TenantContext, UnknownClientError
from yourebrand.creative.ideas import MAX_IDEAS, generate_ideas
from yourebrand.creative.schemas import ContentType, Idea


def _print_idea(number: int, idea: Idea) -> None:
    print(f"\n{number}. {idea.title}")
    print(f"   Formato: {idea.format} | Pilar: {idea.pillar}")
    print(f"   {idea.description}")
    print(f"   Por qué: {idea.rationale}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pide ideas de contenido para un cliente.")
    parser.add_argument("client_id", help="carpeta del cliente dentro de data/, por ejemplo: tambor")
    parser.add_argument("content_type", choices=get_args(ContentType))
    parser.add_argument("count", type=int, nargs="?", default=5, help=f"cantidad de ideas, de 1 a {MAX_IDEAS}")
    args = parser.parse_args(argv)

    # El registro de core/llm.py (tokens, latencia, reintentos) sale por stderr,
    # así la salida estándar queda solo con las ideas.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # La consola de Windows no siempre puede mostrar todo lo que escribe el modelo.
    sys.stdout.reconfigure(errors="replace")

    run_id = new_run_id()
    try:
        ideas = generate_ideas(TenantContext(args.client_id), args.content_type, args.count, run_id=run_id)
    except (InvalidClientIdError, UnknownClientError, ValueError, LLMError) as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(f"run_id={run_id}")
    for number, idea in enumerate(ideas, start=1):
        _print_idea(number, idea)
    return 0


if __name__ == "__main__":
    sys.exit(main())
