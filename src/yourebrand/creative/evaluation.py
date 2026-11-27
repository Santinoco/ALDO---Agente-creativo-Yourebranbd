"""Set de evaluación del Agente Creativo: corre pedidos fijos y verifica qué decidió el agente.

Uso: python -m yourebrand.creative.evaluation [--case ID ...]

Los casos viven en `evals/cases.json`, editables sin tocar código. Cada caso es
una conversación de uno o más mensajes con lo que se espera de cada turno: qué
tools tiene que usar o evitar, si tiene que marcar una idea como repetida y
cómo tiene que ser la respuesta. Las verificaciones son deterministas: miran la
traza del turno, no la opinión de otro modelo (eso queda fuera del TFI).

Hace llamadas reales al modelo de `.env`. El resultado queda en `artifacts/`.
"""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yourebrand.creative.graph import AgentSession, TurnOutcome, build_graph, run_turn
from yourebrand.creative.tools import build_creative_registry

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "evals" / "cases.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts"

Criterion = Literal["tools", "repeticion", "alcance", "estilo", "aprobacion"]
ANY_TOOL = "*"
# "{idea_1}" en un mensaje se reemplaza por el id de la primera idea generada en la conversación.
_IDEA_PLACEHOLDER = re.compile(r"\{idea_(\d+)\}")
MAX_REPLY_IN_ARTIFACT = 1500


class TurnExpectation(BaseModel):
    """Qué tiene que pasar en un turno. Lo que no se indica no se verifica."""

    model_config = ConfigDict(extra="forbid")

    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)  # ["*"]: ninguna tool
    similarity_flagged: bool | None = None  # True: alguna idea marcada como repetida o parecida
    reply_must_not_contain: list[str] = Field(default_factory=list)
    reply_min_chars: int | None = None
    reply_max_chars: int | None = None


class EvalTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    expect: TurnExpectation = Field(default_factory=TurnExpectation)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    criterion: Criterion
    client_id: str
    description: str
    turns: list[EvalTurn] = Field(min_length=1)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TurnResult:
    message: str
    run_id: str = ""
    reply: str = ""
    tools_called: list[str] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    checks: list[Check] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.ok for check in self.checks)


@dataclass
class CaseResult:
    case: EvalCase
    turns: list[TurnResult]

    @property
    def passed(self) -> bool:
        return all(turn.passed for turn in self.turns)


def load_cases(path: Path = DEFAULT_CASES) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(item) for item in data["cases"]]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("hay ids de caso repetidos en el set de evaluación")
    return cases


def _fill_placeholders(message: str, idea_ids: list[str]) -> str:
    def replace(match: re.Match) -> str:
        index = int(match.group(1)) - 1
        if index >= len(idea_ids):
            raise LookupError(f"el mensaje pide {match.group(0)} y la conversación tiene {len(idea_ids)} idea(s)")
        return idea_ids[index]

    return _IDEA_PLACEHOLDER.sub(replace, message)


def _similarity_flagged(outcome: TurnOutcome) -> bool | None:
    """True si check_similarity marcó alguna idea; None si no corrió en el turno."""
    flagged = [event.fields.get("flagged", 0) for event in outcome.events if event.event == "similarity_checked"]
    return None if not flagged else any(count > 0 for count in flagged)


def check_turn(expect: TurnExpectation, outcome: TurnOutcome) -> list[Check]:
    """Compara lo que hizo el agente en un turno con lo esperado."""
    called = outcome.tools_called
    checks = []
    for tool in expect.must_call:
        checks.append(Check(f"usa {tool}", tool in called, f"llamó a: {called or 'ninguna'}"))
    for tool in expect.must_not_call:
        if tool == ANY_TOOL:
            checks.append(Check("no usa tools", not called, f"llamó a: {called or 'ninguna'}"))
        else:
            checks.append(Check(f"no usa {tool}", tool not in called, f"llamó a: {called or 'ninguna'}"))
    if expect.similarity_flagged is not None:
        flagged = _similarity_flagged(outcome)
        label = "marca una idea como parecida" if expect.similarity_flagged else "no marca ideas como parecidas"
        detail = "check_similarity no corrió" if flagged is None else f"marcó: {flagged}"
        checks.append(Check(label, flagged is expect.similarity_flagged, detail))
    reply = outcome.reply
    for text in expect.reply_must_not_contain:
        checks.append(Check(f"la respuesta no contiene {text!r}", text not in reply))
    if expect.reply_min_chars is not None:
        checks.append(Check(f"respuesta de al menos {expect.reply_min_chars} caracteres",
                            len(reply) >= expect.reply_min_chars, f"{len(reply)} caracteres"))
    if expect.reply_max_chars is not None:
        checks.append(Check(f"respuesta de hasta {expect.reply_max_chars} caracteres",
                            len(reply) <= expect.reply_max_chars, f"{len(reply)} caracteres"))
    if outcome.pending_approval:
        # Ningún caso de hoy espera una pausa: si aparece, el agente decidió escribir algo.
        checks.append(Check("no queda en pausa", False, "pidió una aprobación"))
    return checks


def run_case(graph, case: EvalCase) -> CaseResult:
    """Corre una conversación nueva con los mensajes del caso, uno por turno."""
    session = AgentSession(thread_id=f"eval-{case.id}-{uuid.uuid4().hex[:6]}", client_id=case.client_id)
    idea_ids: list[str] = []
    turns: list[TurnResult] = []
    for turn in case.turns:
        result = TurnResult(message=turn.message)
        turns.append(result)
        started = time.perf_counter()
        try:
            result.message = _fill_placeholders(turn.message, idea_ids)
            outcome = run_turn(graph, session, result.message)
        except Exception as exc:  # un caso que falla no corta el set: se registra y se sigue
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            break
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        result.run_id = outcome.run_id
        result.reply = outcome.reply
        result.tools_called = outcome.tools_called
        result.steps = sum(1 for step in outcome.trace if step.kind == "model")
        result.input_tokens = outcome.input_tokens
        result.output_tokens = outcome.output_tokens
        result.checks = check_turn(turn.expect, outcome)
        idea_ids += [idea.id for idea in outcome.ideas]
        if outcome.pending_approval:
            break  # la sesión no acepta mensajes hasta resolver la pausa
    return CaseResult(case=case, turns=turns)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_criterion: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_criterion.setdefault(result.case.criterion, {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += result.passed
    turns = [turn for result in results for turn in result.turns]
    return {
        "cases": len(results),
        "passed": sum(result.passed for result in results),
        "by_criterion": by_criterion,
        "input_tokens": sum(turn.input_tokens for turn in turns),
        "output_tokens": sum(turn.output_tokens for turn in turns),
        "latency_ms": round(sum(turn.latency_ms for turn in turns), 1),
    }


def to_artifact(results: list[CaseResult], *, model_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": model_info,
        "summary": summarize(results),
        "cases": [
            {
                "id": result.case.id,
                "criterion": result.case.criterion,
                "client_id": result.case.client_id,
                "description": result.case.description,
                "passed": result.passed,
                "turns": [
                    {
                        "message": turn.message,
                        "run_id": turn.run_id,
                        "passed": turn.passed,
                        "tools_called": turn.tools_called,
                        "steps": turn.steps,
                        "input_tokens": turn.input_tokens,
                        "output_tokens": turn.output_tokens,
                        "latency_ms": turn.latency_ms,
                        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in turn.checks],
                        "error": turn.error,
                        "reply": turn.reply[:MAX_REPLY_IN_ARTIFACT],
                    }
                    for turn in result.turns
                ],
            }
            for result in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corre el set de evaluación del Agente Creativo.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", action="append", dest="only", help="corre solo este caso (se puede repetir)")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(errors="replace")

    from yourebrand.core.config import get_settings

    settings = get_settings()
    cases = load_cases(args.cases)
    if args.only:
        cases = [case for case in cases if case.id in set(args.only)]
    graph = build_graph(build_creative_registry(video_edit=False))

    results = []
    for case in cases:
        result = run_case(graph, case)
        results.append(result)
        print(f"{'OK   ' if result.passed else 'FALLA'} {case.id} [{case.criterion}]")
        for turn in result.turns:
            for check in turn.checks:
                if not check.ok:
                    print(f"        ✗ {check.name} ({check.detail})")
            if turn.error:
                print(f"        ✗ error: {turn.error}")

    artifact = to_artifact(results, model_info={
        "provider": settings.llm_provider,
        "generation": settings.model_generation,
        "fast": settings.model_fast,
    })
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"evals_{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = artifact["summary"]
    print(f"\n{summary['passed']}/{summary['cases']} casos OK · "
          f"{summary['input_tokens']}/{summary['output_tokens']} tokens · {summary['latency_ms'] / 1000:.0f} s")
    for criterion, bucket in summary["by_criterion"].items():
        print(f"  {criterion}: {bucket['passed']}/{bucket['cases']}")
    print(f"Resultado: {path}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
