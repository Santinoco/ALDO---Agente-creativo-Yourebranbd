"""Eventos de trazabilidad: log, traza del turno, datos privados, anidamiento e hilos."""

import logging
import threading

from yourebrand.core.reliability import run_with_deadline
from yourebrand.core.trace import collect, emit, nested

logger = logging.getLogger("test.trace")


def test_emit_without_open_trace_only_logs(caplog):
    with caplog.at_level(logging.INFO, logger="test.trace"):
        emit(logger, "algo", run_id="r1", count=3)

    assert "algo run_id=r1 count=3" in caplog.text


def test_emit_formats_floats_and_lists(caplog):
    with caplog.at_level(logging.INFO, logger="test.trace"):
        emit(logger, "algo", latency_ms=12.3456, tools=["a", "b"], empty=[])

    assert "latency_ms=12.35 tools=a,b empty=-" in caplog.text


def test_collect_keeps_fields_and_private_but_private_is_not_logged(caplog):
    with caplog.at_level(logging.INFO, logger="test.trace"), collect("run-1") as collector:
        emit(logger, "tool_start", private={"arguments": {"brief": "dato del cliente"}}, tool="generate_ideas")

    [event] = collector.events
    assert event.run_id == "run-1"
    assert event.event == "tool_start"
    assert event.fields == {"tool": "generate_ideas"}
    assert event.private == {"arguments": {"brief": "dato del cliente"}}
    assert event.at_ms >= 0
    assert "dato del cliente" not in caplog.text


def test_events_are_in_order_with_growing_time():
    with collect("run-1") as collector:
        for name in ("a", "b", "c"):
            emit(logger, name)

    events = collector.events
    assert [e.event for e in events] == ["a", "b", "c"]
    assert events[0].at_ms <= events[1].at_ms <= events[2].at_ms


def test_nothing_is_collected_after_the_trace_closes():
    with collect("run-1") as collector:
        emit(logger, "adentro")
    emit(logger, "afuera")

    assert [e.event for e in collector.events] == ["adentro"]


def test_nested_marks_depth():
    with collect("run-1") as collector:
        emit(logger, "afuera")
        with nested():
            emit(logger, "adentro")
            with nested():
                emit(logger, "mas_adentro")
        emit(logger, "de_vuelta")

    assert [(e.event, e.depth) for e in collector.events] == [
        ("afuera", 0), ("adentro", 1), ("mas_adentro", 2), ("de_vuelta", 0),
    ]


def test_events_from_run_with_deadline_thread_reach_the_trace():
    # Las tools y las llamadas al proveedor corren en otro hilo: tienen que
    # seguir emitiendo a la traza del turno, con su profundidad.
    with collect("run-1") as collector, nested():
        run_with_deadline(lambda: emit(logger, "en_otro_hilo"), timeout_seconds=5)

    [event] = collector.events
    assert (event.event, event.depth, event.run_id) == ("en_otro_hilo", 1, "run-1")


def test_concurrent_turns_do_not_mix():
    collected = {}

    def turn(run_id):
        with collect(run_id) as collector:
            for _ in range(50):
                emit(logger, "paso", run=run_id)
        collected[run_id] = collector.events

    threads = [threading.Thread(target=turn, args=(f"run-{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for run_id, events in collected.items():
        assert len(events) == 50
        assert all(e.run_id == run_id and e.fields["run"] == run_id for e in events)
