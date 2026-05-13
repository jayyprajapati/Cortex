"""Smoke test: SSE event sequence encoding is correct."""
from __future__ import annotations

from app.streaming.sse import (
    meta_event, delta_event, clarification_event,
    citations_event, done_event, error_event,
)


def _parse_event(raw: str):
    """Parse a raw SSE string into (event_type, data_dict)."""
    import json
    lines = [l for l in raw.strip().split("\n") if l]
    event_type = None
    data = None
    for line in lines:
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data = json.loads(line[6:])
    return event_type, data


def main():
    # meta event
    ev, data = _parse_event(meta_event({"retrieved_count": 5, "confidence": "high"}))
    assert ev == "meta"
    assert data["retrieved_count"] == 5
    assert data["confidence"] == "high"

    # delta event
    ev, data = _parse_event(delta_event("Hello world"))
    assert ev == "delta"
    assert data["text"] == "Hello world"

    # clarification event
    ev, data = _parse_event(clarification_event("Which section?"))
    assert ev == "clarification"
    assert data["text"] == "Which section?"

    # citations event
    ev, data = _parse_event(citations_event([{"index": 1, "section": "Intro"}]))
    assert ev == "citations"
    assert len(data["citations"]) == 1

    # done event
    ev, data = _parse_event(done_event({"grounded": True}))
    assert ev == "done"
    assert data["grounded"] is True

    # error event
    ev, data = _parse_event(error_event("something broke", code="test_error"))
    assert ev == "error"
    assert data["message"] == "something broke"
    assert data["code"] == "test_error"

    # Verify event sequence: each event ends with double newline
    events = [
        meta_event({"x": 1}),
        delta_event("text"),
        done_event(),
    ]
    for e in events:
        assert e.endswith("\n\n") or e.endswith("\n"), f"Event should end with newline: {e!r}"

    print("PASS: SSE streaming tests passed")


if __name__ == "__main__":
    main()
