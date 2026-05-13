from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi.responses import StreamingResponse


def _encode_event(event_type: str, data: Any, event_id: Optional[str] = None) -> str:
    """Format a single SSE event string."""
    payload = json.dumps(data, ensure_ascii=False)
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append(f"data: {payload}")
    lines.append("")  # blank line to terminate event
    return "\n".join(lines) + "\n"


def meta_event(data: Dict[str, Any]) -> str:
    return _encode_event("meta", data)


def delta_event(text: str) -> str:
    return _encode_event("delta", {"text": text})


def clarification_event(text: str) -> str:
    return _encode_event("clarification", {"text": text})


def citations_event(citations: list) -> str:
    return _encode_event("citations", {"citations": citations})


def done_event(data: Optional[Dict[str, Any]] = None) -> str:
    return _encode_event("done", data or {})


def error_event(message: str, code: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"message": message}
    if code:
        payload["code"] = code
    return _encode_event("error", payload)


def make_sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Wrap an async generator of SSE event strings into a StreamingResponse."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
