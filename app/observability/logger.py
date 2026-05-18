from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Per-request context variables (asyncio-safe, set by RequestContextMiddleware)
# ---------------------------------------------------------------------------
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def set_request_context(request_id: str, user_id: str = "") -> None:
    """Set request_id and user_id in the current async context."""
    _request_id_var.set(request_id)
    _user_id_var.set(user_id or "")


def get_request_context() -> dict:
    """Return a dict with the current request_id and user_id."""
    return {
        "request_id": _request_id_var.get(),
        "user_id": _user_id_var.get(),
    }


# ---------------------------------------------------------------------------
# Logging filter — injects context vars into every log record
# ---------------------------------------------------------------------------

class CortexContextFilter(logging.Filter):
    """Inject request_id and user_id from ContextVars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()  # type: ignore[attr-defined]
        record.user_id = _user_id_var.get()         # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Root "cortex" logger — JSON format via python-json-logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger("cortex")

def _configure_json_logger() -> None:
    """Configure the root cortex logger to emit JSON lines."""
    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter
    except ImportError:
        # Graceful fallback: python-json-logger not installed; use plain text
        return

    if _logger.handlers:
        # Already configured (e.g. during hot-reload); skip double-setup
        return

    handler = logging.StreamHandler()
    fmt = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(fmt)
    handler.addFilter(CortexContextFilter())
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # don't double-emit to root logger


_configure_json_logger()


# ---------------------------------------------------------------------------
# Audit logger — separate handler writing to audit.log
# ---------------------------------------------------------------------------

_AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit.log")
_audit_logger = logging.getLogger("cortex.audit")


def _configure_audit_logger() -> None:
    """Configure the cortex.audit logger to write JSON to audit.log."""
    if _audit_logger.handlers:
        return

    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter
        fmt = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
        )
    except ImportError:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")  # type: ignore[assignment]

    try:
        file_handler = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
    except OSError:
        # If the path is not writable (e.g. read-only filesystem in tests), fall back to stderr
        file_handler = logging.StreamHandler()  # type: ignore[assignment]

    file_handler.setFormatter(fmt)
    file_handler.addFilter(CortexContextFilter())
    _audit_logger.addHandler(file_handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


_configure_audit_logger()


def audit_log(
    action: str,
    *,
    user_id: str = "",
    resource: str = "",
    request_id: str = "",
    **extra: Any,
) -> None:
    """
    Emit a security-relevant audit event to cortex.audit logger.

    Args:
        action:     Event type — e.g. "ingest", "delete", "thread_access", "admin_action".
        user_id:    Identity of the actor (falls back to ContextVar if omitted).
        resource:   Resource being acted on (doc_id, thread_id, collection, …).
        request_id: Correlation ID (falls back to ContextVar if omitted).
        **extra:    Any additional key-value pairs to include in the log record.
    """
    ctx = get_request_context()
    _audit_logger.info(
        action,
        extra={
            "audit_action": action,
            "audit_user_id": user_id or ctx["user_id"],
            "audit_resource": resource,
            "audit_request_id": request_id or ctx["request_id"],
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Structured pipeline logger
# ---------------------------------------------------------------------------

def _compact(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if v is not None}, separators=(",", ":"))


class CortexLogger:
    """Structured JSON logging for every Cortex pipeline stage."""

    def _emit(self, event: str, **kwargs: Any) -> None:
        _logger.info(_compact({"event": event, **kwargs}))

    def log_ingest(
        self,
        *,
        app_name: str,
        doc_id: str,
        user_id: str,
        strategy: str,
        chunk_count: int,
        embed_model: str,
        collection: str,
        chunk_latency_ms: float,
        embed_latency_ms: float,
        store_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        self._emit(
            "ingest",
            app=app_name,
            doc_id=doc_id,
            user_id=user_id,
            strategy=strategy,
            chunks=chunk_count,
            embed_model=embed_model,
            collection=collection,
            chunk_ms=round(chunk_latency_ms, 1),
            embed_ms=round(embed_latency_ms, 1),
            store_ms=round(store_latency_ms, 1),
            total_ms=round(total_latency_ms, 1),
        )

    def log_embed(
        self,
        *,
        app_name: str,
        model: str,
        count: int,
        latency_ms: float,
    ) -> None:
        self._emit(
            "embed",
            app=app_name,
            model=model,
            count=count,
            total_ms=round(latency_ms, 1),
        )

    def log_retrieve(
        self,
        *,
        app_name: str,
        user_id: str,
        query_len: int,
        chunk_count: int,
        rerank_enabled: bool,
        avg_score: Optional[float],
        avg_rerank: Optional[float],
        rerank_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        self._emit(
            "retrieve",
            app=app_name,
            user_id=user_id,
            query_len=query_len,
            chunks=chunk_count,
            rerank=rerank_enabled,
            avg_score=round(avg_score, 4) if avg_score is not None else None,
            avg_rerank=round(avg_rerank, 4) if avg_rerank is not None else None,
            rerank_ms=round(rerank_latency_ms, 1),
            total_ms=round(total_latency_ms, 1),
        )

    def log_generate(
        self,
        *,
        app_name: str,
        user_id: str,
        response_type: str,
        attempt_count: int,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        self._emit(
            "generate",
            app=app_name,
            user_id=user_id,
            response_type=response_type,
            attempts=attempt_count,
            success=success,
            error=error,
            total_ms=round(latency_ms, 1),
        )

    def log_rerank(
        self,
        *,
        app_name: str,
        model: str,
        candidates: int,
        selected: int,
        latency_ms: float,
    ) -> None:
        self._emit(
            "rerank",
            app=app_name,
            model=model,
            candidates=candidates,
            selected=selected,
            total_ms=round(latency_ms, 1),
        )


cortex_logger = CortexLogger()
