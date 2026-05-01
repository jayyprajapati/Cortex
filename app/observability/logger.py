from __future__ import annotations

import json
import logging
from typing import Any, Optional

_logger = logging.getLogger("cortex")


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
