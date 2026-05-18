from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from app.registry.models import ApplicationConfig

_lock = threading.Lock()
DEFAULT_REGISTRY_FILENAME = "registry.json"


def _registry_path() -> Path:
    configured = os.getenv("CORTEX_APP_REGISTRY_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent / DEFAULT_REGISTRY_FILENAME


def _load_raw() -> Dict[str, dict]:
    path = _registry_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_raw(registry: Dict[str, dict]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
        fh.write("\n")


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def list_apps() -> List[str]:
    return sorted(_load_raw().keys())


def _migrate_legacy_app(name: str, raw_app: dict) -> dict:
    """One-shot transform of old ingestion → chunking + loader + vector_store + conversation fields."""
    import logging
    _log = logging.getLogger(__name__)

    if "ingestion" not in raw_app:
        return raw_app  # already new shape

    _log.warning("Migrating legacy registry shape for app '%s' (ingestion → chunking)", name)
    app = dict(raw_app)

    ingestion = app.pop("ingestion")
    # Remap ingestion fields to chunking
    app["chunking"] = {
        "strategy": ingestion.get("strategy", "layout_aware_semantic"),
        "max_tokens": ingestion.get("max_tokens", 512),
        "min_tokens": ingestion.get("min_tokens", 128),
        "keep_tables_atomic": True,
        "keep_code_atomic": True,
    }

    # Add defaults for new required sections
    if "loader" not in app:
        app["loader"] = {"provider": "composite"}
    if "vector_store" not in app:
        app["vector_store"] = {"provider": "qdrant", "distance": "cosine"}
    if "conversation" not in app:
        app["conversation"] = {"clarification_policy": "balanced"}

    # Remap embedding: add provider field if missing
    if "embedding" in app and "provider" not in app["embedding"]:
        app["embedding"]["provider"] = "sentence_transformers"

    # Remap retrieval: hybrid→fusion field
    if "retrieval" in app:
        ret = app["retrieval"]
        if "hybrid" in ret:
            ret.pop("hybrid")  # remove old hybrid bool
        if "fusion" not in ret:
            ret["fusion"] = "rrf"
        if "expand_neighbors" not in ret:
            ret["expand_neighbors"] = True
        if "query_rewrite" not in ret:
            ret["query_rewrite"] = True

    # Remap reranking: add provider field if missing
    if "reranking" in app and "provider" not in app["reranking"]:
        app["reranking"]["provider"] = "sentence_transformers"

    return app


def get_app(app_name: str) -> Optional[ApplicationConfig]:
    name = _normalize_name(app_name)
    if not name:
        return None
    raw = _load_raw()
    payload = raw.get(name)
    if payload is None:
        return None
    payload = _migrate_legacy_app(name, payload)
    return ApplicationConfig.model_validate({"app_name": name, **payload})


def register_app(app_name: str, config_payload: dict) -> ApplicationConfig:
    name = _normalize_name(app_name)
    if not name:
        raise ValueError("app_name is required")

    with _lock:
        raw = _load_raw()
        if name in raw:
            raise ValueError(f"Application '{name}' already exists")

        validated = ApplicationConfig.model_validate({"app_name": name, **config_payload})
        raw[name] = _to_storable(validated)
        _save_raw(raw)

    return validated


def update_app(app_name: str, config_payload: dict) -> ApplicationConfig:
    name = _normalize_name(app_name)
    if not name:
        raise ValueError("app_name is required")

    with _lock:
        raw = _load_raw()
        if name not in raw:
            raise ValueError(f"Unknown application: {name}")

        validated = ApplicationConfig.model_validate({"app_name": name, **config_payload})
        raw[name] = _to_storable(validated)
        _save_raw(raw)

    return validated


def delete_app(app_name: str) -> None:
    name = _normalize_name(app_name)
    if not name:
        raise ValueError("app_name is required")

    with _lock:
        raw = _load_raw()
        if name not in raw:
            raise ValueError(f"Unknown application: {name}")
        del raw[name]
        _save_raw(raw)


def _to_storable(config: ApplicationConfig) -> dict:
    """Serialize ApplicationConfig to registry-safe dict (excludes app_name key)."""
    data = config.model_dump(by_alias=True, exclude_none=True)
    data.pop("app_name", None)
    # Ensure tasks serialize schema alias correctly
    return data
