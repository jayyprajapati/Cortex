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


def get_app(app_name: str) -> Optional[ApplicationConfig]:
    name = _normalize_name(app_name)
    if not name:
        return None
    raw = _load_raw()
    payload = raw.get(name)
    if payload is None:
        return None
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
