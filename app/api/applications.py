from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.registry.store import delete_app, get_app, list_apps, register_app, update_app

router = APIRouter(prefix="/apps", tags=["applications"])


class AppRegisterRequest(BaseModel):
    """Full pipeline contract required for registration."""

    app_name: str
    collection: str
    ingestion: Dict[str, Any]
    embedding: Dict[str, Any]
    retrieval: Dict[str, Any]
    reranking: Dict[str, Any]
    generation: Dict[str, Any]
    defaults: Dict[str, Any]
    tasks: Dict[str, Any] = {}
    default_task: str | None = None


class AppUpdateRequest(BaseModel):
    """Full pipeline contract required for update (app_name taken from URL)."""

    collection: str
    ingestion: Dict[str, Any]
    embedding: Dict[str, Any]
    retrieval: Dict[str, Any]
    reranking: Dict[str, Any]
    generation: Dict[str, Any]
    defaults: Dict[str, Any]
    tasks: Dict[str, Any] = {}
    default_task: str | None = None


@router.post("/register", status_code=201)
def register_application(payload: AppRegisterRequest):
    config_payload = payload.model_dump(exclude={"app_name"})
    try:
        registered = register_app(payload.app_name, config_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "success": True,
        "message": "Application registered",
        "application": registered.model_dump(by_alias=True),
    }


@router.get("")
def list_applications():
    return {"applications": list_apps()}


@router.get("/{app_name}")
def get_application(app_name: str):
    config = get_app(app_name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown application: {app_name!r}")
    return config.model_dump(by_alias=True)


@router.put("/{app_name}")
def update_application(app_name: str, payload: AppUpdateRequest):
    config_payload = payload.model_dump()
    try:
        updated = update_app(app_name, config_payload)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "Unknown application" in detail else 400
        raise HTTPException(status_code=status, detail=detail) from exc

    return {
        "success": True,
        "message": "Application updated",
        "application": updated.model_dump(by_alias=True),
    }


@router.delete("/{app_name}")
def delete_application(app_name: str):
    try:
        delete_app(app_name)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "Unknown application" in detail else 400
        raise HTTPException(status_code=status, detail=detail) from exc

    return {"success": True, "message": "Application deleted", "app_name": app_name}
