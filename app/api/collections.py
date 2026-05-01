from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.vectorstore.qdrant_store import (
    create_collection,
    delete_collection,
    get_collection_info,
    list_collections,
)

router = APIRouter(prefix="/collections", tags=["collections"])

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


class CreateCollectionRequest(BaseModel):
    name: str
    vector_size: int

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError(
                "Collection name must start with a lowercase letter, "
                "contain only [a-z0-9_-], and be at most 63 characters"
            )
        return v

    @field_validator("vector_size")
    @classmethod
    def _vector_size(cls, v: int) -> int:
        if not 1 <= v <= 65536:
            raise ValueError("vector_size must be between 1 and 65536")
        return v


@router.get("")
def list_all_collections():
    try:
        return {"collections": list_collections()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{collection_name}")
def get_collection(collection_name: str):
    try:
        return get_collection_info(collection_name)
    except Exception as exc:
        msg = str(exc).lower()
        status = 404 if ("not found" in msg or "doesn't exist" in msg or "does not exist" in msg) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("")
def create_new_collection(payload: CreateCollectionRequest):
    try:
        create_collection(payload.name, payload.vector_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": True, "name": payload.name, "vector_size": payload.vector_size}


@router.delete("/{collection_name}")
def delete_existing_collection(collection_name: str):
    try:
        delete_collection(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "name": collection_name}
