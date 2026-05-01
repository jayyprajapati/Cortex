from __future__ import annotations

import os
import tempfile
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app.api.applications import router as applications_router
from app.api.collections import router as collections_router
from app.pipeline.generate_pipeline import generate_answer, generate_direct
from app.pipeline.ingest_pipeline import ingest_document, ingest_text, resolve_doc_id
from app.registry.service import build_execution_context
from app.vectorstore.qdrant_store import delete_document_vectors, delete_user_vectors

app = FastAPI(
    title="Cortex RAG Engine",
    description="Registry-driven multi-application RAG orchestration",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(v) for v in value)
    return value


@app.exception_handler(RequestValidationError)
async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LLMOptions(BaseModel):
    provider: Literal["openai", "ollama_local", "ollama_cloud", "ollama"] = "ollama_cloud"
    api_key: Optional[str] = None
    model: Optional[str] = None


class QueryRequest(BaseModel):
    app_name: str
    user_id: str
    query: str
    task: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    llm: Optional[LLMOptions] = None
    prompt_override: Optional[str] = None


class IngestRequest(BaseModel):
    app_name: str
    user_id: str
    doc_id: Optional[str] = None
    file_path: Optional[str] = None
    text: Optional[str] = None


class GenerateRequest(BaseModel):
    app_name: str
    user_id: str
    query: Optional[str] = None
    task: Optional[str] = None
    context: Optional[str] = None
    input: Optional[Any] = None
    llm: Optional[LLMOptions] = None
    prompt_override: Optional[str] = None


class DeleteRequest(BaseModel):
    app_name: str
    user_id: str
    doc_id: str


class DeleteAllRequest(BaseModel):
    app_name: str
    user_id: str


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(applications_router)
app.include_router(collections_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Cortex RAG Engine v2.0 — registry-driven"}


@app.post("/ingest")
async def ingest_endpoint(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    uploaded_file = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = IngestRequest(**body)
        except (ValidationError, Exception) as exc:
            raise HTTPException(status_code=422, detail=_sanitize(str(exc))) from exc

    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to parse multipart form-data.",
            ) from exc

        uploaded_file = form.get("file")
        try:
            payload = IngestRequest(
                app_name=str(form.get("app_name") or "").strip(),
                user_id=str(form.get("user_id") or "").strip(),
                doc_id=(str(form.get("doc_id") or "").strip() or None),
                file_path=(str(form.get("file_path") or "").strip() or None),
                text=(str(form.get("text") or "").strip() or None),
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_sanitize(exc.errors())) from exc
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type. Use application/json or multipart/form-data.",
        )

    has_path = bool(payload.file_path)
    has_upload = uploaded_file is not None and hasattr(uploaded_file, "read")
    has_text = bool(payload.text)
    sources = sum([has_path, has_upload, has_text])

    if sources != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: file_path, file (upload), or text.",
        )

    try:
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
        )
    except HTTPException:
        raise

    doc_id = resolve_doc_id(payload.doc_id)
    temp_path = None

    try:
        if has_upload:
            file_bytes = await uploaded_file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            suffix = os.path.splitext(uploaded_file.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name
            result = ingest_document(ctx, temp_path, doc_id)

        elif has_path:
            result = ingest_document(ctx, payload.file_path, doc_id)

        else:
            result = ingest_text(ctx, payload.text, doc_id)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {"status": "success", **result}


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    try:
        llm_override = None
        if payload.llm:
            llm_override = {
                "provider": payload.llm.provider,
                "model": payload.llm.model,
                "api_key": payload.llm.api_key,
            }

        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task=payload.task,
            doc_ids=payload.doc_ids,
            llm_override=llm_override,
            prompt_override=payload.prompt_override,
        )
        return generate_answer(ctx, payload.query)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate")
def generate_only_endpoint(payload: GenerateRequest):
    try:
        llm_override = None
        if payload.llm:
            llm_override = {
                "provider": payload.llm.provider,
                "model": payload.llm.model,
                "api_key": payload.llm.api_key,
            }

        composed_context = ""
        if payload.context:
            composed_context += str(payload.context).strip()
        if payload.input is not None:
            if composed_context:
                composed_context += "\n\n"
            composed_context += f"Task input:\n{_sanitize(payload.input)}"

        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task=payload.task,
            llm_override=llm_override,
            prompt_override=payload.prompt_override,
        )
        return generate_direct(ctx, query=payload.query or "", context=composed_context)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/delete")
def delete_document_endpoint(payload: DeleteRequest):
    try:
        ctx = build_execution_context(app_name=payload.app_name, user_id=payload.user_id)
        deleted = delete_document_vectors(ctx.collection, payload.user_id, payload.doc_id)
    except HTTPException:
        raise
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "No matching vectors" in detail else 400
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "deleted_points": deleted,
        "app_name": payload.app_name,
        "user_id": payload.user_id,
        "doc_id": payload.doc_id,
    }


@app.post("/delete_all")
def delete_all_documents_endpoint(payload: DeleteAllRequest):
    try:
        ctx = build_execution_context(app_name=payload.app_name, user_id=payload.user_id)
        deleted = delete_user_vectors(ctx.collection, payload.user_id)
    except HTTPException:
        raise
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "No matching vectors" in detail else 400
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "deleted_points": deleted,
        "app_name": payload.app_name,
        "user_id": payload.user_id,
    }
