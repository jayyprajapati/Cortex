import os
import tempfile
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app.llm.factory import get_llm
from app.pipeline.generate_pipeline import generate_answer, resolve_llm_config
from app.pipeline.ingest_pipeline import ingest_document, ingest_text, resolve_doc_id
from app.vectorstore.qdrant_store import delete_document_vectors, delete_user_vectors

app = FastAPI(title="Cortex RAG Engine")


def _sanitize_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(value)} bytes>"

    if isinstance(value, dict):
        return {key: _sanitize_value(inner) for key, inner in value.items()}

    if isinstance(value, list):
        return [_sanitize_value(inner) for inner in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_value(inner) for inner in value)

    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    # FastAPI's default encoder can crash if invalid requests include raw binary payloads.
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_value(exc.errors())},
    )


class LLMOptions(BaseModel):
    provider: Literal["openai", "ollama"] = "ollama"
    api_key: Optional[str] = None
    model: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    user_id: str
    app_name: str = "default"
    doc_id: Optional[str] = None
    llm: Optional[LLMOptions] = None


class IngestRequest(BaseModel):
    user_id: str
    doc_id: Optional[str] = None
    file_path: Optional[str] = None
    text: Optional[str] = None
    llm: Optional[LLMOptions] = None


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


class DeleteRequest(BaseModel):
    user_id: str
    doc_id: str


class DeleteAllRequest(BaseModel):
    user_id: str


def _to_llm_config(llm_options: Optional[LLMOptions]):
    if not llm_options:
        return None

    config = {
        "provider": llm_options.provider,
    }

    api_key = (llm_options.api_key or "").strip()
    if api_key:
        config["api_key"] = api_key

    if llm_options.model:
        config["model"] = llm_options.model

    return config


def _build_llm_options(provider=None, api_key=None, model=None):
    if provider is None and api_key is None and model is None:
        return None

    normalized_provider = (provider or "ollama").strip().lower()

    try:
        return LLMOptions(
            provider=normalized_provider,
            api_key=api_key,
            model=model,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def root():
    return {"message": "RAG engine running"}


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    try:
        return generate_answer(
            query=payload.query,
            user_id=payload.user_id,
            app_name=payload.app_name,
            doc_id=payload.doc_id,
            llm_config=_to_llm_config(payload.llm),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest")
async def ingest_endpoint(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()

    uploaded_file = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = IngestRequest(**body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_sanitize_value(exc.errors())) from exc
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to parse multipart form-data. Ensure python-multipart is installed.",
            ) from exc

        uploaded_file = form.get("file")

        payload = IngestRequest(
            user_id=str(form.get("user_id") or "").strip(),
            doc_id=(str(form.get("doc_id") or "").strip() or None),
            file_path=(str(form.get("file_path") or "").strip() or None),
            text=(str(form.get("text") or "").strip() or None),
            llm=_build_llm_options(
                provider=form.get("llm_provider"),
                api_key=form.get("llm_api_key"),
                model=form.get("llm_model"),
            ),
        )
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported content-type. Use application/json or multipart/form-data.",
        )

    has_path_file = bool(payload.file_path)
    has_uploaded_file = uploaded_file is not None and hasattr(uploaded_file, "read")
    has_text = bool(payload.text)

    if (1 if has_path_file else 0) + (1 if has_uploaded_file else 0) + (1 if has_text else 0) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of file_path, file, or text",
        )

    temp_path = None
    generated_doc_id = resolve_doc_id(payload.doc_id)

    try:
        if has_uploaded_file:
            file_bytes = await uploaded_file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            suffix = os.path.splitext(uploaded_file.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name

            ingest_document(
                path=temp_path,
                doc_id=generated_doc_id,
                user_id=payload.user_id,
                api_key=(payload.llm.api_key if payload.llm else None),
            )
        elif has_path_file:
            ingest_document(
                path=payload.file_path,
                doc_id=generated_doc_id,
                user_id=payload.user_id,
                api_key=(payload.llm.api_key if payload.llm else None),
            )
        else:
            ingest_text(
                text=payload.text,
                doc_id=generated_doc_id,
                user_id=payload.user_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": "success",
        "doc_id": generated_doc_id,
    }


@app.post("/generate")
def generate_only_endpoint(payload: GenerateRequest):
    try:
        llm_config = dict(resolve_llm_config())

        if payload.model:
            llm_config["model"] = payload.model

        llm = get_llm(llm_config)
        answer = llm.generate(payload.prompt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"answer": answer}


@app.post("/delete")
def delete_document_endpoint(payload: DeleteRequest):
    try:
        deleted_points = delete_document_vectors(
            user_id=payload.user_id,
            doc_id=payload.doc_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if "No matching vectors found" in detail:
            raise HTTPException(status_code=404, detail=detail) from exc

        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "deleted_points": deleted_points,
        "user_id": payload.user_id,
        "doc_id": payload.doc_id,
    }


@app.post("/delete_all")
def delete_all_documents_endpoint(payload: DeleteAllRequest):
    try:
        deleted_points = delete_user_vectors(
            user_id=payload.user_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if "No matching vectors found" in detail:
            raise HTTPException(status_code=404, detail=detail) from exc

        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "deleted_points": deleted_points,
        "user_id": payload.user_id,
    }
