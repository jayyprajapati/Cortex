from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm.factory import get_llm
from app.pipeline.generate_pipeline import generate_answer, resolve_llm_config
from app.pipeline.ingest_pipeline import ingest_document, ingest_text

app = FastAPI(title="Cortex RAG Engine")


class QueryRequest(BaseModel):
    query: str
    user_id: str
    app_name: str
    doc_id: Optional[str] = None


class IngestRequest(BaseModel):
    user_id: str
    doc_id: str
    file_path: Optional[str] = None
    text: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


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
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest")
def ingest_endpoint(payload: IngestRequest):
    has_file = bool(payload.file_path)
    has_text = bool(payload.text)

    if has_file == has_text:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of file_path or text",
        )

    try:
        if has_file:
            chunks = ingest_document(
                path=payload.file_path,
                doc_id=payload.doc_id,
                user_id=payload.user_id,
            )
        else:
            chunks = ingest_text(
                text=payload.text,
                doc_id=payload.doc_id,
                user_id=payload.user_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "chunks_stored": len(chunks),
        "doc_id": payload.doc_id,
        "user_id": payload.user_id,
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
