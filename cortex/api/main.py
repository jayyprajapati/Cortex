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
from cortex.core.resume_extractor import extract_resume
from cortex.core.profile_normalizer import merge_profiles
from cortex.core.resume_optimizer import analyze_match, generate_document
from cortex.schemas.resumelab import (
    ExtractRequest,
    ExtractResponse,
    ProfileMergeRequest,
    ProfileMergeResponse,
    CanonicalProfile,
    MatchRequest,
    MatchResponse,
    DocumentRequest,
    DocumentResponse,
    LLMOverride,
)

_env = os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("PYTHON_ENV") or "development"
_is_prod = str(_env).lower() in ("production", "prod")

app = FastAPI(
    title="Cortex RAG Engine",
    description="Registry-driven multi-application RAG orchestration",
    version="2.0.0",
    **({"docs_url": None, "redoc_url": None, "openapi_url": None} if _is_prod else {}),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_LLM_ERROR_MARKERS = (
    "llm provider", "llm extraction failed", "llm generation failed",
    "extraction failed", "generation failed",
)


def _llm_override_from_request(llm_field) -> Optional[dict]:
    """Convert an optional LLMOverride schema object into the dict expected by build_execution_context."""
    if llm_field is None:
        return None
    return {
        "provider": llm_field.provider,
        "api_key": llm_field.api_key,
        "model": llm_field.model,
        "base_url": llm_field.base_url,
    }


def _llm_status(exc: Exception) -> int:
    """Return 502 when the failure is from the LLM provider, 400 for input errors."""
    msg = str(exc).lower()
    return 502 if any(m in msg for m in _LLM_ERROR_MARKERS) else 400


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


@app.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(request: Request):
    """
    POST /extract

    Structured extraction from a resume, profile document, or raw text.
    Accepts application/json (with text or file_path) or multipart/form-data
    (with a file upload).

    Input fields:
      app_name        str         Required. Must be a registered Cortex app.
      user_id         str         Required.
      doc_id          str         Optional. Auto-generated if omitted.
      file            upload      Multipart only. PDF, DOCX, or text file.
      file_path       str         JSON only. Server-side path to the document.
      text            str         Raw document text.
      extraction_type str         "resume" | "generic_profile" | "structured_doc"
                                  Default: "resume"

    Exactly one of file, file_path, or text must be provided.

    Output: ExtractResponse (see cortex/schemas/resumelab.py)
    """
    content_type = (request.headers.get("content-type") or "").lower()
    uploaded_file = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = ExtractRequest(**body)
        except (ValidationError, Exception) as exc:
            raise HTTPException(status_code=422, detail=_sanitize(str(exc))) from exc

    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Unable to parse multipart form-data.") from exc

        uploaded_file = form.get("file")
        try:
            payload = ExtractRequest(
                app_name=str(form.get("app_name") or "").strip(),
                user_id=str(form.get("user_id") or "").strip(),
                doc_id=(str(form.get("doc_id") or "").strip() or None),
                file_path=(str(form.get("file_path") or "").strip() or None),
                text=(str(form.get("text") or "").strip() or None),
                extraction_type=(str(form.get("extraction_type") or "resume").strip() or "resume"),
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

    # Validate app exists (fails fast with 404 for unknown app)
    try:
        build_execution_context(app_name=payload.app_name, user_id=payload.user_id)
    except HTTPException:
        raise

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
            result = extract_resume(
                file_path=temp_path,
                doc_id=payload.doc_id,
                extraction_type=payload.extraction_type,
            )
        elif has_path:
            result = extract_resume(
                file_path=payload.file_path,
                doc_id=payload.doc_id,
                extraction_type=payload.extraction_type,
            )
        else:
            result = extract_resume(
                text=payload.text,
                doc_id=payload.doc_id,
                extraction_type=payload.extraction_type,
            )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    try:
        return ExtractResponse(**result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction produced invalid response shape: {_sanitize(exc.errors())}",
        ) from exc


@app.post("/profile/merge", response_model=ProfileMergeResponse)
def profile_merge_endpoint(payload: ProfileMergeRequest):
    """
    POST /profile/merge

    Merge two profile dicts into a deduplicated CanonicalProfile.

    Input:
      app_name              str     Required. Must be a registered Cortex app.
      user_id               str     Required.
      existing_profile      dict    Current profile (ExtractResponse or CanonicalProfile shape).
      incoming_profile      dict    New profile to merge in.
      similarity_threshold  float   Optional. Cosine similarity threshold for fuzzy dedup.
                                    Default: 0.85. Range: 0.0–1.0.

    Output: ProfileMergeResponse
      canonical_profile     CanonicalProfile  Merged, deduplicated profile.
      added_items           dict              Items from incoming not present in existing.
      merged_duplicates     dict              Items that were semantically merged.
      conflicts             dict              Items with irreconcilable field disagreements.
      stats                 dict              Before/after counts per section.

    Behavior:
      - Skill names normalized via alias map (React.js → React, AWS variants → AWS, etc.)
      - Exact canonical_key dedup first, embedding similarity fallback second
      - Richer content wins: longer string, higher proficiency rank, union of list fields
      - Bullet dedup for experience: semantic similarity > 0.92 considered duplicate
      - Conflicts surfaced for experience date disagreements (non-blocking)
      - Source doc_id traceability preserved on every canonical item
      - Existing Cortex apps (doclens, cvscan) are unaffected
    """
    try:
        build_execution_context(app_name=payload.app_name, user_id=payload.user_id)
    except HTTPException:
        raise

    try:
        result = merge_profiles(
            existing_profile=payload.existing_profile,
            incoming_profile=payload.incoming_profile,
            threshold=payload.similarity_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        canonical = CanonicalProfile(**result["canonical_profile"])
        return ProfileMergeResponse(
            canonical_profile=canonical,
            added_items=result["added_items"],
            merged_duplicates=result["merged_duplicates"],
            conflicts=result["conflicts"],
            stats=result["stats"],
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Merge produced invalid response shape: {_sanitize(exc.errors())}",
        ) from exc


@app.post("/analyze/match", response_model=MatchResponse)
def analyze_match_endpoint(payload: MatchRequest):
    """
    POST /analyze/match

    Compare a job description against a canonical profile and an optional
    current resume.  Returns a structured gap analysis with the critical
    A/B distinction:

      missing_keywords               — skills the candidate does NOT have at all
      existing_but_missing_from_resume — skills in the canonical profile that
                                         were omitted from the current resume
                                         (immediate optimization wins)

    Input:
      app_name           str   Required. Registered Cortex app (e.g. "resumelab").
      user_id            str   Required.
      job_description    str   Required. Raw JD text.
      canonical_profile  dict  Required. Phase 2 CanonicalProfile or ExtractResponse shape.
      base_resume        dict  Optional. The user's current resume submission.

    Output: MatchResponse
      match_score                     0-100 ATS fit score
      required_keywords               All JD keywords extracted
      missing_keywords                Candidate lacks entirely
      existing_but_missing_from_resume Candidate has, but didn't include in resume
      irrelevant_content              Items to remove from resume
      recommended_additions           Items from profile to add
      recommended_removals            Items from resume to cut
      section_rewrites                {summary, skills, projects}
      ats_keyword_clusters            Keywords grouped by theme
      role_seniority                  junior|mid|senior|lead|principal|executive
      domain_fit                      One-sentence domain alignment assessment
    """
    try:
        llm_ov = _llm_override_from_request(payload.llm)
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task="match",
            llm_override=llm_ov,
        )
    except HTTPException:
        raise

    gen = ctx.effective_generation

    try:
        result = analyze_match(
            job_description=payload.job_description,
            canonical_profile=payload.canonical_profile,
            base_resume=payload.base_resume,
            llm_config=ctx.llm_config,
            system_prompt=gen.system_prompt,
            schema=gen.schema or {},
            max_retries=gen.max_retries,
            temperature=gen.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return MatchResponse(**result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Match analysis produced invalid response shape: {_sanitize(exc.errors())}",
        ) from exc


@app.post("/generate/document", response_model=DocumentResponse)
def generate_document_endpoint(payload: DocumentRequest):
    """
    POST /generate/document

    Generate structured, ATS-optimized resume content blocks from a canonical
    profile, targeted at a specific job description and template type.

    TRUTHFUL ONLY — no fabricated skills, experience, metrics, or dates.
    Every output claim is grounded in the canonical profile provided.

    Input:
      app_name           str   Required. Registered Cortex app.
      user_id            str   Required.
      job_description    str   Required. Raw JD text.
      canonical_profile  dict  Required. Phase 2 CanonicalProfile dict.
      base_resume        dict  Optional. Current resume (used for omission detection).
      template_type      str   "frontend" | "backend" | "fullstack" (default: fullstack)

    Output: DocumentResponse
      summary              2-3 sentence professional summary with JD keywords
      skills               Ordered flat list of skill strings, most relevant first
      projects             Array of relevant project objects
      experience           Array of experience objects with JD-relevant bullets
      target_keywords_used JD keywords incorporated into the document
      removed_content      Items excluded (with brief reason)
      match_score_improved Estimated ATS score for the generated document (0-100)

    Integration notes (ReachFlow and downstream apps):
      - Inject summary, skills, experience, projects directly into template slots
      - target_keywords_used can drive keyword-density validation
      - removed_content is useful for UI diff display
      - This endpoint does NOT write to Qdrant; call /ingest separately if desired
    """
    try:
        llm_ov = _llm_override_from_request(payload.llm)
        task = "modify_existing" if payload.mode == "modify_existing" else "generate"
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task=task,
            llm_override=llm_ov,
        )
    except HTTPException:
        raise

    gen = ctx.effective_generation

    try:
        result = generate_document(
            job_description=payload.job_description,
            canonical_profile=payload.canonical_profile,
            base_resume=payload.base_resume,
            template_type=payload.template_type,
            llm_config=ctx.llm_config,
            system_prompt=gen.system_prompt,
            schema=gen.schema or {},
            max_retries=gen.max_retries,
            temperature=gen.temperature,
            mode=payload.mode,
            source_resume_content=payload.source_resume_content,
            user_tweak_prompt=payload.user_tweak_prompt,
            include_missing_profile_keywords=payload.include_missing_profile_keywords,
            include_external_keywords=payload.include_external_keywords,
            remove_irrelevant_keywords=payload.remove_irrelevant_keywords,
            aggressiveness=payload.aggressiveness,
        )
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return DocumentResponse(**result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document generation produced invalid response shape: {_sanitize(exc.errors())}",
        ) from exc
