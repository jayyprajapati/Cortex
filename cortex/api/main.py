from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.applications import router as applications_router
from app.api.collections import router as collections_router
from app.llm.factory import get_llm
from app.pipeline.generate_pipeline import (
    extract_citations,
    generate_answer,
    generate_direct,
    stream_answer,
)
from app.pipeline.ingest_pipeline import ingest_document, ingest_text, resolve_doc_id
from app.registry.service import _resolve_llm_config, build_execution_context
from app.threads import (
    append_message,
    count_messages,
    create_thread,
    delete_thread,
    get_recent_messages,
    get_thread,
    get_thread_with_messages,
    list_threads,
    update_summary,
    update_title,
)
from app.threads.summarize import KEEP_RECENT, SUMMARIZE_AFTER, summarize_old_turns
from app.vectorstore.qdrant_store import delete_document_vectors, delete_user_vectors
from cortex.middleware.docs_block import BlockDocsInProduction
from cortex.middleware.origin import OriginRefererMiddleware, _ALLOWED_ORIGINS as _CORS_ORIGINS
from cortex.core.resume_extractor import extract_resume
from cortex.core.profile_normalizer import merge_profiles
from cortex.core.resume_optimizer import analyze_match, generate_document
from cortex.core.composition import generate_cover_letter, generate_hr_email, rewrite_email
from cortex.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ThreadDetailResponse,
    ThreadListResponse,
    ThreadPatchRequest,
    ThreadSummary,
)
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
    CoverLetterRequest,
    CoverLetterResponse,
    HrEmailRequest,
    HrEmailResponse,
    RewriteRequest,
    RewriteResponse,
)

_env = os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("PYTHON_ENV") or ""
_is_dev = str(_env).lower() in ("dev", "development", "local")

app = FastAPI(
    title="Cortex RAG Engine",
    description="Registry-driven multi-application RAG orchestration",
    version="2.0.0",
    **({"docs_url": None, "redoc_url": None, "openapi_url": None} if not _is_dev else {}),
)

# ---------------------------------------------------------------------------
# Security middleware stack
#
# NOTE ON HEADER-BASED SECURITY SCOPE
# Header-based checks (CORS, Origin/Referer pinning, TrustedHost) block browsers
# and casual scripted abuse, not a determined attacker with a proxy.
# Real security boundary is JWT (P0.4) + admin key (P0.3).
#
# Starlette middleware ordering: the LAST add_middleware call wraps outermost
# (processes requests first).  Stack from outermost → innermost:
#   BlockDocsInProduction → TrustedHostMiddleware → OriginRefererMiddleware → CORSMiddleware
# ---------------------------------------------------------------------------

# CORS — innermost; runs last on requests, first on responses.
# Allowlist is config-driven via CORS_ALLOWED_ORIGINS env var (comma-separated).
# If not set, falls back to the hardcoded defaults in cortex/middleware/origin.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Origin/Referer pinning — blocks browser requests from disallowed origins.
# Server-to-server requests with no Origin/Referer header are always allowed.
# Exempt paths: /health, /ready, OPTIONS (pre-flight).
app.add_middleware(OriginRefererMiddleware)

# TrustedHost — rejects requests with an unexpected Host header.
# Configured via ALLOWED_HOSTS env var (comma-separated).
# Default: localhost variants + *.jayprajapati.dev wildcard.
_raw_allowed_hosts = (os.getenv("ALLOWED_HOSTS") or "").strip()
_allowed_hosts: list[str] = (
    [h.strip() for h in _raw_allowed_hosts.split(",") if h.strip()]
    if _raw_allowed_hosts
    else ["localhost", "127.0.0.1", "0.0.0.0", "*.jayprajapati.dev"]
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)

# Docs blocker — outermost; hides /docs, /redoc, /openapi.json from non-dev hosts.
app.add_middleware(BlockDocsInProduction)


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


class ReindexRequest(BaseModel):
    app_name: str
    user_id: str
    source_dir: str
    drop_first: bool = True


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(applications_router)
app.include_router(collections_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class LLMPingRequest(BaseModel):
    llm: LLMOverride


@app.get("/")
def root():
    return {"message": "Cortex RAG Engine v2.0 — registry-driven"}


@app.post("/llm/ping")
def llm_ping_endpoint(payload: LLMPingRequest):
    """
    POST /llm/ping

    Makes a minimal one-token LLM call to verify provider connectivity.
    Used by the Settings test-connection flow — much faster than /analyze/match.
    Requires an explicit llm override; never falls back to env-configured defaults.
    """
    llm_override = {
        "provider": payload.llm.provider,
        "api_key": payload.llm.api_key,
        "model": payload.llm.model,
        "base_url": payload.llm.base_url,
    }
    try:
        llm_config = _resolve_llm_config(llm_override)
    except HTTPException:
        raise

    try:
        llm_instance = get_llm(llm_config)
        llm_instance.generate("Reply with exactly: ok", temperature=0.0)
        return {
            "ok": True,
            "provider": llm_config.provider,
            "model": llm_config.model,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc


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


@app.post("/generate")
async def generate_only_endpoint(request: Request, stream: bool = False):
    try:
        body = await request.json()
        payload = GenerateRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not stream:
        # Original non-streaming behavior
        try:
            return generate_direct(ctx, query=payload.query or "", context=composed_context)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        # SSE streaming path
        from app.streaming.sse import make_sse_response, meta_event, delta_event, done_event, error_event

        async def _gen():
            yield meta_event({"type": "generate"})
            try:
                result = generate_direct(ctx, query=payload.query or "", context=composed_context)
                answer = result.get("answer", "")
                answer_text = answer if isinstance(answer, str) else json.dumps(answer)
                yield delta_event(answer_text)
                yield done_event({"grounded": result.get("grounded", False)})
            except Exception as exc:
                yield error_event(str(exc))

        return make_sse_response(_gen())


def _auto_title(query: str) -> str:
    words = (query or "").strip().split()
    title = " ".join(words[:8])
    if len(words) > 8:
        title += "…"
    return title or "New chat"


@app.post("/chat")
async def chat_endpoint(request: Request, stream: bool = True):
    """
    POST /chat

    Streaming (default): returns SSE event stream.
    Non-streaming (?stream=false): returns JSON ChatResponse.
    """
    try:
        body = await request.json()
        payload = ChatRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    user_id = (payload.user_id or "").strip()
    app_name = (payload.app_name or "").strip().lower()
    query = (payload.query or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not app_name:
        raise HTTPException(status_code=400, detail="app_name is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    llm_override = _llm_override_from_request(payload.llm)

    # Resolve or create thread.
    # If the caller provides a thread_id that does not yet exist, create it with
    # that ID (client-managed sessions). If it does exist, verify ownership.
    if payload.thread_id:
        thread = get_thread(payload.thread_id)
        if thread is None:
            doc_ids = payload.doc_ids or []
            create_thread(
                app_name=app_name,
                user_id=user_id,
                doc_ids=doc_ids,
                title=_auto_title(query),
                thread_id=payload.thread_id,
            )
            thread = get_thread(payload.thread_id)
        else:
            if thread["user_id"] != user_id or thread["app_name"] != app_name:
                raise HTTPException(status_code=403, detail="Thread does not belong to this user")
            doc_ids = thread["doc_ids"] or payload.doc_ids
    else:
        doc_ids = payload.doc_ids or []
        thread_id_new = create_thread(
            app_name=app_name,
            user_id=user_id,
            doc_ids=doc_ids,
            title=_auto_title(query),
        )
        thread = get_thread(thread_id_new)

    # Check clarification state
    from app.conversation.state import get_clarification_pending, get_clarification_context, clear_clarification_pending
    clarification_reply = False
    if get_clarification_pending(thread):
        clarification_reply = True
        clear_clarification_pending(thread["id"])

    history = get_recent_messages(thread["id"], n=KEEP_RECENT)
    summary = thread.get("summary")
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]
    task = (payload.task or "chat").strip().lower()

    try:
        ctx = build_execution_context(
            app_name=app_name,
            user_id=user_id,
            task=task,
            doc_ids=doc_ids,
            llm_override=llm_override,
            prompt_override=payload.prompt_override,
            voice_footer=payload.voice_footer,
        )
    except HTTPException:
        raise

    if stream:
        # SSE streaming path
        from app.streaming.sse import make_sse_response, error_event

        async def _sse_generator():
            full_answer_parts = []
            citations_list = []
            event_type_received = None
            was_clarification = False
            try:
                async for event_str in stream_answer(
                    ctx=ctx,
                    query=query,
                    chat_history=chat_history,
                    summary=summary,
                    clarification_reply=clarification_reply,
                ):
                    yield event_str
                    # Parse emitted events to extract answer + citations for thread storage
                    import json as _json
                    try:
                        # SSE events are "event: TYPE\ndata: JSON\n\n"
                        for line in event_str.split("\n"):
                            if line.startswith("event: "):
                                event_type_received = line[7:].strip()
                            elif line.startswith("data: "):
                                data = _json.loads(line[6:])
                                if event_type_received == "delta":
                                    full_answer_parts.append(data.get("text", ""))
                                elif event_type_received == "clarification":
                                    full_answer_parts.append(data.get("text", ""))
                                    was_clarification = True
                                elif event_type_received == "citations":
                                    citations_list = data.get("citations", [])
                    except Exception:
                        pass
            except Exception as exc:
                yield error_event(str(exc), code="stream_error")

            # Persist thread messages after stream completes
            try:
                full_answer = "".join(full_answer_parts)
                append_message(thread["id"], "user", query)
                append_message(
                    thread["id"],
                    "assistant",
                    full_answer,
                    citations=citations_list or None,
                    grounded=False,
                )
                if was_clarification:
                    from app.conversation.state import mark_clarification_pending
                    from app.conversation.clarification import build_clarification_context
                    mark_clarification_pending(thread["id"], build_clarification_context({}, query))
                # Rolling summarization (best-effort)
                total = count_messages(thread["id"])
                summarized_up_to = int(thread.get("summary_up_to_message_idx") or 0)
                from app.conversation.history import should_summarize
                if should_summarize(total, summarized_up_to):
                    all_msgs = get_thread_with_messages(thread["id"])["messages"]
                    cutoff = total - KEEP_RECENT
                    to_summarize = all_msgs[summarized_up_to:cutoff]
                    if to_summarize:
                        try:
                            llm_inst = get_llm(ctx.llm_config)
                            new_summary = summarize_old_turns(to_summarize, summary, llm_inst)
                            if new_summary:
                                update_summary(thread["id"], new_summary, cutoff)
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Post-stream thread persistence failed: %s", exc)

        return make_sse_response(_sse_generator())

    else:
        # Non-streaming JSON path (legacy)
        try:
            result = generate_answer(
                ctx, query,
                chat_history=chat_history,
                summary=summary,
                clarification_reply=clarification_reply,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc

        answer = result.get("answer", "")
        grounded = bool(result.get("grounded", False))
        sources = result.get("sources", []) or []
        meta = result.get("meta", {}) or {}

        answer_text_for_citations = (
            answer if isinstance(answer, str)
            else json.dumps(answer) if isinstance(answer, dict)
            else str(answer)
        )
        citations = extract_citations(answer_text_for_citations, sources)

        stored_assistant_content = (
            answer if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False)
        )
        append_message(thread["id"], "user", query)
        append_message(
            thread["id"],
            "assistant",
            stored_assistant_content,
            citations=citations or None,
            grounded=grounded,
        )
        if result.get("needs_clarification"):
            from app.conversation.state import mark_clarification_pending
            from app.conversation.clarification import build_clarification_context
            mark_clarification_pending(thread["id"], build_clarification_context({}, query))

        # Summarization
        total = count_messages(thread["id"])
        summarized_up_to = int(thread.get("summary_up_to_message_idx") or 0)
        new_old_count = max(total - KEEP_RECENT - summarized_up_to, 0)
        if new_old_count >= SUMMARIZE_AFTER:
            all_msgs = get_thread_with_messages(thread["id"])["messages"]
            cutoff = total - KEEP_RECENT
            to_summarize = all_msgs[summarized_up_to:cutoff]
            if to_summarize:
                try:
                    llm_inst = get_llm(ctx.llm_config)
                    new_summary = summarize_old_turns(to_summarize, summary, llm_inst)
                    if new_summary:
                        update_summary(thread["id"], new_summary, cutoff)
                except Exception as exc:
                    logger.warning("Summary update failed: %s", exc)

        meta["history_used"] = len(chat_history)
        meta["summary_in_use"] = bool(summary)

        return ChatResponse(
            thread_id=thread["id"],
            answer=answer,
            grounded=grounded,
            citations=citations,
            sources=sources,
            meta=meta,
        )


@app.get("/threads", response_model=ThreadListResponse)
def list_threads_endpoint(user_id: str, app_name: str, limit: int = 50):
    user_id = (user_id or "").strip()
    app_name = (app_name or "").strip().lower()
    if not user_id or not app_name:
        raise HTTPException(status_code=400, detail="user_id and app_name are required")
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    items = list_threads(user_id=user_id, app_name=app_name, limit=limit)
    summaries = [
        ThreadSummary(
            id=t["id"],
            app_name=t["app_name"],
            user_id=t["user_id"],
            doc_ids=t.get("doc_ids") or [],
            title=t.get("title"),
            message_count=t.get("message_count", 0),
            created_at=t["created_at"],
            updated_at=t["updated_at"],
        )
        for t in items
    ]
    return ThreadListResponse(threads=summaries)


@app.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
def get_thread_endpoint(thread_id: str, user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    thread = get_thread_with_messages(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Thread does not belong to this user")
    return ThreadDetailResponse(**thread)


@app.delete("/threads/{thread_id}")
def delete_thread_endpoint(thread_id: str, user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    thread = get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Thread does not belong to this user")
    delete_thread(thread_id)
    return {"status": "ok", "deleted_thread_id": thread_id}


@app.patch("/threads/{thread_id}")
def patch_thread_endpoint(thread_id: str, user_id: str, payload: ThreadPatchRequest):
    user_id = (user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    thread = get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Thread does not belong to this user")
    if payload.title is not None:
        update_title(thread_id, payload.title.strip())
    return {"status": "ok", "thread_id": thread_id}


@app.post("/delete")
def delete_document_endpoint(payload: DeleteRequest):
    try:
        ctx = build_execution_context(app_name=payload.app_name, user_id=payload.user_id)
        deleted = ctx.components.vector_store.delete_by_doc(ctx.collection, payload.user_id, payload.doc_id)
    except HTTPException:
        raise
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
        deleted = ctx.components.vector_store.delete_by_user(ctx.collection, payload.user_id)
    except HTTPException:
        raise
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
        llm_json_str = str(form.get("llm") or "").strip()
        llm_from_form: Optional[LLMOverride] = None
        if llm_json_str:
            try:
                llm_from_form = LLMOverride(**json.loads(llm_json_str))
            except (ValueError, TypeError, ValidationError):
                pass

        try:
            payload = ExtractRequest(
                app_name=str(form.get("app_name") or "").strip(),
                user_id=str(form.get("user_id") or "").strip(),
                doc_id=(str(form.get("doc_id") or "").strip() or None),
                file_path=(str(form.get("file_path") or "").strip() or None),
                text=(str(form.get("text") or "").strip() or None),
                extraction_type=(str(form.get("extraction_type") or "resume").strip() or "resume"),
                llm=llm_from_form,
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

    llm_ov = _llm_override_from_request(payload.llm)

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
                llm_override=llm_ov,
            )
        elif has_path:
            result = extract_resume(
                file_path=payload.file_path,
                doc_id=payload.doc_id,
                extraction_type=payload.extraction_type,
                llm_override=llm_ov,
            )
        else:
            result = extract_resume(
                text=payload.text,
                doc_id=payload.doc_id,
                extraction_type=payload.extraction_type,
                llm_override=llm_ov,
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
            original_resume_text=payload.original_resume_text,
            user_tweak_prompt=payload.user_tweak_prompt,
            user_system_prompt=payload.user_system_prompt,
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


@app.post("/cover-letter", response_model=CoverLetterResponse)
def cover_letter_endpoint(payload: CoverLetterRequest):
    """
    POST /cover-letter

    Generate a tailored cover letter from the candidate's canonical profile.
    Grounded — no fabricated claims. Respects user style guidance from settings.
    """
    try:
        llm_ov = _llm_override_from_request(payload.llm)
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task="cover_letter",
            llm_override=llm_ov,
        )
    except HTTPException:
        raise

    gen = ctx.effective_generation

    try:
        result = generate_cover_letter(
            job_description=payload.job_description,
            canonical_profile=payload.canonical_profile,
            llm_config=ctx.llm_config,
            system_prompt=gen.system_prompt,
            analysis_summary=payload.analysis_summary,
            user_prompt=payload.user_prompt,
            user_system_prompt=payload.user_system_prompt,
            max_retries=gen.max_retries,
            temperature=gen.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CoverLetterResponse(**result)


@app.post("/hr-email", response_model=HrEmailResponse)
def hr_email_endpoint(payload: HrEmailRequest):
    """
    POST /hr-email

    Generate a structured recruiter/HR outreach email from the candidate's canonical profile.
    Returns subject + body as JSON. Grounded — no fabricated claims.
    """
    try:
        llm_ov = _llm_override_from_request(payload.llm)
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task="hr_email",
            llm_override=llm_ov,
        )
    except HTTPException:
        raise

    gen = ctx.effective_generation

    try:
        result = generate_hr_email(
            job_description=payload.job_description,
            canonical_profile=payload.canonical_profile,
            llm_config=ctx.llm_config,
            system_prompt=gen.system_prompt,
            analysis_summary=payload.analysis_summary,
            recipient_name=payload.recipient_name,
            user_prompt=payload.user_prompt,
            user_system_prompt=payload.user_system_prompt,
            max_retries=gen.max_retries,
            temperature=gen.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return HrEmailResponse(**result)


@app.post("/compose/rewrite", response_model=RewriteResponse)
def compose_rewrite_endpoint(payload: RewriteRequest):
    """
    POST /compose/rewrite

    Rewrite an email according to a user instruction (make concise, more formal, etc.).
    Preserves facts; alters only style, length, and tone.
    Returns rewritten content as HTML.
    """
    try:
        llm_ov = _llm_override_from_request(payload.llm)
        ctx = build_execution_context(
            app_name=payload.app_name,
            user_id=payload.user_id,
            task="compose_rewrite",
            llm_override=llm_ov,
        )
    except HTTPException:
        raise

    gen = ctx.effective_generation

    try:
        result = rewrite_email(
            instruction=payload.instruction,
            llm_config=ctx.llm_config,
            system_prompt=gen.system_prompt,
            body_html=payload.body_html,
            body_text=payload.body_text,
            subject=payload.subject,
            user_system_prompt=payload.user_system_prompt,
            max_retries=gen.max_retries,
            temperature=gen.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=_llm_status(exc), detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RewriteResponse(**result)


@app.post("/admin/reindex")
def admin_reindex_endpoint(payload: ReindexRequest):
    """
    POST /admin/reindex

    Drop the app's collection and re-ingest all supported files from source_dir.
    Use drop_first=true (default) for a clean reindex.
    """
    try:
        from app.admin.reindex import reindex_app
        result = reindex_app(
            app_name=payload.app_name,
            user_id=payload.user_id,
            source_dir=payload.source_dir,
            drop_first=payload.drop_first,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result
