# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run server (dev)
uvicorn main:app --reload
# Equivalent (main.py is just: from cortex.api.main import app)
uvicorn cortex.api.main:app --reload

# Run smoke-test scripts (from repo root with venv active)
python3 -m scripts.test_qdrant
python3 -m scripts.test_ingest
python3 -m scripts.test_retrieve
python3 -m scripts.test_generate
python3 -m scripts.test_resumelab_extract
python3 -m scripts.test_resumelab_match
python3 -m scripts.test_resumelab_generate
python3 -m scripts.test_resumelab_merge

# Frontend admin UI (React + Vite)
cd frontend && npm install && npm run dev
```

Scripts must be run as modules (`python3 -m scripts.foo`), not as files.

## Architecture

Cortex has two distinct subsystems under one FastAPI app (`cortex/api/main.py`):

### 1. Generic RAG Engine (`app/`)

Registry-driven pipeline for any downstream app (DocLens, CVScan, etc.). Every client app registers its config once; Cortex resolves all behaviour at runtime per request.

**Request lifecycle for `/query`:**
```
Request → build_execution_context()
        → retrieve_and_rerank(ctx, query)   # hybrid Qdrant search + cross-encoder rerank
        → generate_answer(ctx, chunks)      # LLM with resolved config
        → Response
```

**Key files:**
- `app/context.py` — `ExecutionContext` and `LLMConfig` dataclasses. The `ExecutionContext` is built once per request and passed through every pipeline stage. Pipeline code **never** reads config or env vars directly.
- `app/registry/service.py` — `build_execution_context()` and `_resolve_llm_config()`. LLM resolution order: request body `llm` field → env vars → `ollama_local` fallback.
- `app/registry/models.py` — Pydantic models for `ApplicationConfig` (ingestion, embedding, retrieval, reranking, generation, tasks).
- `app/registry/registry.json` — persisted app registrations (pre-loaded: `doclens`, `cvscan`).
- `app/pipeline/ingest_pipeline.py` — load → chunk → embed → Qdrant upsert.
- `app/pipeline/retrieve_pipeline.py` — embed query → hybrid search → rerank.
- `app/pipeline/generate_pipeline.py` — LLM call with retry, JSON extraction, schema validation.
- `app/config.py` — Qdrant client (cached via `@lru_cache`) and LLM env-var defaults.

**Ingestion strategies** (in `app/chunking/strategies/`): `semantic_doc`, `resume_structured`, `markdown_aware`, `resume_canonical`. Strategy is set per registered app.

### 2. ResumeLab (`cortex/`)

Domain-specific resume intelligence layer. No Qdrant reads/writes — purely LLM-driven.

**Endpoints** (all in `cortex/api/main.py`):
- `POST /extract` — parse a resume file/text into structured `ExtractResponse`
- `POST /profile/merge` — deduplicate two `CanonicalProfile`s via cosine similarity
- `POST /analyze/match` — score a JD against a profile; returns missing/omitted keywords
- `POST /generate/document` — produce ATS-optimised resume content blocks

**Key files:**
- `cortex/core/resume_extractor.py` — regex section detection + LLM extraction with cache
- `cortex/core/resume_optimizer.py` — `analyze_match()` and `generate_document()`. Pre-computes keyword sets deterministically (no LLM for A/B split), then calls LLM once.
- `cortex/core/profile_normalizer.py` — `merge_profiles()` with cosine-similarity deduplication
- `cortex/schemas/resumelab.py` — all Pydantic schemas for ResumeLab requests/responses

**Generation modes** (`generate_document`): `canonical_only` (build fresh from profile) or `modify_existing` (preserve structure, targeted rewrites). Template types: `frontend`, `backend`, `fullstack`, `custom` — each has different emphasis guidance injected into the prompt.

### LLM providers

Three providers via `app/llm/factory.py`:
- `ollama_local` — no key required, `base_url` optional
- `ollama_cloud` — requires `OLLAMA_CLOUD_API_KEY`
- `openai` — requires `OPENAI_API_KEY`

Auth errors from any provider propagate as HTTP 401 in `cortex/api/main.py`.

### Frontend (`frontend/`)

React + TypeScript + Tailwind + Vite admin debug UI. Four tabs: App Registry, Config Editor, Collections, Debug Console. Not part of the main service; purely for development introspection.

## Environment Variables

```env
# Qdrant (required)
QDRANT_MODE=local          # "cloud" or "local"
QDRANT_HOST=localhost      # local mode
QDRANT_PORT=6333           # local mode
QDRANT_URL=                # cloud mode
QDRANT_API_KEY=            # cloud mode

# LLM defaults
LLM_PROVIDER=ollama_cloud  # openai | ollama_cloud | ollama_local
LLM_MODEL=                 # leave blank for provider defaults
OPENAI_API_KEY=
OLLAMA_CLOUD_API_KEY=
OLLAMA_TIMEOUT=600         # seconds; local models on large prompts can be slow

# Optional
CORTEX_APP_REGISTRY_PATH=  # default: app/registry/registry.json
APP_ENV=development        # "production" disables /docs, /redoc, /openapi.json
```

## Key Invariants

- `ExecutionContext` is constructed **only** by `build_execution_context()` in `app/registry/service.py`. Pipeline modules receive it as a parameter and never instantiate it.
- ResumeLab functions are truthfulness-constrained: they must never fabricate skills, companies, dates, or metrics not present in the input profile.
- The `analyze_match` A/B keyword split (missing vs. omitted) is deterministic — computed before any LLM call — to reduce token usage and make results reproducible.
- Docs UI is disabled when `APP_ENV=production` (or `ENV`/`PYTHON_ENV`).
