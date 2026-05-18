# Cortex — Registry-Driven RAG Engine

## Overview

Cortex is a standalone RAG + LLM backend service shared across multiple personal projects as a single AI core. Instead of rebuilding ingestion, retrieval, and LLM wiring in every app, Cortex exposes one stable API that any downstream client (DocLens, ReachFlow, ResumeLab, etc.) can call with its own `app_name`.

Every application is registered in a central registry that defines its ingestion strategy, embedding model, retrieval parameters, reranking settings, and generation config. Cortex resolves all of this at runtime per request — no code changes needed to add a new app.

**Version:** 2.0 — registry-driven, multi-application, conversation-aware

```
Client (DocLens / ResumeLab / any app)
        |
        v
FastAPI  (cortex/api/main.py)
        |-- POST /ingest          → parse → chunk → embed → Qdrant upsert
        |-- POST /chat            → threaded RAG with SSE streaming
        |-- POST /generate        → LLM generation with supplied context (SSE)
        |-- GET|POST /threads     → conversation thread management
        |-- GET|POST /apps        → registry CRUD
        |-- GET|POST /collections → Qdrant collection management
        |-- POST /admin/reindex   → drop + re-ingest from directory
        |-- POST /extract         → ResumeLab: parse resume → CanonicalProfile
        |-- POST /analyze/match   → ResumeLab: JD gap analysis
        |-- POST /generate/document → ResumeLab: ATS-optimised content
        |-- POST /cover-letter    → ResumeLab: tailored cover letter
        |-- POST /hr-email        → ResumeLab: recruiter outreach email
        |-- POST /compose/rewrite → ResumeLab: email rewrite by instruction
        |-- POST /profile/merge   → ResumeLab: deduplicate two profiles
        |-- POST /llm/ping        → verify LLM provider connectivity
```

---

## Tech Stack

| Component | Package | Version |
|---|---|---|
| API framework | FastAPI + Uvicorn | 0.115.0 / 0.30.6 |
| Data validation | Pydantic | 2.12.5 |
| Embeddings | sentence-transformers | 2.7.0 |
| Default embedding model | BAAI/bge-small-en-v1.5 | 384 dims |
| Vector store | qdrant-client | 1.17.1 |
| Document parsing (layout) | docling | ≥2.0.0 |
| Document parsing (PDF) | PyMuPDF | 1.24.9 |
| LLM (OpenAI) | openai | 1.30.1 |
| LLM (Ollama) | ollama | 0.2.0 |
| ML / reranking | scikit-learn, scipy | 1.8.0 / 1.17.1 |
| Deep learning | torch, transformers | 2.2.2 / 4.41.2 |
| Tokenization | tiktoken | 0.12.0 |
| Python | — | 3.11+ |

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file at the project root.

```env
# --- Qdrant (required) ---
QDRANT_MODE=local          # "cloud" or "local"

# Cloud mode
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Local mode
QDRANT_HOST=localhost
QDRANT_PORT=6333

# --- LLM defaults (used when no llm override is passed per request) ---
LLM_PROVIDER=ollama_cloud  # openai | ollama_cloud | ollama_local
LLM_MODEL=                 # leave blank for provider defaults
OPENAI_API_KEY=
OLLAMA_CLOUD_API_KEY=
OLLAMA_TIMEOUT=600         # seconds; large local model prompts can be slow

# --- Optional ---
CORTEX_APP_REGISTRY_PATH=  # default: app/registry/registry.json
APP_ENV=development        # "production" disables /docs, /redoc, /openapi.json
```

**Qdrant mode rules:**
- `QDRANT_MODE=cloud` → requires `QDRANT_URL` + `QDRANT_API_KEY`
- `QDRANT_MODE=local` → requires `QDRANT_HOST` + `QDRANT_PORT`
- Cortex validates at startup and raises a clear error if config is missing.

**LLM resolution order per request:**
1. `llm` field in the request body (`provider`, `api_key`, `model`)
2. Env var defaults (`LLM_PROVIDER` + corresponding key)
3. Falls back to `ollama_local` if nothing resolves

---

## Running Locally

```bash
# From project root
uvicorn main:app --reload
```

Alternative (same result — root `main.py` is a thin re-export):

```bash
uvicorn cortex.api.main:app --reload
```

API base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## Application Registry

Every client app must be registered before it can use `/ingest`, `/chat`, or `/generate`. Registrations are persisted in `app/registry/registry.json`.

**Pre-registered apps (included in repo):**
- `doclens` — document Q&A (layout_aware_semantic ingestion, markdown generation, query analyzer enabled)
- `resumelab` — resume optimization (resume_canonical ingestion, strict JSON generation, tasks: match/generate/cover_letter/hr_email)
- `cvscan` — CV screening (resume_structured ingestion, structured JSON generation)

### Register a new app

```bash
curl -X POST http://localhost:8000/apps/register \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "myapp",
    "collection": "myapp",
    "ingestion": {
      "strategy": "layout_aware_semantic",
      "max_tokens": 512,
      "min_tokens": 50,
      "overlap_tokens": 64
    },
    "embedding": {
      "model": "BAAI/bge-small-en-v1.5",
      "batch_size": 32,
      "normalize": true
    },
    "retrieval": {
      "top_k": 10,
      "hybrid": true,
      "alpha": 0.7,
      "query_rewrite": true,
      "expand_neighbors": true
    },
    "reranking": {
      "enabled": true,
      "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "top_k": 5,
      "candidate_cap": 20
    },
    "generation": {
      "response_type": "markdown",
      "temperature": 0.1,
      "strict": false,
      "max_retries": 2
    },
    "conversation": {
      "use_query_analyzer": true,
      "clarification_policy": "never"
    },
    "defaults": {
      "system_prompt": "You are a helpful assistant. Answer only from the provided context."
    }
  }'
```

---

## API Reference

### Health

```
GET /
```

Returns a minimal HTML home page for the Cortex service. Use `GET /health` for machine-readable liveness checks.

---

### LLM Ping

```
POST /llm/ping
```

Verifies provider connectivity with a minimal one-token call. Requires an explicit `llm` override — never falls back to env defaults. Used by the frontend Settings test-connection flow.

```bash
curl -X POST http://localhost:8000/llm/ping \
  -H "Content-Type: application/json" \
  -d '{ "llm": { "provider": "openai", "api_key": "sk-...", "model": "gpt-4o-mini" } }'
```

---

### Ingest

```
POST /ingest
```

Accepts `multipart/form-data` (file upload) or `application/json` (file path or raw text). Exactly one content source must be provided.

**Multipart upload:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "app_name=doclens" \
  -F "user_id=user_1" \
  -F "file=@/path/to/document.pdf"
```

**JSON with file path:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"app_name": "doclens", "user_id": "user_1", "file_path": "/path/to/document.pdf"}'
```

**Response:**
```json
{
  "status": "success",
  "doc_id": "a1b2c3d4-...",
  "chunk_count": 18,
  "collection": "doclens",
  "app_name": "doclens"
}
```

Fields:
| Field | Type | Required |
|---|---|---|
| `app_name` | string | yes — must match a registered app |
| `user_id` | string | yes |
| `doc_id` | string | no — auto-generated UUID if omitted |
| `file` | upload | one of three |
| `file_path` | string | one of three |
| `text` | string | one of three |

---

### Chat (RAG with Conversation)

```
POST /chat
```

Full RAG pipeline with persistent conversation threads and SSE streaming by default. Streams token-by-token responses with source citations at the end.

```bash
# Streaming (default)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "doclens",
    "user_id": "user_1",
    "query": "What are the key findings?",
    "doc_ids": ["a1b2c3d4-..."],
    "llm": {
      "provider": "openai",
      "api_key": "sk-...",
      "model": "gpt-4o-mini"
    }
  }'

# Non-streaming
curl -X POST "http://localhost:8000/chat?stream=false" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

**SSE event types (streaming mode):**
- `data: {"type":"token","content":"..."}` — incremental answer token
- `data: {"type":"sources","sources":[...]}` — final source citations
- `data: {"type":"clarification","question":"..."}` — clarification request (when query analyzer triggers)
- `data: [DONE]` — end of stream

**Non-streaming response:**
```json
{
  "answer": "The key findings include...",
  "thread_id": "thr_abc123",
  "sources": [
    {
      "section": "Executive Summary",
      "page": 2,
      "text": "...",
      "score": 0.91,
      "rerank_score": 3.44
    }
  ]
}
```

Fields:
| Field | Type | Required |
|---|---|---|
| `app_name` | string | yes |
| `user_id` | string | yes |
| `query` | string | yes |
| `thread_id` | string | no — creates a new thread if omitted |
| `doc_ids` | `List[string]` | no — searches all user docs if omitted |
| `llm.provider` | `openai` \| `ollama_cloud` \| `ollama_local` | no — falls back to env |
| `llm.api_key` | string | required when `provider=openai` |
| `llm.model` | string | no — provider default if omitted |
| `task` | string | no — named task override from registry |
| `prompt_override` | string | no — replaces system prompt |

---

### Generate

```
POST /generate
```

LLM generation without retrieval. Caller supplies context directly. Supports SSE streaming.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "doclens",
    "user_id": "user_1",
    "query": "Summarize the following text",
    "context": "The report covers...",
    "llm": { "provider": "openai", "api_key": "sk-...", "model": "gpt-4o-mini" }
  }'
```

Add `?stream=true` for SSE streaming.

**Response:**
```json
{ "answer": "..." }
```

---

### Conversation Threads

```
GET  /threads?app_name=doclens&user_id=user_1&limit=20&offset=0
GET  /threads/{thread_id}
DELETE /threads/{thread_id}
PATCH  /threads/{thread_id}
```

**List threads response:**
```json
{
  "threads": [
    { "thread_id": "thr_abc123", "title": "Key findings discussion", "created_at": "...", "message_count": 4 }
  ],
  "total": 12
}
```

`GET /threads/{thread_id}` returns the full thread with all messages. `PATCH` accepts `{ "title": "New title" }`.

---

### Delete Document

```
POST /delete
```

```json
{ "app_name": "doclens", "user_id": "user_1", "doc_id": "a1b2c3d4-..." }
```

**Response:**
```json
{ "status": "ok", "deleted_points": 18, "app_name": "doclens", "user_id": "user_1", "doc_id": "..." }
```

---

### Delete All Documents

```
POST /delete_all
```

```json
{ "app_name": "doclens", "user_id": "user_1" }
```

---

### Admin: Reindex

```
POST /admin/reindex
```

Drops the Qdrant collection for an app and re-ingests all supported files from a source directory. Supported formats: `.pdf`, `.docx`, `.doc`, `.html`, `.htm`, `.md`, `.markdown`, `.txt`.

```json
{
  "app_name": "doclens",
  "user_id": "admin",
  "source_dir": "/path/to/documents"
}
```

---

### Application Registry Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/apps` | List all registered app names |
| `POST` | `/apps/register` | Register a new application |
| `GET` | `/apps/{app_name}` | Get app config |
| `PUT` | `/apps/{app_name}` | Update full app config |
| `DELETE` | `/apps/{app_name}` | Delete app from registry |

---

### Collection Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/collections` | List all Qdrant collections |
| `POST` | `/collections` | Create a new collection |
| `GET` | `/collections/{name}` | Get collection info |
| `DELETE` | `/collections/{name}` | Delete collection |

---

## ResumeLab Endpoints

ResumeLab is the domain-specific resume intelligence layer. No Qdrant reads/writes — purely LLM-driven. All endpoints accept an explicit `llm` override; env defaults apply if omitted.

### Extract

```
POST /extract
```

Accepts `multipart/form-data` (file upload) or `application/json` (raw text/file path). Parses a resume into a structured `CanonicalProfile`.

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@resume.pdf" \
  -F "llm={\"provider\":\"openai\",\"api_key\":\"sk-...\"}"
```

---

### Analyze Match

```
POST /analyze/match
```

Scores a job description against a canonical profile. Returns a deterministic A/B keyword split (computed before any LLM call): `missing` keywords not present in the profile, `omitted` keywords present in the profile but absent from the resume.

```json
{
  "profile": { "...canonical profile..." },
  "job_description": "We are looking for...",
  "llm": { "provider": "openai", "api_key": "sk-..." }
}
```

---

### Generate Document

```
POST /generate/document
```

Produces ATS-optimised resume content blocks from a canonical profile. Generation modes: `canonical_only` (build fresh) or `modify_existing` (targeted rewrites). Template types: `frontend`, `backend`, `fullstack`, `custom`.

---

### Cover Letter

```
POST /cover-letter
```

Generates a tailored cover letter from a canonical profile and job description.

---

### HR Email

```
POST /hr-email
```

Generates a recruiter outreach email (subject + body) from a canonical profile and target role.

---

### Compose / Rewrite

```
POST /compose/rewrite
```

Rewrites an existing email by instruction (adjust style, tone, or length).

---

### Profile Merge

```
POST /profile/merge
```

Deduplicates two `CanonicalProfile` objects using cosine similarity. Accepts a `similarity_threshold` (0.0–1.0) to control deduplication aggressiveness.

---

## Production Deployment (systemd)

Create `/etc/systemd/system/cortex.service`:

```ini
[Unit]
Description=Cortex RAG Engine v2
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/cortex
EnvironmentFile=/opt/cortex/.env
ExecStart=/opt/cortex/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cortex
sudo systemctl start cortex
```

Update workflow:

```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart cortex
sudo systemctl status cortex
curl -sS http://127.0.0.1:8000/
```

---

## Smoke Tests

Scripts must be run as modules from the repo root with the venv active.

```bash
# Infrastructure
python3 -m scripts.test_qdrant
python3 -m scripts.test_embedding

# Core pipeline
python3 -m scripts.test_ingest
python3 -m scripts.test_retrieve
python3 -m scripts.test_generate
python3 -m scripts.test_hybrid_search
python3 -m scripts.test_streaming
python3 -m scripts.test_clarification
python3 -m scripts.test_layout_chunker
python3 -m scripts.test_multi_user

# ResumeLab
python3 -m scripts.test_resumelab_extract
python3 -m scripts.test_resumelab_match
python3 -m scripts.test_resumelab_generate
python3 -m scripts.test_resumelab_merge
```

---

## Logging

```bash
# systemd service
sudo systemctl status cortex
journalctl -u cortex -n 100 --no-pager
journalctl -u cortex -f

# Local (structured JSON logs)
uvicorn main:app --reload --log-level debug
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown application: 'myapp'` | App not in registry | `POST /apps/register` with full config |
| `OpenAI provider requires api_key` | No key in request or env | Pass `llm.api_key` or set `OPENAI_API_KEY` |
| Qdrant connection error | Wrong mode/credentials | Check `QDRANT_MODE` and matching vars |
| `No chunks produced` | Empty or unparseable doc | Verify file is valid and non-empty |
| Empty retrieval results | No docs ingested for user | Re-ingest; confirm `user_id` matches |
| SSE stream cuts off | Client timeout | Increase client read timeout; check `OLLAMA_TIMEOUT` |
| Numpy / torch errors | venv out of sync | `pip install -r requirements.txt --force-reinstall` |

---

## Architecture Notes

- Cortex is a pure backend service — no UI, no rate limiting, no user accounts.
- All behavior per app (chunking, retrieval, generation, conversation) is driven by the registry config.
- LLM provider and key are resolved per request: request body → env vars → local fallback.
- The `ExecutionContext` dataclass carries all resolved config through every pipeline stage; pipeline code never reads config or env vars directly.
- The `analyze_match` A/B keyword split (missing vs. omitted) is deterministic — computed before any LLM call — to reduce token usage and make results reproducible.
- Docs UI is disabled when `APP_ENV=production`.
- ResumeLab functions are truthfulness-constrained: they must never fabricate skills, companies, dates, or metrics not present in the input profile.
