# Cortex

Cortex is a standalone RAG + LLM backend service that you can plug into any personal project as a shared AI core. Instead of rebuilding ingestion, retrieval, and model wiring in every app, Cortex stays independent so each product can integrate with one stable API for knowledge ingestion and grounded generation. This makes it a single source of truth for your personal RAG stack while your other projects stay thin and focused on their own UI or business logic.

## What Cortex Provides

- Document ingestion pipeline for PDF, DOCX, and Markdown
- Chunking + embedding + Qdrant vector storage
- Retrieval + reranking
- Grounded answer generation with source references
- BYOK (Bring Your Own Key) support for LLM requests
- Free-tier controls (in-memory limits with API-key bypass)
- File upload support via JSON path-based ingest and multipart form-data ingest
- Safe fallback behavior when no context is retrieved

## System Flow

1. Ingest document or text
2. Parse by type (PDF, DOCX, Markdown)
3. Chunk text into semantically usable units
4. Embed chunks and upsert into Qdrant
5. Query -> retrieve relevant chunks -> rerank
6. Build prompt with context
7. Generate answer with selected LLM

## Core Components

- API layer: `cortex/api/main.py`
- Ingestion pipeline: `app/pipeline/ingest_pipeline.py`
- Retrieval pipeline: `app/pipeline/retrieve_pipeline.py`
- Generation pipeline: `app/pipeline/generate_pipeline.py`
- LLM adapters: `app/llm/`
- Vector store integration: `app/vectorstore/qdrant_store.py`
- Prompt registry/templates: `cortex/prompts/`
- Utility scripts: `scripts/`

## Prerequisites

- Python 3.11+
- Qdrant running on `localhost:6333`
- Optional: local Ollama server for local generation

### Start Qdrant (example with Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m scripts.setup_vector_db
```

## Environment Variables

Create a `.env` file in the project root.

### Required in most cloud-backed setups

- `OLLAMA_CLOUD_API_KEY`: Required if you use `ollama_cloud` as default provider or fallback

### Required when using OpenAI

- `OPENAI_API_KEY`: Required if default provider is `openai`, or when BYOK is not supplied for OpenAI and default-key fallback is enabled

### Feature flags and limits

- `ALLOW_DEFAULT_LLM` (default: `True`)
- `FREE_MAX_DOCS` (default: `1`)
- `FREE_MAX_QUERIES` (default: `2`)
- `FREE_MAX_PAGES` (default: `3`)

### Model/provider defaults

- `LLM_PROVIDER` (default: `ollama_cloud`)
- `LLM_MODEL` (optional)

### Embedding/vector config

- `EMBEDDING_MODEL` (default: `BAAI/bge-small-en`)
- `VECTOR_SIZE` (default: `384`)

## Run Cortex API

Use either entrypoint:

```bash
uvicorn main:app --reload
```

or

```bash
python3 -m uvicorn cortex.api.main:app --reload
```

Service base URL (local): `http://localhost:8000`

## API Endpoints

### GET /

Health/basic message endpoint.

### POST /query

Request JSON:

```json
{
	"query": "What does this document say about wind energy?",
	"user_id": "user_1",
	"app_name": "default",
	"doc_id": "optional_doc_filter",
	"llm": {
		"provider": "openai",
		"api_key": "optional_user_api_key",
		"model": "optional_model_name"
	}
}
```

Notes:

- `llm` is optional
- `provider` supports `openai` and `ollama`
- If no chunks are found, Cortex returns `No relevant information found.` and does not call an LLM

### POST /ingest

You can ingest in two ways.

#### A) JSON body (file path or raw text)

```json
{
	"user_id": "user_1",
	"doc_id": "doc_1",
	"file_path": "data/sample.pdf",
	"text": null,
	"llm": {
		"provider": "openai",
		"api_key": "optional_user_api_key",
		"model": "optional_model_name"
	}
}
```

Rules:

- Provide exactly one of `file_path` or `text`

#### B) Multipart upload (for external services)

Form fields:

- `user_id` (required)
- `doc_id` (required)
- `file` (required for upload flow)
- `llm_provider` (optional)
- `llm_api_key` (optional)
- `llm_model` (optional)

Example:

```bash
curl -X POST http://localhost:8000/ingest \
	-F "user_id=user_1" \
	-F "doc_id=doc_uploaded_1" \
	-F "file=@/absolute/path/to/file.pdf" \
	-F "llm_provider=openai" \
	-F "llm_api_key=YOUR_KEY"
```

### POST /generate

Direct generation endpoint (no retrieval):

```json
{
	"prompt": "Summarize this in one paragraph",
	"model": "optional_model_name"
}
```

## BYOK and Default-Key Behavior

- If request contains `llm.api_key`, Cortex uses that key
- If request does not include a key and `ALLOW_DEFAULT_LLM=True`, Cortex can use system/default key (when available)
- If `ALLOW_DEFAULT_LLM=False` and request has no API key, request is rejected

## Free-Tier Rules (In-Memory)

Tracked per `user_id` in memory and reset on server restart:

- Documents limit: `FREE_MAX_DOCS` (default 1)
- Query limit: `FREE_MAX_QUERIES` (default 2)

Behavior:

- If user exceeds limit and does not provide API key -> request rejected
- If API key is provided -> limits are bypassed

## File Size/Page Restrictions

- For PDF ingestion, if pages exceed `FREE_MAX_PAGES` (default 3) and no API key is provided, ingest is rejected
- With API key, this free-tier page gate is bypassed

## Markdown Output Behavior

Prompt templates enforce clean markdown formatting with structure and source references.

## Safe Fallback Behavior

When retrieval returns no chunks:

- answer: `No relevant information found.`
- sources: empty list
- no LLM call is made

## Script-by-Script Testing

Activate your environment first:

```bash
source .venv/bin/activate
```

### 1) Verify Qdrant connection

```bash
python3 -m scripts.test_qdrant
```

### 2) Initialize/recreate vector collection

```bash
python3 -m scripts.setup_vector_db
```

### 3) Quick embedding sanity check

```bash
python3 -m scripts.test_embedding
```

### 4) Parse/ingestion loader check

```bash
python3 -m scripts.test_ingestion
```

### 5) Ingest pipeline test

```bash
python3 -m scripts.test_ingest
```

### 6) Retrieval pipeline test

```bash
python3 -m scripts.test_retrieve
```

### 7) End-to-end generation test

```bash
python3 -m scripts.test_generate
```

### 8) Multi-user isolation and doc-filter checks

```bash
python3 -m scripts.test_multi_user
```

## Typical Local Validation Sequence

```bash
python3 -m scripts.setup_vector_db
python3 -m scripts.test_ingest
python3 -m scripts.test_retrieve
python3 -m scripts.test_generate
```

## Troubleshooting

- Multipart ingest errors about form parsing:
	- Ensure `python-multipart` is installed (`pip install -r requirements.txt`)
- No results returned in query:
	- Confirm data was ingested for the same `user_id`
	- Confirm `doc_id` filter (if provided) matches ingested document id
	- Verify Qdrant is running on `localhost:6333`
- Provider/key errors:
	- Verify `LLM_PROVIDER`, `ALLOW_DEFAULT_LLM`, and relevant API key env vars

## Why Cortex as an Independent Service

Keeping Cortex separate gives you a reusable AI foundation across all personal products. Each new app can integrate as a plugin/client to the same RAG core, share ingestion/retrieval logic, and avoid duplicated model plumbing. This reduces maintenance overhead, keeps behavior consistent across projects, and lets you improve one backend that powers everything you build.

