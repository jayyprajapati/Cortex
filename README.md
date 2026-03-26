# Cortex - RAG Engine from Scratch
## 1) Project Overview

Cortex is a standalone RAG + LLM backend service that you can plug into any personal project as a shared AI core. Instead of rebuilding ingestion, retrieval, and model wiring in every app, Cortex stays independent so each product can integrate with one stable API for knowledge ingestion and grounded generation. This makes it a single source of truth for your personal RAG stack while your other projects stay thin and focused on their own UI or business logic.

Problems Cortex solves:

- Standardizes ingestion for PDF, DOCX, and Markdown.
- Provides one vector retrieval backend (Qdrant) for all downstream services.
- Keeps LLM orchestration and prompt-building in one deployable service.
- Supports multi-document retrieval with user-level filtering and reranking.

High-level architecture:

```text
Client/API Consumer
		|
		v
FastAPI (cortex.api.main)
		|-- /ingest  -> parsing -> chunking -> embedding -> Qdrant upsert
		|-- /query   -> query embedding -> Qdrant search -> rerank -> LLM
		|-- /generate (optional direct prompt -> LLM)
```

## 2) Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- Sentence Transformers (default embedding model: BAAI/bge-small-en)
- Qdrant (Cloud or Local)
- PyMuPDF and DOCX/Markdown parsers for ingestion

## 3) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Initialize the Qdrant collection and indexes:

```bash
python3 -m scripts.setup_vector_db
```

## 4) Environment Variables (.env)

Create a .env file at the project root.

Required Qdrant mode switch:

```env
QDRANT_MODE=cloud

# Cloud mode
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Local mode
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

How switching works:

- Set QDRANT_MODE=cloud and provide QDRANT_URL + QDRANT_API_KEY.
- Set QDRANT_MODE=local and provide QDRANT_HOST + QDRANT_PORT.
- No code changes are required. Cortex validates the required variables at startup and raises a clear error when config is invalid.

Additional runtime config:

```env
LLM_PROVIDER=ollama_cloud
LLM_MODEL=
OLLAMA_CLOUD_API_KEY=
OPENAI_API_KEY=

EMBEDDING_MODEL=BAAI/bge-small-en
VECTOR_SIZE=384
```

## 5) Running Locally

Recommended (matches this repository entrypoint):

```bash
uvicorn main:app --reload
```

Alternative module form:

```bash
python3 -m uvicorn cortex.api.main:app --reload
```

Compatibility command (if your deployment path exposes app.api.main):

```bash
uvicorn app.api.main:app --reload
```

API base URL:

```text
http://localhost:8000
```

## 6) Production Deployment (systemd)

Example service file: /etc/systemd/system/cortex.service

```ini
[Unit]
Description=Cortex FastAPI RAG Engine
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

## 7) API Endpoints

### Health

- GET /
- Response:

```json
{
	"message": "RAG engine running"
}
```

### Ingest

- POST /ingest
- Supports application/json and multipart/form-data.
- Exactly one of file_path, file, or text must be provided.

Multipart/form-data example:

```bash
curl -X POST http://localhost:8000/ingest \
	-F "user_id=user_1" \
	-F "doc_id=doc_renewables_001" \
	-F "file=@/absolute/path/to/report.pdf" \
	-F "llm_provider=openai" \
	-F "llm_api_key=YOUR_KEY" \
	-F "llm_model=gpt-4o-mini"
```

Success response:

```json
{
	"status": "success",
	"doc_id": "doc_renewables_001"
}
```

### Query

- POST /query
- Request:

```json
{
	"query": "Summarize the wind energy section",
	"user_id": "user_1",
	"app_name": "default",
	"doc_id": "doc_renewables_001",
	"llm": {
		"provider": "openai",
		"api_key": "optional_key",
		"model": "optional_model"
	}
}
```

Response shape:

```json
{
	"answer": "...",
	"sources": [
		{
			"section": "The Rise of Renewable Energy",
			"page": 1,
			"source": "The Rise of Renewable Energy. Page 1",
			"text": "...",
			"score": 0.90,
			"rerank_score": 3.44
		}
	]
}
```

Fallback behavior: when no context is retrieved, answer is No relevant information found. and sources is an empty list.

### Generate (optional)

- POST /generate
- Request:

```json
{
	"prompt": "Summarize this text in one paragraph",
	"model": "optional_model"
}
```

- Response:

```json
{
	"answer": "..."
}
```

## 8) Qdrant Configuration

- VECTOR_SIZE defaults to 384 for BAAI/bge-small-en.
- Qdrant collection vector size must match embedding output dimension.
- If dimensions do not match, upserts and retrieval operations will fail.
- Recommended approach: always initialize/recreate collection through code.

Collection bootstrap command:

```bash
python3 -m scripts.setup_vector_db
```

This setup also creates payload indexes for user_id and doc_id, which are required for efficient filtered retrieval.

## 9) Logging and Debugging

Service status:

```bash
sudo systemctl status cortex
```

Recent logs:

```bash
journalctl -u cortex -n 100 --no-pager
```

Live logs:

```bash
journalctl -u cortex -f
```

Common issues:

- Numpy not available
	- Reinstall pinned dependencies in active venv.
	- Command: pip install -r requirements.txt --upgrade --force-reinstall

- Qdrant connection errors
	- Verify QDRANT_MODE and required variables for that mode.
	- Validate host/url reachability from the server.
	- Re-run collection setup after endpoint changes.

- Empty retrieval results
	- Confirm ingestion completed for the same user_id.
	- Verify doc_id filter is correct.
	- Check query quality and test broader queries.
	- Confirm collection exists and contains points.

Operational smoke tests:

```bash
python3 -m scripts.test_qdrant
python3 -m scripts.test_ingest
python3 -m scripts.test_retrieve
python3 -m scripts.test_generate
```

## 10) Deployment Workflow

Use this sequence for each production update:

```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart cortex
```

Post-deploy verification:

```bash
sudo systemctl status cortex
curl -sS http://127.0.0.1:8000/
```

## 11) Architecture Notes

- Cortex is an independent backend service.
- Cortex should not contain UI logic.
- Cortex should not contain client-facing rate limiting logic.
- Cortex should remain focused on pure RAG orchestration: ingestion, indexing, retrieval, reranking, and LLM response generation.

## 12) Future Improvements

- Service-to-service authentication and request signing
- Result caching for frequent retrieval queries
- Metrics and monitoring (Prometheus + Grafana)
- Structured tracing for ingestion and query latency

