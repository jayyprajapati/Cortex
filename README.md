test qdrant
```
python3 scripts/test_qdrant.py
```

run main.py
```
uvicorn main:app --reload
```

API endpoints
```
POST /query
{
	"query": "...",
	"user_id": "...",
	"app_name": "default",
	"doc_id": "optional"
}

POST /ingest
{
	"user_id": "...",
	"doc_id": "...",
	"file_path": "optional",
	"text": "optional"
}

POST /generate
{
	"prompt": "...",
	"model": "optional"
}
```

curl examples
```
curl -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"query":"What is this document about?","user_id":"test_user","app_name":"default"}'

curl -X POST http://localhost:8000/ingest \
	-H "Content-Type: application/json" \
	-d '{"user_id":"test_user","doc_id":"doc_text_1","text":"This is a raw text document about RAG systems."}'

curl -X POST http://localhost:8000/generate \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Say hello in one line"}'
```

Cortex package entrypoint
```
python3 -m uvicorn cortex.api.main:app --reload
```

