from app.pipeline.ingest_pipeline import ingest_document

ingest_document("data/sample.pdf", "doc1")

# to run: python -m scripts.test_ingest