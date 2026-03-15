from app.pipeline.ingest_pipeline import ingest_document
# This script is a simple test to verify that the document ingestion pipeline is working correctly.
ingest_document("data/test_pipeline.pdf", "doc2")

# to run: python3 -m scripts.test_ingest