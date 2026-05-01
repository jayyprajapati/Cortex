"""
Thin shim kept for any direct imports that haven't migrated to
app.embeddings.embedder.embed_query. New code should use
app.embeddings.embedder directly.
"""
from app.embeddings.embedder import embed_query as embed_query  # noqa: F401
