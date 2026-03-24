import re

from app.retrieval.query_embedder import embed_query
from app.retrieval.search import (
    search_documents,
    format_results,
    list_user_document_ids,
    fetch_document_chunks,
)
from app.reranker.reranker import rerank
from app.vectorstore.qdrant_store import get_collection_size


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "tell", "that", "the", "this",
    "to", "what", "when", "where", "which", "who", "why", "with", "about", "can", "you",
}

REFERENCE_TOKENS = {"this", "that", "it", "document", "doc", "file", "pdf", "notes", "note"}
DOC_INTENT_TOKENS = {"summarize", "summarise", "summary", "overview", "describe", "explain", "about", "tell"}
SUMMARY_QUERY_TOKENS = {"summarize", "summarise", "summary", "overview"}


def _normalize_query(text):
    cleaned = " ".join(str(text or "").strip().lower().split())
    return cleaned.rstrip("?.! ")


def _tokenize_query(query):
    return re.findall(r"[a-z0-9']+", _normalize_query(query))


def _has_doc_reference(tokens):
    return any(token in REFERENCE_TOKENS for token in tokens)


def _query_specific_terms(tokens):
    return [
        token for token in tokens
        if len(token) > 2
        and token not in STOPWORDS
        and token not in REFERENCE_TOKENS
        and token not in DOC_INTENT_TOKENS
    ]


def _is_document_level_query(query):
    tokens = _tokenize_query(query)

    if not tokens:
        return True

    specific_terms = _query_specific_terms(tokens)
    has_doc_reference = _has_doc_reference(tokens)
    has_summary_intent = any(token in DOC_INTENT_TOKENS for token in tokens)

    if has_doc_reference and len(specific_terms) <= 2:
        return True

    if has_summary_intent and len(specific_terms) <= 1:
        return True

    return len(tokens) <= 4 and len(specific_terms) == 0


def _is_summary_query(query):
    tokens = _tokenize_query(query)
    return any(token in SUMMARY_QUERY_TOKENS for token in tokens)


def _build_clarification_response(doc_ids):
    docs = "\n".join(f"- {doc_id}" for doc_id in doc_ids)
    return (
        "I found multiple documents for your account. "
        "Please ask again with a specific doc_id.\n"
        "Available documents:\n"
        f"{docs}"
    )


def retrieve(query, user_id, doc_id=None, return_meta=False):

    summary_query = _is_summary_query(query)
    document_level_query = _is_document_level_query(query)
    resolved_doc_id = str(doc_id).strip() if doc_id else None
    known_docs = []

    if document_level_query and not resolved_doc_id:
        known_docs = list_user_document_ids(user_id)

        if len(known_docs) > 1:
            payload = {
                "chunks": [],
                "is_vague": True,
                "doc_ids": known_docs,
                "clarification": _build_clarification_response(known_docs),
            }
            return payload if return_meta else []

        if len(known_docs) == 1:
            resolved_doc_id = known_docs[0]

        if len(known_docs) == 0:
            payload = {
                "chunks": [],
                "is_vague": True,
                "doc_ids": [],
                "clarification": "I could not find any documents for your account yet. Please ingest a document first.",
            }
            return payload if return_meta else []

    query_vector = embed_query(query)

    collection_size = get_collection_size()
    top_k_cap = 25 if summary_query else 10
    top_k_floor = 3 if summary_query else 1
    top_k = max(top_k_floor, min(top_k_cap, collection_size))

    results = search_documents(
        query_vector,
        user_id=user_id,
        doc_id=resolved_doc_id,
        top_k=top_k
    )

    chunks = format_results(results)

    if not chunks and document_level_query and resolved_doc_id:
        chunks = fetch_document_chunks(user_id=user_id, doc_id=resolved_doc_id, limit=60)

    if not chunks:
        if document_level_query:
            message = "I could not retrieve enough content to summarize this document. Try a more specific question."
            if resolved_doc_id:
                message = (
                    f"I could not retrieve enough content from doc_id '{resolved_doc_id}'. "
                    "Try a more specific question."
                )

            payload = {
                "chunks": [],
                "is_vague": True,
                "doc_ids": [resolved_doc_id] if resolved_doc_id else known_docs,
                "clarification": message,
            }
            return payload if return_meta else []

        return {"chunks": [], "is_vague": False, "doc_ids": [], "clarification": None} if return_meta else []
        
    reranked = rerank(query, chunks)

    best_score = reranked[0]["rerank_score"]

    final = [
        c for c in reranked
        if c["rerank_score"] >= best_score * 0.5
    ]

    limit = 6 if summary_query else 3

    selected = final[:limit]
    if return_meta:
        return {
            "chunks": selected,
            "is_vague": document_level_query,
            "doc_ids": [resolved_doc_id] if resolved_doc_id else known_docs,
            "clarification": None,
        }

    return selected