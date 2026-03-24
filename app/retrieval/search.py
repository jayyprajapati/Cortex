from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", port=6333)


def list_user_document_ids(user_id, max_docs=1000):

    normalized_user_id = str(user_id).strip() if user_id is not None else ""
    if not normalized_user_id:
        raise ValueError("user_id is required for retrieval")

    query_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=normalized_user_id)
            )
        ]
    )

    doc_ids = set()
    offset = None

    while len(doc_ids) < max_docs:
        points, next_offset = client.scroll(
            collection_name="documents",
            scroll_filter=query_filter,
            offset=offset,
            limit=100,
            with_payload=["doc_id"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id") or "").strip()
            if doc_id:
                doc_ids.add(doc_id)

        if next_offset is None:
            break

        offset = next_offset

    return sorted(doc_ids)


def fetch_document_chunks(user_id, doc_id=None, limit=200):

    normalized_user_id = str(user_id).strip() if user_id is not None else ""
    if not normalized_user_id:
        raise ValueError("user_id is required for retrieval")

    must_conditions = [
        FieldCondition(
            key="user_id",
            match=MatchValue(value=normalized_user_id)
        )
    ]

    if doc_id:
        normalized_doc_id = str(doc_id).strip()
        must_conditions.append(
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=normalized_doc_id)
            )
        )

    query_filter = Filter(must=must_conditions)

    points, _ = client.scroll(
        collection_name="documents",
        scroll_filter=query_filter,
        limit=limit,
        with_payload=["text", "page", "doc_id", "section"],
        with_vectors=False,
    )

    chunks = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            continue

        chunks.append({
            "text": text,
            "page": payload.get("page"),
            "doc_id": payload.get("doc_id"),
            "section": payload.get("section"),
            "score": 0.0,
        })

    return chunks

def search_documents(query_vector, user_id, doc_id=None, top_k=5):

    normalized_user_id = str(user_id).strip() if user_id is not None else ""
    if not normalized_user_id:
        raise ValueError("user_id is required for retrieval")

    must_conditions = [
        FieldCondition(
            key="user_id",
            match=MatchValue(value=normalized_user_id)
        )
    ]

    if doc_id:
        normalized_doc_id = str(doc_id).strip()
        must_conditions.append(
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=normalized_doc_id)
            )
        )

    query_filter = Filter(must=must_conditions)

    response = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=top_k,
        query_filter=query_filter
    )

    return response.points
    
def format_results(results):

    chunks = []

    for r in results:

        payload = r.payload

        chunks.append({
            "text": payload["text"],
            "page": payload["page"],
            "doc_id": payload["doc_id"],
            "section": payload.get("section"),
            "score": r.score
        })

    return chunks