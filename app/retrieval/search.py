from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("localhost", port=6333)

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