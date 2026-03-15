from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

def search_documents(query_vector, top_k=5):

    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=top_k
    )

    return results.points

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