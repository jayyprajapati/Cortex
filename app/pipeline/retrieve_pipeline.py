from app.retrieval.query_embedder import embed_query
from app.retrieval.search import search_documents, format_results
from app.reranker.reranker import rerank
from app.vectorstore.qdrant_store import get_collection_size

def retrieve(query):

    query_vector = embed_query(query)
    collection_size = get_collection_size()
    top_k = min(10, collection_size)
    results = search_documents(query_vector, top_k=top_k)

    chunks = format_results(results)

    reranked = rerank(query, chunks)

    best_score = reranked[0]["rerank_score"]

    final = [
        c for c in reranked
        if c["rerank_score"] >= best_score * 0.5
    ]

    return final[:3]