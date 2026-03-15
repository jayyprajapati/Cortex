from app.retrieval.query_embedder import embed_query
from app.retrieval.search import search_documents, format_results


def retrieve(query):

    query_vector = embed_query(query)

    results = search_documents(query_vector)

    chunks = format_results(results)

    return chunks