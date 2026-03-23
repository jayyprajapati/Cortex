from app.config import FREE_MAX_PAGES
from app.ingestion.inspector import inspect_pdf
from app.ingestion.router import detect_file_type
from app.ingestion.loader import load_document
from app.chunking.chunker import create_chunks
from app.embeddings.embedder import embed_chunks
from app.vectorstore.qdrant_store import store_chunks

# This module defines the main function for the ingestion pipeline, which orchestrates the entire process of loading a document, creating text chunks, generating embeddings, and storing them in a vector database.
# The ingest_document function takes the file path and document ID as input, and performs the following steps:
# 1. Loads the document using the load_document function, which extracts text elements from the file.
# 2. Creates text chunks from the extracted elements using the create_chunks function, which organizes the text into manageable pieces based on a token limit.
# 3. Generates vector embeddings for the created chunks using the embed_chunks function, which utilizes a pre-trained sentence transformer model to encode the text into vector representations.
# 4. Stores the chunks and their corresponding embeddings in a Qdrant vector database using the store_chunks function, which upserts the data into a collection for efficient retrieval during question-answering tasks.
# This function is a critical part of the pipeline that ensures the processed and embedded chunks of text are stored in a way that facilitates fast and accurate retrieval when users query the system.
def _ingest_elements(elements, doc_id, user_id):

    chunks = create_chunks(elements, doc_id)

    embeddings = embed_chunks(chunks)
    print(f"Prepared {len(chunks)} chunks for doc_id={doc_id}")
    store_chunks(chunks, embeddings, user_id)

    return chunks


def _validate_document_limits(path, api_key=None):
    file_type = detect_file_type(path)
    user_api_key = (api_key or "").strip()

    if file_type == "pdf" and not user_api_key:
        meta = inspect_pdf(path)
        if meta["pages"] > FREE_MAX_PAGES:
            raise ValueError(
                f"Free tier supports up to {FREE_MAX_PAGES} PDF pages. Provide an API key to ingest larger files."
            )


def ingest_document(path, doc_id, user_id, api_key=None):

    _validate_document_limits(path, api_key=api_key)

    elements = load_document(path)

    return _ingest_elements(elements, doc_id, user_id)


def ingest_text(text, doc_id, user_id):
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned:
        raise ValueError("text is required for raw text ingestion")

    elements = [{"text": cleaned, "page": 1}]
    return _ingest_elements(elements, doc_id, user_id)