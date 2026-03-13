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
def ingest_document(path, doc_id):

    elements = load_document(path)

    chunks = create_chunks(elements, doc_id)

    embeddings = embed_chunks(chunks)

    store_chunks(chunks, embeddings)