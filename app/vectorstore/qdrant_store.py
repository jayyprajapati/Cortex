import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

# This function creates a collection in the Qdrant vector database with the name "documents". 
# The collection is configured to store vector embeddings of size 384 and uses cosine distance for similarity search. 
# This setup is essential for storing the embeddings generated from text chunks and enabling efficient retrieval based on semantic similarity during question-answering tasks.
def create_collection():

    client.recreate_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

# This function takes a list of Chunk objects and their corresponding embeddings, and stores them in the Qdrant collection. 
# Each chunk is converted into a PointStruct, which includes the chunk's unique ID, its vector embedding, and a payload containing the original text, document ID, and page number. 
# The points are then upserted into the "documents" collection in Qdrant, allowing for efficient retrieval of relevant chunks based on their vector representations during question-answering tasks. 
# This function is a critical part of the pipeline that ensures the processed and embedded chunks of text are stored in a way that facilitates fast and accurate retrieval when users query the system.
from qdrant_client.models import PointStruct

def store_chunks(chunks, embeddings):

    points = []

    for chunk, vector in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section
                }
            )
        )

    client.upsert(
        collection_name="documents",
        points=points
    )

def get_collection_size(collection_name="documents"):
    info = client.get_collection(collection_name)
    return info.points_count