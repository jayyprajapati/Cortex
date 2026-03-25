import uuid
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)
from app.config import COLLECTION_NAME, VECTOR_SIZE, get_qdrant_client

client = get_qdrant_client()

# This function creates a collection in the Qdrant vector database with the name "documents". 
# The collection is configured to store vector embeddings of size 384 and uses cosine distance for similarity search. 
# This setup is essential for storing the embeddings generated from text chunks and enabling efficient retrieval based on semantic similarity during question-answering tasks.
def create_collection():

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    # Ensure filtered lookups remain fast and valid across Qdrant deployments.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

# This function takes a list of Chunk objects and their corresponding embeddings, and stores them in the Qdrant collection. 
# Each chunk is converted into a PointStruct, which includes the chunk's unique ID, its vector embedding, and a payload containing the original text, document ID, and page number. 
# The points are then upserted into the "documents" collection in Qdrant, allowing for efficient retrieval of relevant chunks based on their vector representations during question-answering tasks. 
# This function is a critical part of the pipeline that ensures the processed and embedded chunks of text are stored in a way that facilitates fast and accurate retrieval when users query the system.
from qdrant_client.models import PointStruct

def store_chunks(chunks, embeddings, user_id):

    normalized_user_id = str(user_id).strip() if user_id is not None else ""
    if not normalized_user_id:
        raise ValueError("user_id is required for chunk storage")

    points = []

    for chunk, vector in zip(chunks, embeddings):
        normalized_doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
        if not normalized_doc_id:
            raise ValueError("doc_id is required for chunk storage")

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk.text,
                    "doc_id": normalized_doc_id,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "user_id": normalized_user_id,
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def get_collection_size(collection_name=COLLECTION_NAME):
    info = client.get_collection(collection_name)
    return info.points_count


def _build_delete_filter(user_id, doc_id=None):
    normalized_user_id = str(user_id).strip() if user_id is not None else ""
    if not normalized_user_id:
        raise ValueError("user_id is required for deletion")

    must_conditions = [
        FieldCondition(
            key="user_id",
            match=MatchValue(value=normalized_user_id)
        )
    ]

    if doc_id is not None:
        normalized_doc_id = str(doc_id).strip()
        if not normalized_doc_id:
            raise ValueError("doc_id is required for document deletion")

        must_conditions.append(
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=normalized_doc_id)
            )
        )

    return Filter(must=must_conditions)


def _delete_by_filter(delete_filter):
    before_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=delete_filter,
        exact=True,
    ).count

    if before_count == 0:
        raise ValueError("No matching vectors found to delete")

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=delete_filter,
        wait=True,
    )

    after_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=delete_filter,
        exact=True,
    ).count

    deleted_points = before_count - after_count
    if deleted_points <= 0:
        raise RuntimeError("Delete operation did not remove any vectors")

    return deleted_points


def delete_document_vectors(user_id, doc_id):
    delete_filter = _build_delete_filter(user_id=user_id, doc_id=doc_id)
    return _delete_by_filter(delete_filter)


def delete_user_vectors(user_id):
    delete_filter = _build_delete_filter(user_id=user_id)
    return _delete_by_filter(delete_filter)