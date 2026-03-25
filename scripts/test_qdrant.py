from app.config import get_qdrant_client

client = get_qdrant_client()

collections = client.get_collections()

print(collections)