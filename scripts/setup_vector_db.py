from app.vectorstore.qdrant_store import create_collection

create_collection()

print("Vector DB initialized")

# to run: python3 -m scripts.setup_vector_db