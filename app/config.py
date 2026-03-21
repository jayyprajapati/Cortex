from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))

COLLECTION_NAME = "documents"