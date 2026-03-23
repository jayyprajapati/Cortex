from dotenv import load_dotenv
import os

load_dotenv()


def _env_bool(name, default=False):
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}

OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOW_DEFAULT_LLM = _env_bool("ALLOW_DEFAULT_LLM", True)

FREE_MAX_DOCS = int(os.getenv("FREE_MAX_DOCS", 1))
FREE_MAX_QUERIES = int(os.getenv("FREE_MAX_QUERIES", 2))
FREE_MAX_PAGES = int(os.getenv("FREE_MAX_PAGES", 3))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama_cloud")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))

COLLECTION_NAME = "documents"