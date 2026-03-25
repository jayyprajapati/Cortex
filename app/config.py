from dotenv import load_dotenv
import os
import logging
from functools import lru_cache

from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama_cloud")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))

COLLECTION_NAME = "documents"


def _require_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_qdrant_port(port_value: str) -> int:
    try:
        return int(port_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("QDRANT_PORT must be a valid integer") from exc


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    mode = str(os.getenv("QDRANT_MODE") or "").strip().lower()

    if mode not in {"cloud", "local"}:
        raise ValueError(
            "QDRANT_MODE must be set to either 'cloud' or 'local'"
        )

    if mode == "cloud":
        url = _require_env("QDRANT_URL")
        api_key = _require_env("QDRANT_API_KEY")
        logger.info("Using Qdrant mode: cloud")
        logger.info("Qdrant endpoint: %s", url)
        return QdrantClient(url=url, api_key=api_key)

    host = _require_env("QDRANT_HOST")
    port = _parse_qdrant_port(_require_env("QDRANT_PORT"))
    logger.info("Using Qdrant mode: local")
    logger.info("Qdrant endpoint: %s:%s", host, port)
    return QdrantClient(host=host, port=port)