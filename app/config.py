from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)

# LLM configuration — resolved at runtime by registry/service.py
OLLAMA_CLOUD_API_KEY: str | None = os.getenv("OLLAMA_CLOUD_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama_cloud")
LLM_MODEL: str | None = os.getenv("LLM_MODEL")


def _require_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_port(raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("QDRANT_PORT must be a valid integer") from exc


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    mode = str(os.getenv("QDRANT_MODE") or "").strip().lower()

    if mode not in {"cloud", "local"}:
        raise ValueError("QDRANT_MODE must be set to 'cloud' or 'local'")

    if mode == "cloud":
        url = _require_env("QDRANT_URL")
        api_key = _require_env("QDRANT_API_KEY")
        logger.info("Qdrant mode=cloud endpoint=%s", url)
        return QdrantClient(url=url, api_key=api_key)

    host = _require_env("QDRANT_HOST")
    port = _parse_port(_require_env("QDRANT_PORT"))
    logger.info("Qdrant mode=local endpoint=%s:%s", host, port)
    return QdrantClient(host=host, port=port)
