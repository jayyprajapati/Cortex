"""Admin API key dependency for protecting admin/registry routes."""
import hmac
import os
import logging
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin_key(key: str | None = Security(_header)) -> None:
    """FastAPI dependency that validates the X-Admin-Key header.

    - If ADMIN_API_KEY env var is not set: return 503 (fail closed in prod)
    - If key is missing or wrong: return 401
    """
    expected = os.environ.get("ADMIN_API_KEY")
    if not expected:
        logger.error("ADMIN_API_KEY env var not set — rejecting admin request (fail closed)")
        raise HTTPException(
            status_code=503,
            detail="Admin endpoint unavailable: server misconfiguration",
        )
    if not key:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    if not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin key")
