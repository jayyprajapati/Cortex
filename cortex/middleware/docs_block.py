"""
BlockDocsInProduction middleware.

Even when FastAPI is configured with docs_url=None, this middleware provides
defense-in-depth by rejecting any request to documentation paths from non-dev
hosts with a 404. This ensures docs are never reachable from production domains
(e.g. cortex.jp.dev) regardless of how the FastAPI app is instantiated.

Allowed hosts are controlled by the DOCS_ALLOWED_HOSTS env var (comma-separated).
Default allowlist: localhost, 127.0.0.1, 0.0.0.0
"""

from __future__ import annotations

import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that must never be reachable from non-dev hosts.
_DOCS_PREFIXES: tuple[str, ...] = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/openapi",
    "/docs/oauth2-redirect",
)

_DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "0.0.0.0")


def _build_allowed_hosts() -> tuple[str, ...]:
    """Return the effective allowlist from env var or built-in defaults."""
    raw = os.getenv("DOCS_ALLOWED_HOSTS", "").strip()
    if raw:
        entries = tuple(h.strip() for h in raw.split(",") if h.strip())
        if entries:
            return entries
    return _DEFAULT_ALLOWED_HOSTS


# Compute once at module load time — avoids repeated os.getenv on every request.
_ALLOWED_HOSTS: tuple[str, ...] = _build_allowed_hosts()


def _is_dev_host(host: str) -> bool:
    """Return True when the Host header value matches a known dev host."""
    if not host:
        # No Host header — treat as non-dev (fail closed).
        return False

    # Strip port from hostname for bare-host comparison.
    bare_host = host.split(":")[0].lower()
    return bare_host in _ALLOWED_HOSTS


def _is_docs_path(path: str) -> bool:
    """Return True when the request path targets a documentation endpoint."""
    for prefix in _DOCS_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class BlockDocsInProduction(BaseHTTPMiddleware):
    """
    Middleware that returns 404 for documentation paths when the request
    originates from a non-dev host.

    Usage::

        app.add_middleware(BlockDocsInProduction)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_docs_path(request.url.path):
            host = request.headers.get("host", "")
            if not _is_dev_host(host):
                logger.warning(
                    "Blocked docs request from non-dev host=%r path=%r",
                    host,
                    request.url.path,
                )
                return Response(status_code=404)
        return await call_next(request)
