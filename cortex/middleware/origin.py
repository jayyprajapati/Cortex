"""
OriginRefererMiddleware — browser origin / referer pinning.

NOTE ON SECURITY SCOPE
----------------------
Header-based checks block browsers and casual scripted abuse, not a determined
attacker with a proxy. Real security boundary is JWT (P0.4) + admin key (P0.3).
This middleware is a first-layer convenience filter, not a cryptographic control.

BEHAVIOUR
---------
- OPTIONS requests are always passed through (pre-flight must not be blocked).
- /health and /ready are always passed through (infra probes carry no Origin).
- When no Origin *and* no Referer header is present the request is allowed
  (server-to-server calls, curl without headers, internal services).
- When Origin or Referer is present and does NOT match the allowlist → 403.
- Wildcard patterns such as ``https://*.jayprajapati.dev`` are supported.
  ``*`` matches exactly ONE DNS label (no dots), so
  ``https://app.jayprajapati.dev`` matches but
  ``https://evil.com.jayprajapati.dev`` does NOT.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist configuration (shared with CORS — same env var)
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS: list[str] = [
    "https://*.jayprajapati.dev",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
]

# Paths exempt from origin checks.
_EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/ready"})


def _build_allowed_origins() -> list[str]:
    """Return the effective origin allowlist from env or built-in defaults."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        entries = [o.strip() for o in raw.split(",") if o.strip()]
        if entries:
            return entries
    return list(_DEFAULT_ORIGINS)


# Computed once at module load — avoids repeated env look-ups per request.
_ALLOWED_ORIGINS: list[str] = _build_allowed_origins()


def _origin_matches(origin: str, allowed: list[str]) -> bool:
    """
    Return True if *origin* matches any pattern in *allowed*.

    Comparison is performed on the scheme+host portion (no path).  Patterns
    may use a ``*.`` wildcard prefix, e.g. ``https://*.jayprajapati.dev``.
    The wildcard matches exactly ONE DNS label (no dots), so:
    - ``https://app.jayprajapati.dev``       → MATCH
    - ``https://evil.com.jayprajapati.dev``  → NO MATCH
    """
    try:
        o = urlparse(origin.rstrip("/").lower())
        for pattern in allowed:
            p = urlparse(pattern.rstrip("/").lower())
            # Scheme must match.
            if p.scheme != o.scheme:
                continue
            # Exact netloc match (covers literal hosts and explicit ports).
            if p.netloc == o.netloc:
                return True
            # Wildcard subdomain: *.example.com
            if p.hostname and p.hostname.startswith("*."):
                suffix = p.hostname[2:]  # e.g. "jayprajapati.dev"
                # Ports must agree (both absent, or both the same value).
                if p.port != o.port:
                    continue
                oh = o.hostname or ""
                # Origin host must end with ".<suffix>".
                if oh.endswith("." + suffix):
                    subdomain = oh[: -(len(suffix) + 1)]
                    # The subdomain part must be a single label (no dots).
                    if subdomain and "." not in subdomain:
                        return True
    except Exception:
        pass
    return False


def _extract_origin_from_referer(referer: str) -> str:
    """
    Extract scheme+host (i.e. the effective origin) from a full Referer URL.

    Returns an empty string if the Referer is malformed.
    """
    try:
        parsed = urlparse(referer)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return ""


class OriginRefererMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates the Origin (or Referer) header against the
    configured allowlist.

    Requests without either header are allowed through — they are assumed to be
    server-to-server or tooling calls where no browser is involved.

    Usage::

        app.add_middleware(OriginRefererMiddleware)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Always pass pre-flight and exempt infra paths.
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        origin = (request.headers.get("origin") or "").strip()
        referer = (request.headers.get("referer") or "").strip()

        # No browser-set header → allow (server-to-server / curl).
        if not origin and not referer:
            return await call_next(request)

        # Prefer Origin; fall back to extracting the origin from Referer.
        effective_origin = origin or _extract_origin_from_referer(referer)

        if not effective_origin:
            # Referer present but unparseable → block to be safe.
            logger.warning(
                "Blocked request: unparseable Referer=%r, path=%r",
                referer,
                request.url.path,
            )
            return Response(
                status_code=403,
                content="Forbidden: unparseable origin",
                media_type="text/plain",
            )

        if not _origin_matches(effective_origin, _ALLOWED_ORIGINS):
            logger.warning(
                "Blocked request from disallowed origin=%r, path=%r",
                effective_origin,
                request.url.path,
            )
            return Response(
                status_code=403,
                content="Forbidden: origin not allowed",
                media_type="text/plain",
            )

        return await call_next(request)
