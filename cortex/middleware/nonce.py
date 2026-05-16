"""Signed-nonce request token middleware and session endpoint.

This is defense-in-depth only. A determined attacker with server-side access
to set headers can bypass this. The real security boundaries are JWT (P0.4)
and admin key (P0.3). This layer raises the cost of casual Postman/curl abuse.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import OrderedDict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_NONCES = 100_000
_TOKEN_TTL_SECONDS = 300  # 5 minutes

_EXEMPT_METHODS = frozenset(["GET", "HEAD", "OPTIONS"])
_EXEMPT_PATHS = frozenset(["/health", "/ready", "/auth/session", "/llm/ping"])

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# Session secrets: user_id -> {session_id: secret}
# Kept small; sessions expire with token TTL.
_SESSION_SECRETS: dict[str, dict[str, str]] = {}

# In-memory LRU of consumed nonces. OrderedDict for LRU eviction.
_consumed_nonces: "OrderedDict[str, float]" = OrderedDict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evict_expired(now: float) -> None:
    """Evict nonces older than TTL and keep size under cap."""
    cutoff = now - _TOKEN_TTL_SECONDS
    # Evict by timestamp
    to_delete = [k for k, ts in list(_consumed_nonces.items()) if ts < cutoff]
    for k in to_delete:
        del _consumed_nonces[k]
    # Also enforce cap
    while len(_consumed_nonces) > _MAX_NONCES:
        _consumed_nonces.popitem(last=False)


def _verify_request_token(token: str, user_id: str) -> bool:
    """Verify and consume a request token. Returns True if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        nonce, ts_str, sig = parts
        ts = int(ts_str)
        now = time.time()
        if abs(now - ts) > _TOKEN_TTL_SECONDS:
            return False  # expired or future

        # Find a valid session secret for this user
        user_sessions = _SESSION_SECRETS.get(user_id, {})
        valid = False
        for secret in user_sessions.values():
            expected = hmac.new(
                secret.encode(),
                f"{nonce}.{ts_str}".encode(),
                hashlib.sha256,
            ).hexdigest()
            if hmac.compare_digest(sig, expected):
                valid = True
                break
        if not valid:
            return False

        # Check single-use
        _evict_expired(now)
        nonce_key = f"{user_id}:{nonce}"
        if nonce_key in _consumed_nonces:
            return False  # replay attempt

        _consumed_nonces[nonce_key] = now
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def issue_session_token(user_id: str) -> dict:
    """Issue a session secret for a user. Returns {session_id, session_secret}."""
    session_id = secrets.token_hex(16)
    secret = secrets.token_hex(32)
    if user_id not in _SESSION_SECRETS:
        _SESSION_SECRETS[user_id] = {}
    _SESSION_SECRETS[user_id][session_id] = secret
    return {"session_id": session_id, "session_secret": secret}


def mint_request_token(session_secret: str) -> str:
    """Mint a single-use request token.

    Token format: ``{nonce}.{timestamp}.{hmac_signature}``

    This is a helper for tests; the frontend uses session_secret directly to
    mint tokens client-side with the same algorithm.
    """
    nonce = secrets.token_hex(16)
    ts = str(int(time.time()))
    sig = hmac.new(
        session_secret.encode(),
        f"{nonce}.{ts}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{nonce}.{ts}.{sig}"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RequestNonceMiddleware(BaseHTTPMiddleware):
    """Middleware that validates X-Request-Token on mutating routes.

    Enabled only when the ``ENABLE_REQUEST_NONCE`` environment variable is set
    to ``1``, ``true``, or ``yes``. This allows rolling out the feature safely
    without breaking existing integrations.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in _EXEMPT_METHODS:
            return await call_next(request)
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Feature gate — opt-in via env var
        if os.getenv("ENABLE_REQUEST_NONCE", "").lower() not in ("1", "true", "yes"):
            return await call_next(request)

        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            # JWT middleware hasn't run or route is exempted. Skip nonce check.
            return await call_next(request)

        token = request.headers.get("X-Request-Token", "")
        if not token:
            return Response(
                content='{"detail": "X-Request-Token header required"}',
                status_code=403,
                media_type="application/json",
            )

        if not _verify_request_token(token, user_id):
            return Response(
                content='{"detail": "Invalid or expired request token"}',
                status_code=403,
                media_type="application/json",
            )

        return await call_next(request)
