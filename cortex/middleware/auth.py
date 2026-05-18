"""JWT authentication middleware."""
from __future__ import annotations

import logging
import os
from typing import Optional

import jwt  # PyJWT
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logger import set_request_context

logger = logging.getLogger(__name__)

# Routes that don't require auth
_EXEMPT_PATHS = frozenset([
    "/",
    "/health",
])

_EXEMPT_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi",
)


def _get_public_key() -> Optional[str]:
    return os.getenv("JWT_PUBLIC_KEY")


def _get_jwks_url() -> Optional[str]:
    return os.getenv("JWT_JWKS_URL")


def _get_hs256_secret() -> Optional[str]:
    return os.getenv("CORTEX_JWT_SECRET") or os.getenv("JWT_SECRET")


def _verify_token(token: str) -> dict:
    """Verify JWT and return claims. Raises ValueError on failure."""
    public_key = _get_public_key()
    jwks_url = _get_jwks_url()
    hs256_secret = _get_hs256_secret()

    if hs256_secret:
        try:
            return jwt.decode(token, hs256_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}") from e

    if not public_key and not jwks_url:
        # Dev mode: if neither is configured, decode without verification.
        # This lets the app work without JWT configured (dev only).
        env = (os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("PYTHON_ENV") or "").lower()
        if env in ("dev", "development", "local"):
            try:
                return jwt.decode(token, options={"verify_signature": False})
            except Exception as e:
                raise ValueError(f"Invalid JWT format: {e}") from e
        raise ValueError(
            "JWT verification not configured: set CORTEX_JWT_SECRET, JWT_PUBLIC_KEY, or JWT_JWKS_URL"
        )

    if public_key:
        try:
            return jwt.decode(token, public_key, algorithms=["RS256", "ES256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}") from e

    # JWKS URL — use PyJWT's PyJWKClient
    if jwks_url:
        try:
            from jwt import PyJWKClient  # type: ignore[attr-defined]
            jwks_client = PyJWKClient(jwks_url, cache_keys=True)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(token, signing_key.key, algorithms=["RS256", "ES256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except Exception as e:
            raise ValueError(f"JWT verification failed: {e}") from e


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates JWT tokens and stamps request.state.user_id."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Exempt certain paths
        if path in _EXEMPT_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        for prefix in _EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"detail": "Authorization header missing or invalid"}',
                status_code=401,
                media_type="application/json",
            )

        token = auth_header[7:]  # Strip "Bearer "
        try:
            claims = _verify_token(token)
            user_id = claims.get("sub")
            if not user_id:
                return Response(
                    content='{"detail": "JWT missing sub claim"}',
                    status_code=401,
                    media_type="application/json",
                )
            request.state.user_id = user_id
            # Update ContextVar so subsequent log records include user_id.
            # request_id was already set by RequestContextMiddleware (outermost).
            request_id = getattr(request.state, "request_id", "")
            set_request_context(request_id, user_id)
        except ValueError as e:
            return Response(
                content=f'{{"detail": "{str(e)}"}}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
