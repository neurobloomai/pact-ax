"""
pact_ax/api/middleware.py
─────────────────────────
Request logging, error handling, and API key enforcement for PACT-AX.
"""

import logging
import time
import uuid
from typing import Callable, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths that bypass API key enforcement
_OPEN_PATHS: Set[str] = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/access/register",
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every inbound request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        logger.info(
            "[%s] → %s %s",
            request_id, request.method, request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "[%s] ✗ %s %s — unhandled exception after %.1fms: %s",
                request_id, request.method, request.url.path, elapsed, exc,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "[%s] ← %s %s %d (%.1fms)",
            request_id, request.method, request.url.path,
            response.status_code, elapsed,
        )
        response.headers["X-Request-ID"] = request_id
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Enforce API key authentication and rate limiting on all protected routes.

    Open paths (/, /health, /docs, /access/register) pass through without a
    key. Everything else requires a valid X-API-Key header and counts against
    the key's hourly and daily windows.

    On success, sets request.state.api_key, .org, and .tier so routes and
    the instrumentation layer can identify the caller without re-querying.
    """

    def __init__(self, app, store, limiter, open_paths: Set[str] = _OPEN_PATHS):
        super().__init__(app)
        self._store   = store
        self._limiter = limiter
        self._open    = open_paths

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._open:
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if not key:
            return JSONResponse(
                status_code=401,
                content={"detail": "X-API-Key header required"},
            )

        record = self._store.get_by_key(key)
        if record is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or inactive API key"},
            )

        allowed, rl_headers = self._limiter.check(key)
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail":       "Rate limit exceeded",
                    "hourly_limit": int(rl_headers["RateLimit-Limit"]),
                    "daily_limit":  int(rl_headers["X-RateLimit-Limit-Day"]),
                    "reset_at":     rl_headers["RateLimit-Reset"],
                },
            )
            for k, v in rl_headers.items():
                response.headers[k] = v
            return response

        # Attach caller identity — instrumentation layer reads these
        request.state.api_key = key
        request.state.org     = record.org
        request.state.tier    = record.tier

        response = await call_next(request)

        for k, v in rl_headers.items():
            response.headers[k] = v

        return response
