"""
pact_ax/api/middleware.py
─────────────────────────
Request logging and error-handling middleware for the PACT-AX REST API.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


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
