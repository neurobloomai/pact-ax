"""
pact_ax/api/server.py
──────────────────────
PACT-AX FastAPI application.

Usage
─────
    uvicorn pact_ax.api.server:app --reload

Or from Python:

    from pact_ax.api.server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pact_ax.access.keys import KeyStore
from pact_ax.access.rate_limit import RateLimiter
from pact_ax.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from pact_ax.api.routes.access import router as access_router
from pact_ax.api.routes.context_share import router as context_router
from pact_ax.api.routes.state_transfer import router as transfer_router
from pact_ax.api.routes.policy_align import router as policy_router
from pact_ax.api.routes.trust import router as trust_router
from pact_ax.api.routes.story_keeper import router as story_router

# Shared access-layer singletons — same instances used by middleware and routes
_key_store = KeyStore(db_path="access.db")
_limiter   = RateLimiter()

app = FastAPI(
    title="PACT-AX Agent Collaboration API",
    description=(
        "REST layer for PACT-AX primitives: context sharing, "
        "state transfer, policy alignment, and trust scoring between heterogeneous AI agents."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)

# Auth enforcement is on by default. Set PACT_ENFORCE_AUTH=0 to disable
# (e.g. in test environments that test primitives without the access layer).
if os.getenv("PACT_ENFORCE_AUTH", "1") != "0":
    app.add_middleware(APIKeyMiddleware, store=_key_store, limiter=_limiter)

# Bind the shared singletons into the access routes
import pact_ax.api.routes.access as _access_routes
_access_routes._store   = _key_store
_access_routes._limiter = _limiter

app.include_router(access_router)
app.include_router(context_router)
app.include_router(transfer_router)
app.include_router(policy_router)
app.include_router(trust_router)
app.include_router(story_router)


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "pact-ax"})


@app.get("/", tags=["meta"], include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({
        "service": "PACT-AX",
        "docs":    "/docs",
        "health":  "/health",
    })
