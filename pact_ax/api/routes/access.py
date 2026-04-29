"""
pact_ax/api/routes/access.py
──────────────────────────────
Free tier registration and key status endpoints.

POST /access/register — validate institutional email, issue API key
GET  /access/status   — check key validity and current usage (requires X-API-Key)
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from pact_ax.access.email import validate_institutional_email
from pact_ax.access.keys import KeyStore
from pact_ax.access.rate_limit import RateLimiter

router = APIRouter(prefix="/access", tags=["access"])

# Module-level singletons — replaced in tests via dependency override
_store   = KeyStore(db_path=":memory:")   # overridden to a real path in production
_limiter = RateLimiter()


def get_store() -> KeyStore:
    return _store


def get_limiter() -> RateLimiter:
    return _limiter


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5)
    name:  str = Field("", description="Optional display name")

    @field_validator("email")
    @classmethod
    def strip_email(cls, v: str) -> str:
        return v.strip().lower()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register for a free tier API key")
def register(
    req: RegisterRequest,
    store: KeyStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Validate institutional email and issue an API key.

    Rejects free consumer providers (Gmail, Yahoo, Outlook, etc.).
    Returns the key immediately — store it, it won't be shown again.
    """
    valid, result = validate_institutional_email(req.email)
    if not valid:
        raise HTTPException(status_code=422, detail=result)

    org = result  # validate_institutional_email returns domain on success
    try:
        api_key = store.create(email=req.email, org=org)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "api_key":    api_key.key,
        "email":      api_key.email,
        "org":        api_key.org,
        "tier":       api_key.tier,
        "created_at": api_key.created_at.isoformat(),
        "message": (
            "Store this key — it won't be shown again. "
            "Include it as X-API-Key in every request."
        ),
    }


@router.get("/status", summary="Check API key validity and usage")
def status(
    request: Request,
    store:   KeyStore    = Depends(get_store),
    limiter: RateLimiter = Depends(get_limiter),
) -> Dict[str, Any]:
    """
    Return the authenticated key's registration details and current usage.
    Requires X-API-Key header (same as all other endpoints).
    """
    key = request.headers.get("X-API-Key", "")
    if not key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    record = store.get_by_key(key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    usage = limiter.usage(key)
    return {
        "email":      record.email,
        "org":        record.org,
        "tier":       record.tier,
        "active":     record.active,
        "created_at": record.created_at.isoformat(),
        "usage":      usage,
    }
