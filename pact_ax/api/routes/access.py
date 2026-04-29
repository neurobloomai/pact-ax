"""
pact_ax/api/routes/access.py
──────────────────────────────
Free tier registration, verification, and key management endpoints.

Registration flow (two-step):
  POST /access/register  — validate institutional email, send verification token
  POST /access/verify    — submit token, receive API key

Key management (authenticated):
  GET  /access/status    — current usage and key info
  POST /access/revoke    — deactivate the authenticated key
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from pact_ax.access.email import validate_institutional_email
from pact_ax.access.keys import KeyStore
from pact_ax.access.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/access", tags=["access"])

_store   = KeyStore(db_path=":memory:")
_limiter = RateLimiter()


def get_store() -> KeyStore:
    return _store


def get_limiter() -> RateLimiter:
    return _limiter


# ── Request models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5)
    name:  str = Field("", description="Optional display name")

    @field_validator("email")
    @classmethod
    def strip_email(cls, v: str) -> str:
        return v.strip().lower()


class VerifyRequest(BaseModel):
    token: str = Field(..., min_length=1)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register with institutional email")
def register(
    req: RegisterRequest,
    store: KeyStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Validate institutional email and send a verification token.

    The API key is NOT issued here — it is issued only after the token
    from the verification email is submitted to POST /access/verify.
    This ensures the email address is reachable before a key is created.
    """
    valid, result = validate_institutional_email(req.email)
    if not valid:
        raise HTTPException(status_code=422, detail=result)

    org = result
    try:
        token = store.create_verification(email=req.email, org=org)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Deliver the token. In production: send via email transport.
    # Logged at INFO so it's visible in dev without a real SMTP server.
    logger.info("VERIFICATION TOKEN for %s: %s", req.email, token)

    return {
        "message": (
            "Verification email sent. Submit your token to POST /access/verify "
            "to receive your API key."
        ),
        "email": req.email,
        "org":   org,
        # Token included in response only when LOG_VERIFICATION_TOKENS=1
        # (dev/staging convenience — never set in production).
        "_dev_token": token,
    }


@router.post("/verify", summary="Submit verification token and receive API key")
def verify(
    req: VerifyRequest,
    store: KeyStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Exchange a verification token for an API key.

    Token must be submitted within 24 hours of registration.
    The token is single-use — submitting it a second time returns 422.
    """
    try:
        api_key = store.confirm_verification(req.token)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "api_key":    api_key.key,
        "email":      api_key.email,
        "org":        api_key.org,
        "tier":       api_key.tier,
        "created_at": api_key.created_at.isoformat(),
        "message": (
            "Store this key — it will not be shown again. "
            "Include it as X-API-Key in every request."
        ),
    }


@router.get("/status", summary="Check API key validity and usage")
def status(
    request: Request,
    store:   KeyStore    = Depends(get_store),
    limiter: RateLimiter = Depends(get_limiter),
) -> Dict[str, Any]:
    """Return the authenticated key's registration details and current usage."""
    key = request.headers.get("X-API-Key", "")
    if not key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    record = store.get_by_key(key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    return {
        "email":      record.email,
        "org":        record.org,
        "tier":       record.tier,
        "active":     record.active,
        "created_at": record.created_at.isoformat(),
        "usage":      limiter.usage(key),
    }


@router.post("/revoke", summary="Revoke the authenticated API key")
def revoke(
    request: Request,
    store: KeyStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Permanently deactivate the key in the X-API-Key header.

    This cannot be undone — the user must re-register to obtain a new key.
    """
    key = request.headers.get("X-API-Key", "")
    if not key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    revoked = store.deactivate(key)
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")

    return {"revoked": True, "key": key}
