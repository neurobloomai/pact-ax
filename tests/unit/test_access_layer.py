"""
tests/unit/test_access_layer.py
─────────────────────────────────
Tests for email validation, key store, rate limiter, and /access/* endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from pact_ax.access.email import validate_institutional_email
from pact_ax.access.keys import KeyStore, generate_key
from pact_ax.access.rate_limit import RateLimiter
from pact_ax.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from pact_ax.api.routes.access import router as access_router, get_store, get_limiter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(store: KeyStore, limiter: RateLimiter) -> TestClient:
    """Spin up a minimal app with the access layer wired in."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(APIKeyMiddleware, store=store, limiter=limiter)
    app.include_router(access_router)

    # Wire singletons
    app.dependency_overrides[get_store]   = lambda: store
    app.dependency_overrides[get_limiter] = lambda: limiter

    # Protected sentinel route
    @app.get("/protected")
    def protected():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def store():
    return KeyStore(db_path=":memory:")


@pytest.fixture()
def limiter():
    return RateLimiter(hourly_limit=5, daily_limit=10)


@pytest.fixture()
def client(store, limiter):
    return _make_app(store, limiter)


# ── Email validation ──────────────────────────────────────────────────────────

class TestEmailValidation:

    def test_valid_institutional_email(self):
        ok, result = validate_institutional_email("alice@mit.edu")
        assert ok is True
        assert result == "mit.edu"

    def test_valid_company_email(self):
        ok, result = validate_institutional_email("bob@neurobloom.ai")
        assert ok is True
        assert result == "neurobloom.ai"

    def test_gmail_rejected(self):
        ok, reason = validate_institutional_email("alice@gmail.com")
        assert ok is False
        assert "gmail.com" in reason

    def test_yahoo_rejected(self):
        ok, reason = validate_institutional_email("alice@yahoo.com")
        assert ok is False

    def test_outlook_rejected(self):
        ok, reason = validate_institutional_email("alice@outlook.com")
        assert ok is False

    def test_hotmail_rejected(self):
        ok, reason = validate_institutional_email("alice@hotmail.com")
        assert ok is False

    def test_icloud_rejected(self):
        ok, reason = validate_institutional_email("alice@icloud.com")
        assert ok is False

    def test_invalid_format_rejected(self):
        ok, reason = validate_institutional_email("not-an-email")
        assert ok is False
        assert "format" in reason.lower()

    def test_missing_tld_rejected(self):
        ok, _ = validate_institutional_email("alice@mit")
        assert ok is False

    def test_strips_whitespace(self):
        ok, domain = validate_institutional_email("  alice@stanford.edu  ")
        assert ok is True
        assert domain == "stanford.edu"

    def test_case_insensitive(self):
        ok, domain = validate_institutional_email("Alice@MIT.EDU")
        assert ok is True
        assert domain == "mit.edu"


# ── Key generation and store ──────────────────────────────────────────────────

class TestKeyGeneration:

    def test_key_has_correct_prefix(self):
        key = generate_key()
        assert key.startswith("pax_")

    def test_key_length(self):
        key = generate_key()
        assert len(key) == 4 + 32  # "pax_" + 32 hex chars

    def test_keys_are_unique(self):
        keys = {generate_key() for _ in range(100)}
        assert len(keys) == 100


class TestKeyStore:

    def test_create_returns_api_key(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        assert k.key.startswith("pax_")
        assert k.email == "alice@mit.edu"
        assert k.org == "mit.edu"
        assert k.tier == "free"
        assert k.active is True

    def test_get_by_key_found(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        found = store.get_by_key(k.key)
        assert found is not None
        assert found.email == "alice@mit.edu"

    def test_get_by_key_not_found(self, store):
        assert store.get_by_key("pax_nonexistent") is None

    def test_get_by_email_found(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        found = store.get_by_email("alice@mit.edu")
        assert found is not None
        assert found.key == k.key

    def test_duplicate_email_raises(self, store):
        store.create(email="alice@mit.edu", org="mit.edu")
        with pytest.raises(ValueError, match="already registered"):
            store.create(email="alice@mit.edu", org="mit.edu")

    def test_deactivate_key(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        store.deactivate(k.key)
        assert store.get_by_key(k.key) is None  # inactive keys not returned

    def test_deactivate_nonexistent_returns_false(self, store):
        assert store.deactivate("pax_doesnotexist") is False


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestRateLimiter:

    def test_allows_requests_within_limit(self):
        lim = RateLimiter(hourly_limit=5, daily_limit=10)
        for _ in range(5):
            allowed, _ = lim.check("key-a")
            assert allowed is True

    def test_blocks_after_hourly_limit(self):
        lim = RateLimiter(hourly_limit=3, daily_limit=100)
        for _ in range(3):
            lim.check("key-a")
        allowed, headers = lim.check("key-a")
        assert allowed is False
        assert headers["RateLimit-Remaining"] == "0"

    def test_blocks_after_daily_limit(self):
        lim = RateLimiter(hourly_limit=1000, daily_limit=3)
        for _ in range(3):
            lim.check("key-a")
        allowed, _ = lim.check("key-a")
        assert allowed is False

    def test_rate_limit_headers_present(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        _, headers = lim.check("key-a")
        assert "RateLimit-Limit" in headers
        assert "RateLimit-Remaining" in headers
        assert "RateLimit-Reset" in headers
        assert "X-RateLimit-Limit-Day" in headers
        assert "X-RateLimit-Remaining-Day" in headers

    def test_remaining_decrements(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        _, h1 = lim.check("key-a")
        _, h2 = lim.check("key-a")
        assert int(h2["RateLimit-Remaining"]) < int(h1["RateLimit-Remaining"])

    def test_different_keys_independent(self):
        lim = RateLimiter(hourly_limit=2, daily_limit=10)
        lim.check("key-a")
        lim.check("key-a")
        lim.check("key-a")  # key-a exhausted
        allowed, _ = lim.check("key-b")
        assert allowed is True

    def test_usage_returns_counts(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        lim.check("key-a")
        lim.check("key-a")
        usage = lim.usage("key-a")
        assert usage["hourly_used"] == 2
        assert usage["daily_used"] == 2
        assert usage["hourly_limit"] == 10
        assert usage["daily_limit"] == 100

    def test_usage_unknown_key_returns_zeros(self):
        lim = RateLimiter()
        usage = lim.usage("never-seen")
        assert usage["hourly_used"] == 0
        assert usage["daily_used"] == 0


# ── /access/register endpoint ─────────────────────────────────────────────────

class TestRegisterEndpoint:

    def test_institutional_email_returns_200(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert r.status_code == 200

    def test_response_contains_api_key(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        body = r.json()
        assert "api_key" in body
        assert body["api_key"].startswith("pax_")

    def test_response_contains_org_and_tier(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        body = r.json()
        assert body["org"] == "mit.edu"
        assert body["tier"] == "free"

    def test_free_email_rejected_422(self, client):
        r = client.post("/access/register", json={"email": "alice@gmail.com"})
        assert r.status_code == 422

    def test_duplicate_email_rejected_409(self, client):
        client.post("/access/register", json={"email": "alice@mit.edu"})
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert r.status_code == 409

    def test_register_is_open_path_no_key_needed(self, client):
        # Should work without X-API-Key
        r = client.post("/access/register", json={"email": "bob@stanford.edu"})
        assert r.status_code == 200


# ── /access/status endpoint ───────────────────────────────────────────────────

class TestStatusEndpoint:

    def test_valid_key_returns_200(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        r = client.get("/access/status", headers={"X-API-Key": reg["api_key"]})
        assert r.status_code == 200

    def test_status_contains_usage(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        r = client.get("/access/status", headers={"X-API-Key": reg["api_key"]})
        body = r.json()
        assert "usage" in body
        assert "hourly_used" in body["usage"]
        assert "daily_used" in body["usage"]

    def test_missing_key_returns_401(self, client):
        r = client.get("/access/status")
        assert r.status_code == 401

    def test_invalid_key_returns_401(self, client):
        r = client.get("/access/status", headers={"X-API-Key": "pax_invalid"})
        assert r.status_code == 401


# ── Middleware enforcement ────────────────────────────────────────────────────

class TestAPIKeyMiddleware:

    def test_protected_route_requires_key(self, client):
        r = client.get("/protected")
        assert r.status_code == 401

    def test_protected_route_with_valid_key(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        r = client.get("/protected", headers={"X-API-Key": reg["api_key"]})
        assert r.status_code == 200

    def test_invalid_key_blocked(self, client):
        r = client.get("/protected", headers={"X-API-Key": "pax_fake"})
        assert r.status_code == 401

    def test_health_is_open(self, client):
        # /health not in this minimal app but open paths tested via register
        r = client.post("/access/register", json={"email": "check@open.edu"})
        assert r.status_code == 200  # no key required

    def test_rate_limit_headers_on_success(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        r = client.get("/protected", headers={"X-API-Key": reg["api_key"]})
        assert "RateLimit-Limit" in r.headers
        assert "RateLimit-Remaining" in r.headers

    def test_rate_limit_exceeded_returns_429(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        key = reg["api_key"]
        # Exhaust the 5-request hourly limit
        for _ in range(5):
            client.get("/protected", headers={"X-API-Key": key})
        r = client.get("/protected", headers={"X-API-Key": key})
        assert r.status_code == 429

    def test_429_body_contains_limit_info(self, client):
        reg = client.post("/access/register", json={"email": "alice@mit.edu"}).json()
        key = reg["api_key"]
        for _ in range(5):
            client.get("/protected", headers={"X-API-Key": key})
        r = client.get("/protected", headers={"X-API-Key": key})
        body = r.json()
        assert "hourly_limit" in body
        assert "reset_at" in body
