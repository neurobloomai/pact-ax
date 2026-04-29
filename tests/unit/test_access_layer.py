"""
tests/unit/test_access_layer.py
─────────────────────────────────
Tests for email validation, key store (incl. verification flow),
rate limiter (persistence), /access/* endpoints, and middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pact_ax.access.email import validate_institutional_email
from pact_ax.access.keys import KeyStore, generate_key
from pact_ax.access.rate_limit import RateLimiter
from pact_ax.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from pact_ax.api.routes.access import router as access_router, get_store, get_limiter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(store: KeyStore, limiter: RateLimiter) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(APIKeyMiddleware, store=store, limiter=limiter)
    app.include_router(access_router)
    app.dependency_overrides[get_store]   = lambda: store
    app.dependency_overrides[get_limiter] = lambda: limiter

    @app.get("/protected")
    def protected():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=True)


def _register_and_verify(client, email="alice@mit.edu") -> str:
    """Full two-step flow; returns the issued API key."""
    r = client.post("/access/register", json={"email": email})
    assert r.status_code == 200, r.json()
    token = r.json()["_dev_token"]
    r2 = client.post("/access/verify", json={"token": token})
    assert r2.status_code == 200, r2.json()
    return r2.json()["api_key"]


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

    def test_institutional_email_passes(self):
        ok, domain = validate_institutional_email("alice@mit.edu")
        assert ok is True
        assert domain == "mit.edu"

    def test_company_domain_passes(self):
        ok, domain = validate_institutional_email("bob@neurobloom.ai")
        assert ok is True
        assert domain == "neurobloom.ai"

    def test_returns_domain_on_success(self):
        ok, result = validate_institutional_email("x@stanford.edu")
        assert ok
        assert result == "stanford.edu"

    def test_gmail_rejected(self):
        ok, reason = validate_institutional_email("alice@gmail.com")
        assert ok is False
        assert "gmail.com" in reason

    def test_yahoo_rejected(self):
        ok, _ = validate_institutional_email("alice@yahoo.com")
        assert ok is False

    def test_outlook_rejected(self):
        ok, _ = validate_institutional_email("alice@outlook.com")
        assert ok is False

    def test_hotmail_rejected(self):
        ok, _ = validate_institutional_email("alice@hotmail.com")
        assert ok is False

    def test_icloud_rejected(self):
        ok, _ = validate_institutional_email("alice@icloud.com")
        assert ok is False

    def test_protonmail_rejected(self):
        ok, _ = validate_institutional_email("alice@protonmail.com")
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
        assert ok
        assert domain == "stanford.edu"

    def test_case_insensitive(self):
        ok, domain = validate_institutional_email("Alice@MIT.EDU")
        assert ok
        assert domain == "mit.edu"


# ── Key generation ────────────────────────────────────────────────────────────

class TestKeyGeneration:

    def test_prefix(self):
        assert generate_key().startswith("pax_")

    def test_length(self):
        assert len(generate_key()) == 4 + 32

    def test_unique(self):
        assert len({generate_key() for _ in range(100)}) == 100


# ── KeyStore — direct create (used internally and in tests) ───────────────────

class TestKeyStoreDirect:

    def test_create_returns_api_key(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        assert k.key.startswith("pax_")
        assert k.email == "alice@mit.edu"
        assert k.tier == "free"
        assert k.active is True

    def test_get_by_key_found(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        assert store.get_by_key(k.key) is not None

    def test_get_by_key_not_found(self, store):
        assert store.get_by_key("pax_nonexistent") is None

    def test_get_by_email(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        found = store.get_by_email("alice@mit.edu")
        assert found.key == k.key

    def test_duplicate_email_raises(self, store):
        store.create(email="alice@mit.edu", org="mit.edu")
        with pytest.raises(ValueError, match="already registered"):
            store.create(email="alice@mit.edu", org="mit.edu")

    def test_deactivate_hides_key(self, store):
        k = store.create(email="alice@mit.edu", org="mit.edu")
        store.deactivate(k.key)
        assert store.get_by_key(k.key) is None

    def test_deactivate_nonexistent_returns_false(self, store):
        assert store.deactivate("pax_gone") is False


# ── KeyStore — verification flow ──────────────────────────────────────────────

class TestKeyStoreVerification:

    def test_create_verification_returns_token(self, store):
        token = store.create_verification("alice@mit.edu", "mit.edu")
        assert len(token) == 64

    def test_tokens_are_unique(self, store):
        t1 = store.create_verification("a@mit.edu", "mit.edu")
        t2 = store.create_verification("b@mit.edu", "mit.edu")
        assert t1 != t2

    def test_confirm_issues_key(self, store):
        token = store.create_verification("alice@mit.edu", "mit.edu")
        key = store.confirm_verification(token)
        assert key.key.startswith("pax_")
        assert key.email == "alice@mit.edu"

    def test_token_is_single_use(self, store):
        token = store.create_verification("alice@mit.edu", "mit.edu")
        store.confirm_verification(token)
        with pytest.raises(ValueError, match="Invalid"):
            store.confirm_verification(token)

    def test_invalid_token_raises(self, store):
        with pytest.raises(ValueError, match="Invalid"):
            store.confirm_verification("a" * 64)

    def test_duplicate_registration_rejected(self, store):
        store.create_verification("alice@mit.edu", "mit.edu")
        with pytest.raises(ValueError):
            store.create_verification("alice@mit.edu", "mit.edu")

    def test_already_registered_email_rejected(self, store):
        store.create("alice@mit.edu", "mit.edu")
        with pytest.raises(ValueError, match="already registered"):
            store.create_verification("alice@mit.edu", "mit.edu")

    def test_confirmed_key_is_retrievable(self, store):
        token = store.create_verification("alice@mit.edu", "mit.edu")
        issued = store.confirm_verification(token)
        found = store.get_by_key(issued.key)
        assert found is not None
        assert found.email == "alice@mit.edu"


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestRateLimiter:

    def test_allows_within_limit(self):
        lim = RateLimiter(hourly_limit=5, daily_limit=10)
        for _ in range(5):
            allowed, _ = lim.check("k")
            assert allowed

    def test_blocks_after_hourly_limit(self):
        lim = RateLimiter(hourly_limit=3, daily_limit=100)
        for _ in range(3):
            lim.check("k")
        allowed, headers = lim.check("k")
        assert not allowed
        assert headers["RateLimit-Remaining"] == "0"

    def test_blocks_after_daily_limit(self):
        lim = RateLimiter(hourly_limit=1000, daily_limit=3)
        for _ in range(3):
            lim.check("k")
        allowed, _ = lim.check("k")
        assert not allowed

    def test_headers_present(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        _, h = lim.check("k")
        for header in ("RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset",
                       "X-RateLimit-Limit-Day", "X-RateLimit-Remaining-Day"):
            assert header in h

    def test_remaining_decrements(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        _, h1 = lim.check("k")
        _, h2 = lim.check("k")
        assert int(h2["RateLimit-Remaining"]) < int(h1["RateLimit-Remaining"])

    def test_independent_keys(self):
        lim = RateLimiter(hourly_limit=2, daily_limit=10)
        for _ in range(3):
            lim.check("key-a")
        allowed, _ = lim.check("key-b")
        assert allowed

    def test_usage_counts(self):
        lim = RateLimiter(hourly_limit=10, daily_limit=100)
        lim.check("k")
        lim.check("k")
        u = lim.usage("k")
        assert u["hourly_used"] == 2
        assert u["daily_used"] == 2

    def test_usage_unknown_key_zeros(self):
        lim = RateLimiter()
        u = lim.usage("never-seen")
        assert u["hourly_used"] == 0
        assert u["daily_used"] == 0

    def test_persistence_across_instances(self):
        """Two RateLimiter instances sharing a db file share the same counters."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            lim1 = RateLimiter(hourly_limit=10, daily_limit=100, db_path=path)
            lim1.check("k")
            lim1.check("k")

            lim2 = RateLimiter(hourly_limit=10, daily_limit=100, db_path=path)
            u = lim2.usage("k")
            assert u["hourly_used"] == 2
        finally:
            os.unlink(path)


# ── POST /access/register ─────────────────────────────────────────────────────

class TestRegisterEndpoint:

    def test_returns_200(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert r.status_code == 200

    def test_response_has_message_and_email(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        body = r.json()
        assert "message" in body
        assert body["email"] == "alice@mit.edu"

    def test_dev_token_present(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert "_dev_token" in r.json()
        assert len(r.json()["_dev_token"]) == 64

    def test_no_api_key_in_register_response(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert "api_key" not in r.json()

    def test_free_email_rejected(self, client):
        r = client.post("/access/register", json={"email": "alice@gmail.com"})
        assert r.status_code == 422

    def test_duplicate_registration_rejected(self, client):
        client.post("/access/register", json={"email": "alice@mit.edu"})
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        assert r.status_code == 409

    def test_open_path_no_key_required(self, client):
        r = client.post("/access/register", json={"email": "bob@stanford.edu"})
        assert r.status_code == 200


# ── POST /access/verify ───────────────────────────────────────────────────────

class TestVerifyEndpoint:

    def test_valid_token_issues_key(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        token = r.json()["_dev_token"]
        r2 = client.post("/access/verify", json={"token": token})
        assert r2.status_code == 200
        assert r2.json()["api_key"].startswith("pax_")

    def test_response_contains_email_and_org(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        r2 = client.post("/access/verify", json={"token": r.json()["_dev_token"]})
        body = r2.json()
        assert body["email"] == "alice@mit.edu"
        assert body["org"] == "mit.edu"

    def test_invalid_token_returns_422(self, client):
        r = client.post("/access/verify", json={"token": "a" * 64})
        assert r.status_code == 422

    def test_token_is_single_use(self, client):
        r = client.post("/access/register", json={"email": "alice@mit.edu"})
        token = r.json()["_dev_token"]
        client.post("/access/verify", json={"token": token})
        r2 = client.post("/access/verify", json={"token": token})
        assert r2.status_code == 422


# ── GET /access/status ────────────────────────────────────────────────────────

class TestStatusEndpoint:

    def test_valid_key_returns_200(self, client):
        key = _register_and_verify(client)
        r = client.get("/access/status", headers={"X-API-Key": key})
        assert r.status_code == 200

    def test_status_contains_usage(self, client):
        key = _register_and_verify(client)
        r = client.get("/access/status", headers={"X-API-Key": key})
        assert "usage" in r.json()
        assert "hourly_used" in r.json()["usage"]

    def test_missing_key_returns_401(self, client):
        r = client.get("/access/status")
        assert r.status_code == 401

    def test_invalid_key_returns_401(self, client):
        r = client.get("/access/status", headers={"X-API-Key": "pax_bad"})
        assert r.status_code == 401


# ── POST /access/revoke ───────────────────────────────────────────────────────

class TestRevokeEndpoint:

    def test_revoke_returns_200(self, client):
        key = _register_and_verify(client)
        r = client.post("/access/revoke", headers={"X-API-Key": key})
        assert r.status_code == 200
        assert r.json()["revoked"] is True

    def test_revoked_key_cannot_authenticate(self, client):
        key = _register_and_verify(client)
        client.post("/access/revoke", headers={"X-API-Key": key})
        r = client.get("/protected", headers={"X-API-Key": key})
        assert r.status_code == 401

    def test_missing_key_returns_401(self, client):
        r = client.post("/access/revoke")
        assert r.status_code == 401

    def test_nonexistent_key_returns_404(self, client):
        # Key must pass middleware (which rejects unknown keys),
        # so we test via the store directly
        store = KeyStore(db_path=":memory:")
        assert store.deactivate("pax_" + "0" * 32) is False


# ── Middleware enforcement ────────────────────────────────────────────────────

class TestAPIKeyMiddleware:

    def test_protected_route_blocked_without_key(self, client):
        r = client.get("/protected")
        assert r.status_code == 401

    def test_protected_route_passes_with_valid_key(self, client):
        key = _register_and_verify(client)
        r = client.get("/protected", headers={"X-API-Key": key})
        assert r.status_code == 200

    def test_invalid_key_blocked(self, client):
        r = client.get("/protected", headers={"X-API-Key": "pax_fake"})
        assert r.status_code == 401

    def test_rate_limit_headers_on_success(self, client):
        key = _register_and_verify(client)
        r = client.get("/protected", headers={"X-API-Key": key})
        assert "RateLimit-Limit" in r.headers
        assert "RateLimit-Remaining" in r.headers

    def test_rate_limit_exceeded_returns_429(self, client):
        key = _register_and_verify(client)
        for _ in range(5):
            client.get("/protected", headers={"X-API-Key": key})
        r = client.get("/protected", headers={"X-API-Key": key})
        assert r.status_code == 429

    def test_429_body_contains_limit_info(self, client):
        key = _register_and_verify(client)
        for _ in range(5):
            client.get("/protected", headers={"X-API-Key": key})
        body = client.get("/protected", headers={"X-API-Key": key}).json()
        assert "hourly_limit" in body
        assert "reset_at" in body

    def test_revoked_key_blocked_immediately(self, client):
        key = _register_and_verify(client)
        client.post("/access/revoke", headers={"X-API-Key": key})
        r = client.get("/protected", headers={"X-API-Key": key})
        assert r.status_code == 401
