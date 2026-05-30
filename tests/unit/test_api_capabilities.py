"""
tests/unit/test_api_capabilities.py
─────────────────────────────────────
Tests for /capabilities/* REST endpoints.

Run with:  pytest tests/unit/test_api_capabilities.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.capabilities as cap_module
from pact_ax.primitives.capability_registry import CapabilityRegistry

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def fresh_registry(tmp_path):
    """Replace the module-level registry with a fresh in-memory one before each test."""
    cap_module._registry = CapabilityRegistry(":memory:")
    yield
    cap_module._registry = CapabilityRegistry(":memory:")


def register(agent_id, skill, description="", tags=None, version="1.0"):
    return client.post("/capabilities/register", json={
        "agent_id": agent_id, "skill": skill,
        "description": description, "tags": tags or [], "version": version
    })


# ── POST /capabilities/register ───────────────────────────────────────────────

class TestRegisterEndpoint:

    def test_returns_200(self):
        r = register("agent-a", "contract_review")
        assert r.status_code == 200

    def test_response_contains_fields(self):
        r = register("agent-a", "contract_review", description="NDA review", tags=["legal"])
        body = r.json()
        assert body["agent_id"]    == "agent-a"
        assert body["skill"]       == "contract_review"
        assert body["description"] == "NDA review"
        assert "legal" in body["tags"]
        assert body["registered"]  is True

    def test_upsert_on_duplicate(self):
        register("agent-a", "contract_review", description="old")
        r = register("agent-a", "contract_review", description="new")
        assert r.status_code == 200
        # Only one entry
        caps = client.get("/capabilities/agent-a").json()
        assert len(caps["capabilities"]) == 1
        assert caps["capabilities"][0]["description"] == "new"

    def test_missing_agent_id_returns_422(self):
        r = client.post("/capabilities/register", json={"skill": "x"})
        assert r.status_code == 422

    def test_missing_skill_returns_422(self):
        r = client.post("/capabilities/register", json={"agent_id": "agent-a"})
        assert r.status_code == 422


# ── GET /capabilities/{agent_id} ─────────────────────────────────────────────

class TestGetAgentCapabilities:

    def test_returns_registered_skills(self):
        register("agent-a", "contract_review")
        register("agent-a", "tax_analysis")
        r = client.get("/capabilities/agent-a")
        assert r.status_code == 200
        skills = {c["skill"] for c in r.json()["capabilities"]}
        assert skills == {"contract_review", "tax_analysis"}

    def test_empty_for_unknown_agent(self):
        r = client.get("/capabilities/nobody")
        assert r.status_code == 200
        assert r.json()["count"] == 0


# ── GET /capabilities/skills ──────────────────────────────────────────────────

class TestListSkills:

    def test_returns_unique_skills(self):
        register("agent-a", "contract_review")
        register("agent-b", "contract_review")
        register("agent-a", "tax_analysis")
        r = client.get("/capabilities/skills")
        assert r.status_code == 200
        skills = r.json()["skills"]
        assert sorted(skills) == ["contract_review", "tax_analysis"]


# ── GET /capabilities ─────────────────────────────────────────────────────────

class TestListAll:

    def test_returns_all_entries(self):
        register("agent-a", "contract_review")
        register("agent-b", "tax_analysis")
        r = client.get("/capabilities")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_empty_registry(self):
        r = client.get("/capabilities")
        assert r.json()["count"] == 0


# ── POST /capabilities/find ───────────────────────────────────────────────────

class TestFindCapable:

    def test_finds_agents_for_skill(self):
        register("agent-a", "contract_review")
        register("agent-b", "contract_review")
        r = client.post("/capabilities/find", json={"skill": "contract_review"})
        assert r.status_code == 200
        ids = {c["agent_id"] for c in r.json()["candidates"]}
        assert ids == {"agent-a", "agent-b"}

    def test_empty_for_unknown_skill(self):
        r = client.post("/capabilities/find", json={"skill": "nonexistent"})
        assert r.json()["count"] == 0

    def test_missing_skill_returns_422(self):
        r = client.post("/capabilities/find", json={})
        assert r.status_code == 422


# ── POST /capabilities/search ─────────────────────────────────────────────────

class TestSearch:

    def test_fuzzy_match(self):
        register("agent-a", "contract_review", description="Reviews NDAs", tags=["legal"])
        r = client.post("/capabilities/search", json={"query": "NDA"})
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_no_match_returns_empty(self):
        r = client.post("/capabilities/search", json={"query": "zzznomatch"})
        assert r.json()["count"] == 0

    def test_missing_query_returns_422(self):
        r = client.post("/capabilities/search", json={})
        assert r.status_code == 422


# ── DELETE /capabilities/{agent_id}/{skill} ───────────────────────────────────

class TestDeregisterSkill:

    def test_removes_skill(self):
        register("agent-a", "contract_review")
        r = client.delete("/capabilities/agent-a/contract_review")
        assert r.status_code == 200
        assert r.json()["removed"] is True
        caps = client.get("/capabilities/agent-a").json()
        assert caps["count"] == 0

    def test_404_for_missing(self):
        r = client.delete("/capabilities/nobody/nonexistent")
        assert r.status_code == 404


# ── DELETE /capabilities/{agent_id} ──────────────────────────────────────────

class TestDeregisterAgent:

    def test_removes_all_skills(self):
        register("agent-a", "contract_review")
        register("agent-a", "tax_analysis")
        r = client.delete("/capabilities/agent-a")
        assert r.status_code == 200
        assert r.json()["removed"] == 2
        assert client.get("/capabilities/agent-a").json()["count"] == 0
