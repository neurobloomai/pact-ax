"""
tests/unit/test_api_trust.py
─────────────────────────────
Tests for /trust/* REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.trust as trust_module

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def clear_registry():
    trust_module._managers.clear()
    yield
    trust_module._managers.clear()


# ── POST /trust/register/{agent_id} ──────────────────────────────────────────

class TestRegister:

    def test_register_returns_200(self):
        r = client.post("/trust/register/agent-a")
        assert r.status_code == 200

    def test_register_response_body(self):
        r = client.post("/trust/register/agent-a")
        body = r.json()
        assert body["registered"] is True
        assert body["agent_id"] == "agent-a"

    def test_register_creates_manager(self):
        client.post("/trust/register/agent-x")
        assert "agent-x" in trust_module._managers

    def test_register_is_idempotent(self):
        client.post("/trust/register/agent-a")
        client.post("/trust/register/agent-a")
        assert len(trust_module._managers) == 1


# ── GET /trust/{agent_id}/{target_id} ────────────────────────────────────────

class TestGetTrust:

    def test_default_trust_is_neutral(self):
        r = client.get("/trust/agent-a/agent-b")
        assert r.status_code == 200
        body = r.json()
        assert body["score"] == 0.5

    def test_response_contains_required_fields(self):
        r = client.get("/trust/agent-a/agent-b")
        body = r.json()
        assert "agent_id" in body
        assert "target_id" in body
        assert "score" in body

    def test_with_valid_context_type(self):
        r = client.get("/trust/agent-a/agent-b?context_type=task_knowledge")
        assert r.status_code == 200

    def test_with_invalid_context_type_returns_422(self):
        r = client.get("/trust/agent-a/agent-b?context_type=nonexistent")
        assert r.status_code == 422


# ── POST /trust/{agent_id}/update ────────────────────────────────────────────

class TestUpdateTrust:

    def test_positive_outcome_raises_score(self):
        r = client.post("/trust/agent-a/update", json={
            "target_id": "agent-b",
            "outcome": "positive",
        })
        assert r.status_code == 200
        assert r.json()["new_score"] > 0.5

    def test_negative_outcome_lowers_score(self):
        r = client.post("/trust/agent-a/update", json={
            "target_id": "agent-b",
            "outcome": "negative",
        })
        assert r.status_code == 200
        assert r.json()["new_score"] < 0.5

    def test_score_never_exceeds_ceiling(self):
        for _ in range(20):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b",
                "outcome": "positive",
                "impact": 1.0,
            })
        r = client.get("/trust/agent-a/agent-b")
        assert r.json()["score"] <= 0.95

    def test_invalid_outcome_returns_422(self):
        r = client.post("/trust/agent-a/update", json={
            "target_id": "agent-b",
            "outcome": "supergood",
        })
        assert r.status_code == 422

    def test_invalid_context_type_returns_422(self):
        r = client.post("/trust/agent-a/update", json={
            "target_id": "agent-b",
            "outcome": "positive",
            "context_type": "bogus_type",
        })
        assert r.status_code == 422

    def test_response_contains_new_score(self):
        r = client.post("/trust/agent-a/update", json={
            "target_id": "agent-b",
            "outcome": "positive",
        })
        body = r.json()
        assert "new_score" in body
        assert "outcome" in body
        assert "target_id" in body


# ── POST /trust/{agent_id}/decay ─────────────────────────────────────────────

class TestDecay:

    def test_decay_drifts_high_trust_toward_neutral(self):
        for _ in range(10):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b", "outcome": "positive",
            })
        before = client.get("/trust/agent-a/agent-b").json()["score"]

        client.post("/trust/agent-a/decay", json={
            "target_id": "agent-b",
            "days_inactive": 30,
        })
        after = client.get("/trust/agent-a/agent-b").json()["score"]
        assert after < before

    def test_decay_drifts_low_trust_toward_neutral(self):
        for _ in range(10):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b", "outcome": "negative",
            })
        before = client.get("/trust/agent-a/agent-b").json()["score"]

        client.post("/trust/agent-a/decay", json={
            "target_id": "agent-b",
            "days_inactive": 30,
        })
        after = client.get("/trust/agent-a/agent-b").json()["score"]
        assert after > before

    def test_decay_returns_200(self):
        r = client.post("/trust/agent-a/decay", json={"days_inactive": 5})
        assert r.status_code == 200


# ── GET /trust/{agent_id}/network/{target_id} ────────────────────────────────

class TestNetworkTrust:

    def test_unknown_target_returns_default(self):
        r = client.get("/trust/agent-a/network/unknown-agent")
        assert r.status_code == 200
        assert r.json()["network_trust"] == 0.5

    def test_source_is_direct_for_known_agent(self):
        client.post("/trust/agent-a/update", json={
            "target_id": "agent-b", "outcome": "positive",
        })
        r = client.get("/trust/agent-a/network/agent-b")
        assert r.json()["source"] == "direct"

    def test_source_is_transitive_for_unknown_agent(self):
        r = client.get("/trust/agent-a/network/never-seen")
        assert r.json()["source"] == "transitive"


# ── POST /trust/{agent_id}/external ─────────────────────────────────────────

class TestExternalTrust:
    """Cross-system trust sharing."""

    def test_external_trust_returns_200(self):
        r = client.post("/trust/agent-a/external", json={
            "source_agent": "system-b",
            "target_agent": "agent-c",
            "score": 0.8,
        })
        assert r.status_code == 200

    def test_external_trust_response_body(self):
        r = client.post("/trust/agent-a/external", json={
            "source_agent": "system-b",
            "target_agent": "agent-c",
            "score": 0.8,
        })
        body = r.json()
        assert body["recorded"] is True
        assert body["source_agent"] == "system-b"
        assert body["score"] == 0.8

    def test_external_trust_score_above_range_rejected(self):
        r = client.post("/trust/agent-a/external", json={
            "source_agent": "system-b",
            "target_agent": "agent-c",
            "score": 1.5,
        })
        assert r.status_code == 422

    def test_external_trust_score_below_range_rejected(self):
        r = client.post("/trust/agent-a/external", json={
            "source_agent": "system-b",
            "target_agent": "agent-c",
            "score": -0.1,
        })
        assert r.status_code == 422

    def test_multiple_external_signals_accumulate(self):
        for score in [0.8, 0.9, 0.7]:
            client.post("/trust/agent-a/external", json={
                "source_agent": "system-b",
                "target_agent": "agent-c",
                "score": score,
            })
        r = client.get("/trust/agent-a/insights")
        assert "system-b" in r.json()["relationships"]


# ── GET /trust/{agent_id}/insights ───────────────────────────────────────────

class TestInsights:

    def test_insights_returns_required_keys(self):
        r = client.get("/trust/agent-a/insights")
        assert r.status_code == 200
        body = r.json()
        assert "agent_id" in body
        assert "tracked_agents" in body
        assert "relationships" in body

    def test_insights_empty_when_no_interactions(self):
        r = client.get("/trust/agent-a/insights")
        assert r.json()["tracked_agents"] == 0

    def test_insights_populated_after_update(self):
        client.post("/trust/agent-a/update", json={
            "target_id": "agent-b", "outcome": "positive",
        })
        r = client.get("/trust/agent-a/insights")
        body = r.json()
        assert body["tracked_agents"] == 1
        assert "agent-b" in body["relationships"]


# ── POST /trust/{agent_id}/agents ────────────────────────────────────────────

class TestTrustedAgents:

    def test_returns_empty_list_with_no_history(self):
        r = client.post("/trust/agent-a/agents", json={"min_trust": 0.6})
        assert r.status_code == 200
        assert r.json()["trusted_agents"] == []

    def test_returns_agent_above_threshold(self):
        for _ in range(10):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b", "outcome": "positive",
            })
        r = client.post("/trust/agent-a/agents", json={"min_trust": 0.6})
        assert "agent-b" in r.json()["trusted_agents"]

    def test_excludes_low_trust_agent(self):
        for _ in range(5):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b", "outcome": "negative",
            })
        r = client.post("/trust/agent-a/agents", json={"min_trust": 0.6})
        assert "agent-b" not in r.json()["trusted_agents"]


# ── DELETE /trust/{agent_id}/{target_id} ─────────────────────────────────────

class TestResetTrust:

    def test_reset_returns_200(self):
        r = client.delete("/trust/agent-a/agent-b")
        assert r.status_code == 200
        assert r.json()["reset"] is True

    def test_reset_clears_history(self):
        for _ in range(5):
            client.post("/trust/agent-a/update", json={
                "target_id": "agent-b", "outcome": "positive",
            })
        client.delete("/trust/agent-a/agent-b")
        r = client.get("/trust/agent-a/agent-b")
        assert r.json()["score"] == 0.5
