"""
tests/unit/test_api_episodic_memory.py
────────────────────────────────────────
Tests for /memory/* REST endpoints.

Run with:  pytest tests/unit/test_api_episodic_memory.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.episodic_memory as mem_module
from pact_ax.primitives.episodic_memory import EpisodicMemory

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def fresh_memory(tmp_path):
    mem_module._memory = EpisodicMemory(tmp_path / "ep.db")
    yield
    mem_module._memory = EpisodicMemory(":memory:")


def record(agent_id, action, **kwargs):
    body = {"action": action, **kwargs}
    return client.post(f"/memory/episodes/{agent_id}", json=body)


# ── POST /memory/episodes/{agent_id} ─────────────────────────────────────────

class TestRecord:

    def test_returns_200(self):
        assert record("agent-a", "contract_review").status_code == 200

    def test_response_fields(self):
        r = record("agent-a", "review",
                   partner_id="agent-b", outcome="positive",
                   importance=0.8, tags=["legal"])
        body = r.json()
        assert body["agent_id"]   == "agent-a"
        assert body["action"]     == "review"
        assert body["partner_id"] == "agent-b"
        assert body["outcome"]    == "positive"
        assert body["importance"] == 0.8
        assert "legal" in body["tags"]

    def test_missing_action_returns_422(self):
        r = client.post("/memory/episodes/agent-a", json={})
        assert r.status_code == 422

    def test_default_outcome_neutral(self):
        body = record("agent-a", "action").json()
        assert body["outcome"] == "neutral"


# ── GET /memory/episodes/{agent_id} ──────────────────────────────────────────

class TestRecall:

    def test_returns_recorded_episodes(self):
        record("agent-a", "review")
        record("agent-a", "handoff")
        r = client.get("/memory/episodes/agent-a")
        assert r.json()["count"] == 2

    def test_filter_by_partner(self):
        record("agent-a", "review", partner_id="agent-b")
        record("agent-a", "review", partner_id="agent-c")
        r = client.get("/memory/episodes/agent-a?partner_id=agent-b")
        assert r.json()["count"] == 1

    def test_filter_by_outcome(self):
        record("agent-a", "r1", outcome="positive")
        record("agent-a", "r2", outcome="negative")
        r = client.get("/memory/episodes/agent-a?outcome=positive")
        assert r.json()["count"] == 1

    def test_filter_by_min_importance(self):
        record("agent-a", "r1", importance=0.9)
        record("agent-a", "r2", importance=0.2)
        r = client.get("/memory/episodes/agent-a?min_importance=0.5")
        body = r.json()
        assert all(e["importance"] >= 0.5 for e in body["episodes"])

    def test_empty_for_unknown_agent(self):
        r = client.get("/memory/episodes/nobody")
        assert r.json()["count"] == 0

    def test_limit_parameter(self):
        for i in range(5):
            record("agent-a", f"action-{i}")
        r = client.get("/memory/episodes/agent-a?limit=2")
        assert len(r.json()["episodes"]) <= 2


# ── GET /memory/episodes/{agent_id}/{partner_id} ──────────────────────────────

class TestRecallPartner:

    def test_returns_partner_episodes(self):
        record("agent-a", "r1", partner_id="agent-b")
        record("agent-a", "r2", partner_id="agent-b")
        record("agent-a", "r3", partner_id="agent-c")
        r = client.get("/memory/episodes/agent-a/agent-b")
        assert r.json()["count"] == 2

    def test_empty_for_unknown_partner(self):
        r = client.get("/memory/episodes/agent-a/nobody")
        assert r.json()["count"] == 0


# ── GET /memory/summary/{agent_id} ───────────────────────────────────────────

class TestSummary:

    def test_total_count(self):
        record("agent-a", "r1")
        record("agent-a", "r2")
        r = client.get("/memory/summary/agent-a")
        assert r.json()["total_episodes"] == 2

    def test_outcome_breakdown(self):
        record("agent-a", "r1", outcome="positive")
        record("agent-a", "r2", outcome="negative")
        record("agent-a", "r3", outcome="positive")
        body = client.get("/memory/summary/agent-a").json()
        assert body["outcome_breakdown"]["positive"] == 2
        assert body["outcome_breakdown"]["negative"] == 1

    def test_top_partners_included(self):
        record("agent-a", "r1", partner_id="agent-b")
        record("agent-a", "r2", partner_id="agent-b")
        body = client.get("/memory/summary/agent-a").json()
        partner_ids = [p["partner_id"] for p in body["top_partners"]]
        assert "agent-b" in partner_ids

    def test_empty_agent(self):
        body = client.get("/memory/summary/nobody").json()
        assert body["total_episodes"] == 0


# ── DELETE /memory/episodes/{agent_id} ────────────────────────────────────────

class TestClear:

    def test_clears_all_episodes(self):
        record("agent-a", "r1")
        record("agent-a", "r2")
        r = client.delete("/memory/episodes/agent-a")
        assert r.json()["cleared"] == 2
        assert client.get("/memory/episodes/agent-a").json()["count"] == 0

    def test_does_not_affect_other_agents(self):
        record("agent-a", "r1")
        record("agent-b", "r2")
        client.delete("/memory/episodes/agent-a")
        assert client.get("/memory/episodes/agent-b").json()["count"] == 1


# ── DELETE /memory/episodes/{agent_id}/{episode_id} ──────────────────────────

class TestDeleteEpisode:

    def test_deletes_specific_episode(self):
        ep_id = record("agent-a", "r1").json()["id"]
        r = client.delete(f"/memory/episodes/agent-a/{ep_id}")
        assert r.status_code == 200
        assert client.get("/memory/episodes/agent-a").json()["count"] == 0

    def test_404_for_unknown(self):
        r = client.delete("/memory/episodes/agent-a/nonexistent-id")
        assert r.status_code == 404
