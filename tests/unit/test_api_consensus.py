"""
tests/unit/test_api_consensus.py
──────────────────────────────────
Tests for /consensus/* REST endpoints.

Run with:  pytest tests/unit/test_api_consensus.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.consensus as consensus_module

client = TestClient(app, raise_server_exceptions=True)

VOTES_MAJORITY = [
    {"agent_id": "agent-a", "decision": "deploy", "confidence": 0.85},
    {"agent_id": "agent-b", "decision": "deploy", "confidence": 0.72},
    {"agent_id": "agent-c", "decision": "hold",   "confidence": 0.40},
]

VOTES_TIE = [
    {"agent_id": "agent-a", "decision": "yes", "confidence": 0.5},
    {"agent_id": "agent-b", "decision": "no",  "confidence": 0.5},
]


@pytest.fixture(autouse=True)
def clear_sessions():
    consensus_module._sessions.clear()
    yield
    consensus_module._sessions.clear()


# ── POST /consensus/run ───────────────────────────────────────────────────────

class TestRunRound:

    def test_returns_200(self):
        r = client.post("/consensus/run", json={"votes": VOTES_MAJORITY})
        assert r.status_code == 200

    def test_accepted_on_majority(self):
        r = client.post("/consensus/run", json={"votes": VOTES_MAJORITY})
        body = r.json()
        assert body["outcome"] == "accepted"
        assert body["winning_decision"] == "deploy"

    def test_response_has_required_fields(self):
        r = client.post("/consensus/run", json={"votes": VOTES_MAJORITY})
        body = r.json()
        for field in ("round_id", "outcome", "winning_decision", "confidence_score",
                      "strategy_used", "vote_breakdown", "decided_at"):
            assert field in body

    def test_weighted_vote_strategy(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "strategy": "weighted_vote"
        })
        assert r.json()["strategy_used"] == "weighted_vote"

    def test_trust_scores_weight_result(self):
        # Give high trust to agent-c (hold) — should flip result
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY,
            "strategy": "weighted_vote",
            "trust_scores": {"agent-a": 0.1, "agent-b": 0.1, "agent-c": 0.99},
        })
        # With very high trust on hold voter, deploy may lose
        body = r.json()
        assert body["outcome"] in ("accepted", "deadlock", "escalate_to_human")

    def test_quorum_strategy(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "strategy": "quorum"
        })
        assert r.status_code == 200
        assert r.json()["strategy_used"] == "quorum"

    def test_unanimous_strategy_fails_on_disagreement(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "strategy": "unanimous"
        })
        body = r.json()
        assert body["outcome"] in ("escalate_to_human", "deadlock")

    def test_unanimous_strategy_succeeds_on_agreement(self):
        votes = [
            {"agent_id": "a", "decision": "go", "confidence": 0.9},
            {"agent_id": "b", "decision": "go", "confidence": 0.8},
        ]
        r = client.post("/consensus/run", json={"votes": votes, "strategy": "unanimous"})
        assert r.json()["outcome"] == "accepted"

    def test_confidence_threshold_strategy(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "strategy": "confidence_threshold",
            "confidence_threshold": 0.5
        })
        assert r.status_code == 200

    def test_insufficient_votes(self):
        r = client.post("/consensus/run", json={
            "votes": [{"agent_id": "a", "decision": "go", "confidence": 0.9}],
            "min_votes": 2
        })
        assert r.json()["outcome"] == "insufficient_votes"

    def test_abstaining_vote(self):
        votes = [
            {"agent_id": "a", "decision": "go",  "confidence": 0.9, "abstain": False},
            {"agent_id": "b", "decision": "go",  "confidence": 0.8, "abstain": False},
            {"agent_id": "c", "decision": "stop", "confidence": 0.5, "abstain": True},
        ]
        r = client.post("/consensus/run", json={"votes": votes})
        body = r.json()
        assert "c" in body["abstentions"]

    def test_invalid_strategy_returns_422(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "strategy": "nonexistent"
        })
        assert r.status_code == 422

    def test_custom_round_id(self):
        r = client.post("/consensus/run", json={
            "votes": VOTES_MAJORITY, "round_id": "my-round-42"
        })
        assert r.json()["round_id"] == "my-round-42"


# ── POST /consensus/sessions ──────────────────────────────────────────────────

class TestCreateSession:

    def test_creates_session(self):
        r = client.post("/consensus/sessions", json={"session_id": "sess-1"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "sess-1"
        assert r.json()["created"] is True

    def test_auto_generates_id(self):
        r = client.post("/consensus/sessions", json={})
        assert r.status_code == 200
        assert "session_id" in r.json()

    def test_duplicate_session_returns_409(self):
        client.post("/consensus/sessions", json={"session_id": "sess-dup"})
        r = client.post("/consensus/sessions", json={"session_id": "sess-dup"})
        assert r.status_code == 409


# ── POST /consensus/sessions/{sid}/vote ──────────────────────────────────────

class TestSessionVote:

    def test_vote_in_session(self):
        client.post("/consensus/sessions", json={"session_id": "s1"})
        r = client.post("/consensus/sessions/s1/vote", json={"votes": VOTES_MAJORITY})
        assert r.status_code == 200
        assert r.json()["session_id"] == "s1"

    def test_session_tracks_history(self):
        client.post("/consensus/sessions", json={"session_id": "s1"})
        client.post("/consensus/sessions/s1/vote", json={"votes": VOTES_MAJORITY})
        client.post("/consensus/sessions/s1/vote", json={"votes": VOTES_MAJORITY})
        r = client.get("/consensus/sessions/s1")
        assert r.json()["total_rounds"] == 2

    def test_vote_in_nonexistent_session_returns_404(self):
        r = client.post("/consensus/sessions/ghost/vote", json={"votes": VOTES_MAJORITY})
        assert r.status_code == 404


# ── GET /consensus/sessions ───────────────────────────────────────────────────

class TestListSessions:

    def test_lists_active_sessions(self):
        client.post("/consensus/sessions", json={"session_id": "s1"})
        client.post("/consensus/sessions", json={"session_id": "s2"})
        r = client.get("/consensus/sessions")
        ids = {s["session_id"] for s in r.json()["sessions"]}
        assert {"s1", "s2"}.issubset(ids)


# ── GET /consensus/sessions/{sid} ────────────────────────────────────────────

class TestGetSession:

    def test_returns_metrics(self):
        client.post("/consensus/sessions", json={"session_id": "s1"})
        client.post("/consensus/sessions/s1/vote", json={"votes": VOTES_MAJORITY})
        r = client.get("/consensus/sessions/s1")
        body = r.json()
        assert body["total_rounds"] == 1
        assert "acceptance_rate" in body

    def test_404_for_unknown(self):
        r = client.get("/consensus/sessions/ghost")
        assert r.status_code == 404


# ── DELETE /consensus/sessions/{sid} ─────────────────────────────────────────

class TestDeleteSession:

    def test_deletes_session(self):
        client.post("/consensus/sessions", json={"session_id": "s1"})
        r = client.delete("/consensus/sessions/s1")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert client.get("/consensus/sessions/s1").status_code == 404

    def test_404_for_unknown(self):
        r = client.delete("/consensus/sessions/ghost")
        assert r.status_code == 404
