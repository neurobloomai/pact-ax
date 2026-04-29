"""
tests/unit/test_api_story_keeper.py
─────────────────────────────────────
Tests for /story/* REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.story_keeper as sk_module

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def clear_registry():
    sk_module._keepers.clear()
    yield
    sk_module._keepers.clear()


# ── POST /story/register ──────────────────────────────────────────────────────

class TestRegister:

    def test_returns_200(self):
        r = client.post("/story/register", json={"agent_id": "agent-a"})
        assert r.status_code == 200

    def test_response_body(self):
        r = client.post("/story/register", json={"agent_id": "agent-a"})
        body = r.json()
        assert body["registered"] is True
        assert body["agent_id"] == "agent-a"

    def test_creates_keeper(self):
        client.post("/story/register", json={"agent_id": "agent-x"})
        assert "agent-x" in sk_module._keepers

    def test_with_session_id(self):
        r = client.post("/story/register", json={
            "agent_id": "agent-a",
            "session_id": "session-1",
        })
        assert r.json()["session_id"] == "session-1"
        assert sk_module._keepers["agent-a"].session_id == "session-1"

    def test_register_is_idempotent(self):
        client.post("/story/register", json={"agent_id": "agent-a"})
        client.post("/story/register", json={"agent_id": "agent-a"})
        assert len(sk_module._keepers) == 1

    def test_register_resets_existing_keeper(self):
        client.post("/story/register", json={"agent_id": "agent-a"})
        client.post("/story/agent-a/turn", json={"user_message": "hello"})
        client.post("/story/register", json={"agent_id": "agent-a"})
        assert len(sk_module._keepers["agent-a"].interactions) == 0


# ── POST /story/{agent_id}/turn ───────────────────────────────────────────────

class TestProcessTurn:

    def test_returns_200(self):
        r = client.post("/story/agent-a/turn", json={"user_message": "hello"})
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/story/agent-a/turn", json={"user_message": "hello"})
        body = r.json()
        assert "beat" in body
        assert "current_arc" in body
        assert "interaction_count" in body

    def test_interaction_count_increments(self):
        client.post("/story/agent-a/turn", json={"user_message": "hello"})
        r = client.post("/story/agent-a/turn", json={"user_message": "world"})
        assert r.json()["interaction_count"] == 2

    def test_arc_is_valid_value(self):
        r = client.post("/story/agent-a/turn", json={"user_message": "hello"})
        assert r.json()["current_arc"] in ("exploration", "collaboration", "integration")

    def test_exploration_keywords_detected(self):
        r = client.post("/story/agent-a/turn", json={
            "user_message": "what is PACT and how does it work?"
        })
        assert r.json()["current_arc"] == "exploration"

    def test_collaboration_keywords_detected(self):
        r = client.post("/story/agent-a/turn", json={
            "user_message": "let's build this together and create something new"
        })
        assert r.json()["current_arc"] == "collaboration"

    def test_empty_message_rejected(self):
        r = client.post("/story/agent-a/turn", json={"user_message": ""})
        assert r.status_code == 422


# ── POST /story/{agent_id}/interaction ───────────────────────────────────────

class TestProcessInteraction:

    def test_returns_200(self):
        r = client.post("/story/agent-a/interaction", json={
            "user_input": "Hello",
            "agent_response": "Hi there!",
        })
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/story/agent-a/interaction", json={
            "user_input": "Hello",
            "agent_response": "Hi there!",
        })
        body = r.json()
        assert "timestamp" in body
        assert "user_input" in body
        assert "agent_response" in body
        assert "arc" in body

    def test_user_input_preserved(self):
        r = client.post("/story/agent-a/interaction", json={
            "user_input": "Tell me about PACT",
            "agent_response": "PACT is a protocol...",
        })
        assert r.json()["user_input"] == "Tell me about PACT"

    def test_metadata_preserved(self):
        r = client.post("/story/agent-a/interaction", json={
            "user_input": "Hello",
            "agent_response": "Hi",
            "metadata": {"source": "test"},
        })
        assert r.json()["metadata"]["source"] == "test"

    def test_empty_user_input_rejected(self):
        r = client.post("/story/agent-a/interaction", json={
            "user_input": "",
            "agent_response": "Hi",
        })
        assert r.status_code == 422


# ── GET /story/{agent_id}/state ───────────────────────────────────────────────

class TestGetState:

    def test_returns_200(self):
        r = client.get("/story/agent-a/state")
        assert r.status_code == 200

    def test_initial_state_structure(self):
        r = client.get("/story/agent-a/state")
        body = r.json()
        assert "themes" in body
        assert "arc" in body
        assert "context" in body
        assert "characters" in body
        assert "last_beat" in body

    def test_themes_accumulate(self):
        client.post("/story/agent-a/turn", json={"user_message": "building trust protocol"})
        r = client.get("/story/agent-a/state")
        assert len(r.json()["themes"]) > 0


# ── POST /story/{agent_id}/state ──────────────────────────────────────────────

class TestLoadState:

    def test_load_state_returns_200(self):
        saved = client.get("/story/agent-a/state").json()
        r = client.post("/story/agent-a/state", json={"state": saved})
        assert r.status_code == 200

    def test_load_state_replaces_current(self):
        client.post("/story/agent-a/turn", json={"user_message": "first message"})
        saved = client.get("/story/agent-a/state").json()

        client.post("/story/agent-a/turn", json={"user_message": "second message"})
        client.post("/story/agent-a/state", json={"state": saved})

        restored = client.get("/story/agent-a/state").json()
        assert restored["last_beat"] == saved["last_beat"]


# ── GET /story/{agent_id}/summary ────────────────────────────────────────────

class TestGetSummary:

    def test_returns_200(self):
        r = client.get("/story/agent-a/summary")
        assert r.status_code == 200

    def test_summary_has_required_fields(self):
        r = client.get("/story/agent-a/summary")
        body = r.json()
        assert "agent_id" in body
        assert "current_arc" in body
        assert "total_interactions" in body
        assert "arc_transitions" in body

    def test_interaction_count_reflected(self):
        client.post("/story/agent-a/turn", json={"user_message": "hello"})
        client.post("/story/agent-a/turn", json={"user_message": "world"})
        r = client.get("/story/agent-a/summary")
        assert r.json()["total_interactions"] == 2


# ── POST /story/{agent_id}/reset ─────────────────────────────────────────────

class TestReset:

    def test_reset_returns_200(self):
        r = client.post("/story/agent-a/reset")
        assert r.status_code == 200
        assert r.json()["reset"] is True

    def test_reset_clears_interactions(self):
        client.post("/story/agent-a/turn", json={"user_message": "hello"})
        client.post("/story/agent-a/reset")
        r = client.get("/story/agent-a/summary")
        assert r.json()["total_interactions"] == 0

    def test_reset_clears_themes(self):
        client.post("/story/agent-a/turn", json={"user_message": "building trust protocol"})
        client.post("/story/agent-a/reset")
        r = client.get("/story/agent-a/state")
        assert r.json()["themes"] == []


# ── POST /story/{agent_id}/recall/arc ────────────────────────────────────────

class TestRecallArc:

    def test_valid_arc_returns_200(self):
        r = client.post("/story/agent-a/recall/arc", json={"arc": "exploration"})
        assert r.status_code == 200

    def test_invalid_arc_returns_422(self):
        r = client.post("/story/agent-a/recall/arc", json={"arc": "nostalgia"})
        assert r.status_code == 422

    def test_empty_arc_returns_empty_list(self):
        r = client.post("/story/agent-a/recall/arc", json={"arc": "integration"})
        assert r.json() == []

    def test_recall_returns_matching_interactions(self):
        client.post("/story/agent-a/turn", json={
            "user_message": "what is this and how does it work?"
        })
        r = client.post("/story/agent-a/recall/arc", json={"arc": "exploration", "k": 5})
        assert isinstance(r.json(), list)
        assert len(r.json()) > 0

    def test_recall_result_has_required_fields(self):
        client.post("/story/agent-a/turn", json={"user_message": "what is PACT?"})
        results = client.post("/story/agent-a/recall/arc", json={"arc": "exploration"}).json()
        if results:
            ix = results[0]
            assert "timestamp" in ix
            assert "arc" in ix
            assert "user_input" in ix


# ── POST /story/{agent_id}/recall/context ────────────────────────────────────

class TestRecallContext:

    def test_returns_200(self):
        r = client.post("/story/agent-a/recall/context", json={})
        assert r.status_code == 200

    def test_returns_list(self):
        r = client.post("/story/agent-a/recall/context", json={})
        assert isinstance(r.json(), list)

    def test_k_limits_results(self):
        for i in range(10):
            client.post("/story/agent-a/turn", json={"user_message": f"message {i}"})
        r = client.post("/story/agent-a/recall/context", json={"k": 3})
        assert len(r.json()) <= 3

    def test_prefer_current_arc_false_returns_recent(self):
        for i in range(6):
            client.post("/story/agent-a/turn", json={"user_message": f"message {i}"})
        r = client.post("/story/agent-a/recall/context", json={
            "prefer_current_arc": False, "k": 5
        })
        assert len(r.json()) == 5
