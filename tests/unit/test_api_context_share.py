"""
tests/unit/test_api_context_share.py
──────────────────────────────────────
Integration tests for /context/* REST endpoints.

Run with:  pytest tests/unit/test_api_context_share.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.context_share as cs_module

client = TestClient(app, raise_server_exceptions=True)


# ── Fixture: clear registry between tests ────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_registry():
    cs_module._managers.clear()
    yield
    cs_module._managers.clear()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── POST /context/register ────────────────────────────────────────────────────

class TestRegister:

    def test_register_returns_200(self):
        r = client.post("/context/register", json={
            "agent_id": "agent-001",
            "agent_type": "analyst",
            "capabilities": ["nlp"],
        })
        assert r.status_code == 200

    def test_register_response_body(self):
        r = client.post("/context/register", json={"agent_id": "agent-001"})
        body = r.json()
        assert body["registered"] is True
        assert body["agent_id"] == "agent-001"

    def test_register_creates_manager(self):
        client.post("/context/register", json={
            "agent_id": "agent-reg",
            "agent_type": "support",
            "capabilities": ["billing"],
        })
        assert "agent-reg" in cs_module._managers
        assert cs_module._managers["agent-reg"].identity.agent_type == "support"

    def test_register_is_idempotent(self):
        for _ in range(2):
            r = client.post("/context/register", json={"agent_id": "agent-x"})
            assert r.status_code == 200


# ── POST /context/packet ──────────────────────────────────────────────────────

class TestCreatePacket:

    def test_create_packet_returns_200(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "payload": {"task": "billing_support"},
        })
        assert r.status_code == 200

    def test_response_has_expected_keys(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "payload": {"task": "billing_support"},
        })
        body = r.json()
        for key in ("packet_id", "from_agent", "to_agent",
                    "context_type", "priority", "trust_required"):
            assert key in body

    def test_correct_agents_in_response(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-sender",
            "target_agent": "agent-receiver",
            "context_type": "task_knowledge",
            "payload": {"x": 1},
        })
        body = r.json()
        assert body["from_agent"] == "agent-sender"
        assert body["to_agent"]   == "agent-receiver"

    def test_ttl_sets_expires_at(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "payload": {"x": 1},
            "ttl_seconds": 300,
        })
        assert r.json()["expires_at"] is not None

    def test_invalid_context_type_returns_422(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "invalid_type",
            "payload": {"x": 1},
        })
        assert r.status_code == 422

    def test_invalid_priority_returns_422(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "payload": {"x": 1},
            "priority": "ludicrous_speed",
        })
        assert r.status_code == 422

    def test_emotional_state_has_strong_trust_required(self):
        r = client.post("/context/packet", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "emotional_state",
            "payload": {"mood": "calm"},
        })
        assert r.json()["trust_required"] == "strong"


# ── POST /context/trust ───────────────────────────────────────────────────────

class TestAssessTrust:

    def test_returns_200(self):
        r = client.post("/context/trust", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
        })
        assert r.status_code == 200

    def test_response_keys(self):
        r = client.post("/context/trust", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
        })
        body = r.json()
        for key in ("agent_id", "context_type", "base_trust",
                    "final_trust", "recommendation"):
            assert key in body

    def test_trust_in_valid_range(self):
        r = client.post("/context/trust", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
        })
        assert 0.0 <= r.json()["final_trust"] <= 1.0

    def test_high_stakes_lowers_trust(self):
        base = client.post("/context/trust", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "task_knowledge",
        }).json()["final_trust"]

        high = client.post("/context/trust", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "task_knowledge",
            "current_situation": {"stakes": "high"},
        }).json()["final_trust"]

        assert high < base

    def test_invalid_context_type_returns_422(self):
        r = client.post("/context/trust", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "bogus",
        })
        assert r.status_code == 422


# ── POST /context/capability ──────────────────────────────────────────────────

class TestSenseCapability:

    def test_returns_200(self):
        r = client.post("/context/capability", json={
            "agent_id": "agent-001",
            "current_task": "billing_resolution",
        })
        assert r.status_code == 200

    def test_response_keys(self):
        r = client.post("/context/capability", json={
            "agent_id": "agent-001",
            "current_task": "billing_resolution",
        })
        for key in ("task", "current_confidence", "threshold",
                    "approaching_limit", "recommendation"):
            assert key in r.json()

    def test_unknown_task_full_confidence(self):
        r = client.post("/context/capability", json={
            "agent_id": "agent-001",
            "current_task": "never_seen_before",
        })
        assert r.json()["current_confidence"] == 1.0


# ── POST /context/capability/update ──────────────────────────────────────────

class TestUpdateConfidence:

    def test_returns_200(self):
        r = client.post("/context/capability/update", json={
            "agent_id": "agent-001",
            "task": "billing",
            "confidence": 0.6,
        })
        assert r.status_code == 200

    def test_updated_flag_true(self):
        r = client.post("/context/capability/update", json={
            "agent_id": "agent-001",
            "task": "billing",
            "confidence": 0.6,
        })
        assert r.json()["updated"] is True

    def test_confidence_reflects_in_sense(self):
        client.post("/context/capability/update", json={
            "agent_id": "agent-001", "task": "hard_task", "confidence": 0.2,
        })
        r = client.post("/context/capability", json={
            "agent_id": "agent-001",
            "current_task": "hard_task",
            "confidence_threshold": 0.7,
        })
        assert r.json()["approaching_limit"] is True

    def test_out_of_range_confidence_returns_422(self):
        r = client.post("/context/capability/update", json={
            "agent_id": "agent-001", "task": "x", "confidence": 1.5,
        })
        assert r.status_code == 422


# ── POST /context/outcome ─────────────────────────────────────────────────────

class TestRecordOutcome:

    def test_returns_200(self):
        r = client.post("/context/outcome", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "outcome": "positive",
        })
        assert r.status_code == 200

    def test_recorded_flag(self):
        r = client.post("/context/outcome", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "outcome": "positive",
        })
        assert r.json()["recorded"] is True

    def test_outcome_affects_trust(self):
        base = client.post("/context/trust", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "task_knowledge",
        }).json()["base_trust"]

        for _ in range(8):
            client.post("/context/outcome", json={
                "agent_id": "a1", "target_agent": "a2",
                "context_type": "task_knowledge", "outcome": "positive",
            })

        after = client.post("/context/trust", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "task_knowledge",
        }).json()["base_trust"]

        assert after > base

    def test_invalid_context_type_returns_422(self):
        r = client.post("/context/outcome", json={
            "agent_id": "a1", "target_agent": "a2",
            "context_type": "bad_type", "outcome": "positive",
        })
        assert r.status_code == 422


# ── POST /context/handoff ─────────────────────────────────────────────────────

class TestPrepareHandoff:

    def test_returns_200(self):
        r = client.post("/context/handoff", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "current_task": "billing_resolution",
        })
        assert r.status_code == 200

    def test_response_keys(self):
        r = client.post("/context/handoff", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "current_task": "billing_resolution",
        })
        for key in ("packet_id", "from_agent", "to_agent",
                    "context_type", "payload_keys"):
            assert key in r.json()

    def test_context_type_is_handoff_request(self):
        r = client.post("/context/handoff", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "current_task": "billing_resolution",
        })
        assert r.json()["context_type"] == "handoff_request"


# ── GET /context/insights/{agent_id} ─────────────────────────────────────────

class TestGetInsights:

    def test_returns_200(self):
        r = client.get("/context/insights/agent-001")
        assert r.status_code == 200

    def test_response_keys(self):
        r = client.get("/context/insights/agent-001")
        for key in ("trust_summary", "capability_status", "collaboration_patterns"):
            assert key in r.json()

    def test_insights_reflect_recorded_outcomes(self):
        client.post("/context/outcome", json={
            "agent_id": "agent-001",
            "target_agent": "agent-002",
            "context_type": "task_knowledge",
            "outcome": "positive",
        })
        r = client.get("/context/insights/agent-001")
        assert "agent-002" in r.json()["trust_summary"]
