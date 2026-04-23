"""
tests/unit/test_api_policy_align.py
─────────────────────────────────────
Integration tests for /policy/* REST endpoints.

Run with:  pytest tests/unit/test_api_policy_align.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.policy_align as pa_module

client = TestClient(app, raise_server_exceptions=True)


# ── Fixture: reset singletons between tests ──────────────────────────────────

@pytest.fixture(autouse=True)
def reset_manager():
    from pact_ax.coordination.policy_alignment import (
        PolicyAlignmentManager, PolicyLearning,
    )
    pa_module._manager = PolicyAlignmentManager()
    pa_module._learner  = PolicyLearning()
    yield
    pa_module._manager = PolicyAlignmentManager()
    pa_module._learner  = PolicyLearning()


# ── Helpers ───────────────────────────────────────────────────────────────────

DEPLOY = {
    "decision": "deploy",
    "confidence": "CONFIDENT",
    "reasoning": "metrics look good",
    "agent_id": "agent-A",
    "domain": "deployment",
}

HOLD = {
    "decision": "hold",
    "confidence": "MODERATE",
    "reasoning": "need more data",
    "agent_id": "agent-B",
    "domain": "deployment",
}


# ── POST /policy/constraint ───────────────────────────────────────────────────

class TestAddConstraint:

    def test_returns_200(self):
        r = client.post("/policy/constraint", json={
            "name": "min-conf",
            "description": "needs confidence",
            "min_confidence": 0.5,
        })
        assert r.status_code == 200

    def test_added_true(self):
        r = client.post("/policy/constraint", json={
            "name": "c1", "description": "d1",
        })
        assert r.json()["added"] is True

    def test_constraint_stored_in_manager(self):
        client.post("/policy/constraint", json={
            "name": "stored-c", "description": "test",
        })
        assert "stored-c" in pa_module._manager.constraints


# ── POST /policy/evaluate ─────────────────────────────────────────────────────

class TestEvaluate:

    def test_returns_200(self):
        r = client.post("/policy/evaluate", json={"decision": DEPLOY})
        assert r.status_code == 200

    def test_valid_decision_passes(self):
        r = client.post("/policy/evaluate", json={"decision": DEPLOY})
        assert r.json()["valid"] is True
        assert r.json()["issues"] == []

    def test_decision_included_in_response(self):
        r = client.post("/policy/evaluate", json={"decision": DEPLOY})
        assert r.json()["decision"]["decision"] == "deploy"

    def test_unknown_confidence_decision_invalid(self):
        bad = {**DEPLOY, "confidence": "UNKNOWN"}
        r = client.post("/policy/evaluate", json={"decision": bad})
        assert r.json()["valid"] is False

    def test_constraint_violation_reported(self):
        client.post("/policy/constraint", json={
            "name": "high-conf", "description": "test", "min_confidence": 0.9,
        })
        low = {**DEPLOY, "confidence": "LOW"}
        r = client.post("/policy/evaluate", json={"decision": low})
        assert r.json()["valid"] is False
        assert any("high-conf" in i for i in r.json()["issues"])

    def test_missing_decision_field_returns_422(self):
        r = client.post("/policy/evaluate", json={
            "decision": {"confidence": "CONFIDENT"}
        })
        assert r.status_code == 422

    def test_invalid_confidence_level_returns_422(self):
        bad = {**DEPLOY, "confidence": "MEGA_SURE"}
        r = client.post("/policy/evaluate", json={"decision": bad})
        assert r.status_code == 422


# ── POST /policy/resolve ──────────────────────────────────────────────────────

class TestResolve:

    def test_returns_200(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD],
            "strategy": "defer_confident",
        })
        assert r.status_code == 200

    def test_defer_confident_picks_higher(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD],
            "strategy": "defer_confident",
        })
        # CONFIDENT (0.80) > MODERATE (0.60)
        assert r.json()["resolved_decision"]["decision"] == "deploy"

    def test_conservative_picks_lower(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD],
            "strategy": "conservative",
        })
        assert r.json()["resolved_decision"]["decision"] == "hold"

    def test_escalate_returns_escalate_decision(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD],
            "strategy": "escalate_human",
        })
        assert r.json()["resolved_decision"]["decision"] == "ESCALATE_TO_HUMAN"

    def test_consensus_when_agree(self):
        agree1 = {**DEPLOY, "agent_id": "a1"}
        agree2 = {**DEPLOY, "agent_id": "a2"}
        r = client.post("/policy/resolve", json={
            "decisions": [agree1, agree2],
            "strategy": "consensus",
        })
        assert r.json()["resolved_decision"]["decision"] == "deploy"

    def test_no_consensus_when_disagree(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD],
            "strategy": "consensus",
        })
        assert r.json()["resolved_decision"]["decision"] == "NO_CONSENSUS"

    def test_empty_decisions_returns_422(self):
        r = client.post("/policy/resolve", json={
            "decisions": [], "strategy": "defer_confident",
        })
        assert r.status_code == 422

    def test_unknown_strategy_returns_422(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY], "strategy": "magic_wand",
        })
        assert r.status_code == 422

    def test_strategy_used_in_response(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY], "strategy": "defer_confident",
        })
        assert r.json()["strategy_used"] == "defer_confident"

    def test_input_count_in_response(self):
        r = client.post("/policy/resolve", json={
            "decisions": [DEPLOY, HOLD], "strategy": "defer_confident",
        })
        assert r.json()["input_count"] == 2


# ── POST /policy/align ────────────────────────────────────────────────────────

class TestAlign:

    def test_returns_200(self):
        r = client.post("/policy/align", json={"decisions": [DEPLOY]})
        assert r.status_code == 200

    def test_returns_final_decision(self):
        r = client.post("/policy/align", json={"decisions": [DEPLOY]})
        assert "final_decision" in r.json()

    def test_valid_decision_passes_through(self):
        r = client.post("/policy/align", json={"decisions": [DEPLOY]})
        assert r.json()["final_decision"]["decision"] == "deploy"

    def test_invalid_decisions_filtered_out(self):
        client.post("/policy/constraint", json={
            "name": "high-conf", "description": "test", "min_confidence": 0.9,
        })
        low = {**DEPLOY, "confidence": "LOW"}
        r = client.post("/policy/align", json={"decisions": [low]})
        assert r.json()["final_decision"]["decision"] == "NO_VALID_DECISIONS"

    def test_empty_decisions_returns_422(self):
        r = client.post("/policy/align", json={"decisions": []})
        assert r.status_code == 422

    def test_input_count_in_response(self):
        r = client.post("/policy/align", json={"decisions": [DEPLOY, HOLD]})
        assert r.json()["input_count"] == 2


# ── GET /policy/metrics ───────────────────────────────────────────────────────

class TestMetrics:

    def test_returns_200(self):
        r = client.get("/policy/metrics")
        assert r.status_code == 200

    def test_empty_manager_returns_message(self):
        r = client.get("/policy/metrics")
        assert "message" in r.json()

    def test_metrics_populated_after_align(self):
        client.post("/policy/align", json={"decisions": [DEPLOY]})
        r = client.get("/policy/metrics")
        assert "total_decisions" in r.json()
        assert r.json()["total_decisions"] == 1


# ── POST /policy/learn/outcome ────────────────────────────────────────────────

class TestRecordOutcome:

    def test_returns_200(self):
        r = client.post("/policy/learn/outcome", json={
            "decision": DEPLOY,
            "actual_outcome": "succeeded",
            "was_correct": True,
        })
        assert r.status_code == 200

    def test_recorded_flag(self):
        r = client.post("/policy/learn/outcome", json={
            "decision": DEPLOY,
            "actual_outcome": "succeeded",
            "was_correct": True,
        })
        assert r.json()["recorded"] is True

    def test_agent_id_in_response(self):
        r = client.post("/policy/learn/outcome", json={
            "decision": DEPLOY,
            "actual_outcome": "ok",
            "was_correct": True,
        })
        assert r.json()["agent_id"] == "agent-A"


# ── GET /policy/learn/calibration/{agent_id} ─────────────────────────────────

class TestCalibration:

    def test_no_data_returns_message(self):
        r = client.get("/policy/learn/calibration/unknown-agent")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_calibration_after_outcomes(self):
        for correct in [True, True, False]:
            client.post("/policy/learn/outcome", json={
                "decision": DEPLOY,
                "actual_outcome": "ok",
                "was_correct": correct,
            })
        r = client.get("/policy/learn/calibration/agent-A")
        body = r.json()
        assert body["total_decisions"] == 3
        assert "accuracy" in body
        assert "tendency" in body
