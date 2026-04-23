"""
tests/integration/test_e2e_agent_handoff.py
─────────────────────────────────────────────
End-to-end scenario: a billing-escalation handoff chain between two agents.

Scenario
────────
1. Agent-A (tier-1 support) registers and processes a billing dispute
2. A evaluates a "resolve" decision, aligns it, records it for learning
3. A's confidence drops — it detects it's approaching its capability limit
4. A prepares a state handoff packet to Agent-B (billing specialist)
5. B receives the packet and integrates it
6. B records a positive collaboration outcome → A's trust score rises
7. A reads its updated trust metrics
8. Policy layer confirms the final decision is well-aligned

Run with:  pytest tests/integration/test_e2e_agent_handoff.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.context_share as cs_module
import pact_ax.api.routes.state_transfer as st_module
import pact_ax.api.routes.policy_align as pa_module
from pact_ax.coordination.policy_alignment import PolicyAlignmentManager, PolicyLearning

client = TestClient(app, raise_server_exceptions=True)


# ── Fixture: clean state for every test ──────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    cs_module._managers.clear()
    st_module._managers.clear()
    pa_module._manager = PolicyAlignmentManager()
    pa_module._learner = PolicyLearning()
    yield
    cs_module._managers.clear()
    st_module._managers.clear()
    pa_module._manager = PolicyAlignmentManager()
    pa_module._learner = PolicyLearning()


# ── Full handoff scenario ─────────────────────────────────────────────────────

class TestBillingEscalationHandoff:
    """
    Two-agent billing escalation: tier-1 (agent-A) hands off to specialist (agent-B).
    Tests every layer — context, policy, state transfer — in one coherent flow.
    """

    # ── Step 1: both agents register ─────────────────────────────────────────

    def test_01_agent_a_registers(self):
        r = client.post("/context/register", json={
            "agent_id": "agent-A",
            "agent_type": "tier1-support",
            "capabilities": ["billing_basic", "account_lookup"],
        })
        assert r.status_code == 200
        assert r.json()["registered"] is True

    def test_02_agent_b_registers(self):
        r = client.post("/context/register", json={
            "agent_id": "agent-B",
            "agent_type": "billing-specialist",
            "capabilities": ["billing_advanced", "refund_authority", "dispute_resolution"],
        })
        assert r.status_code == 200
        assert r.json()["registered"] is True

    # ── Step 2: A evaluates and aligns a decision ─────────────────────────────

    def test_03_policy_evaluate_decision(self):
        r = client.post("/policy/evaluate", json={"decision": {
            "decision": "escalate",
            "confidence": "MODERATE",
            "reasoning": "dispute exceeds tier-1 authority",
            "agent_id": "agent-A",
            "domain": "billing",
        }})
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_04_policy_align_decision(self):
        r = client.post("/policy/align", json={"decisions": [{
            "decision": "escalate",
            "confidence": "CONFIDENT",
            "reasoning": "dispute exceeds tier-1 authority",
            "agent_id": "agent-A",
            "domain": "billing",
        }]})
        assert r.status_code == 200
        body = r.json()
        assert body["final_decision"]["decision"] == "escalate"
        assert body["input_count"] == 1

    def test_05_record_outcome_for_learning(self):
        r = client.post("/policy/learn/outcome", json={
            "decision": {
                "decision": "escalate",
                "confidence": "CONFIDENT",
                "reasoning": "dispute exceeds authority",
                "agent_id": "agent-A",
                "domain": "billing",
            },
            "actual_outcome": "resolved_by_specialist",
            "was_correct": True,
        })
        assert r.status_code == 200
        assert r.json()["recorded"] is True

    # ── Step 3: A detects capability limit ────────────────────────────────────

    def test_06_update_capability_confidence(self):
        r = client.post("/context/capability/update", json={
            "agent_id": "agent-A",
            "task": "billing_dispute",
            "confidence": 0.35,
        })
        assert r.status_code == 200
        assert r.json()["updated"] is True

    def test_07_sense_capability_limit(self):
        client.post("/context/capability/update", json={
            "agent_id": "agent-A",
            "task": "billing_dispute",
            "confidence": 0.35,
        })
        r = client.post("/context/capability", json={
            "agent_id": "agent-A",
            "current_task": "billing_dispute",
            "confidence_threshold": 0.7,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["approaching_limit"] is True
        assert "handoff" in body["recommendation"].lower()

    # ── Step 4: A creates a context handoff packet ────────────────────────────

    def test_08_prepare_context_handoff(self):
        r = client.post("/context/handoff", json={
            "agent_id": "agent-A",
            "target_agent": "agent-B",
            "current_task": "billing_dispute",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["from_agent"] == "agent-A"
        assert body["to_agent"] == "agent-B"
        assert body["context_type"] == "handoff_request"

    # ── Step 5: state transfer — prepare → send → receive ────────────────────

    def test_09_prepare_state_transfer(self):
        r = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": {
                "task": "billing_dispute",
                "customer_id": "cust-9912",
                "dispute_amount": 349.99,
                "progress": 0.4,
                "notes": "customer insists charge is duplicated",
            },
            "reason": "escalation",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "preparing"
        assert body["packet_id"].startswith("pkt-")

    def test_10_send_state_packet(self):
        r_prep = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": {"task": "billing_dispute", "progress": 0.4},
            "reason": "escalation",
        })
        pid = r_prep.json()["packet_id"]
        r_send = client.post("/transfer/send", json={
            "agent_id": "agent-A",
            "packet_id": pid,
        })
        assert r_send.status_code == 200
        assert r_send.json()["status"] == "in_flight"

    def test_11_receive_state_packet(self):
        # Full prepare → send → receive roundtrip
        state = {"task": "billing_dispute", "progress": 0.4, "customer_id": "cust-9912"}
        r_prep = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": state,
            "reason": "escalation",
        })
        pid = r_prep.json()["packet_id"]
        r_send = client.post("/transfer/send", json={
            "agent_id": "agent-A", "packet_id": pid,
        })
        packet = r_send.json()

        r_recv = client.post("/transfer/receive", json={
            "agent_id": "agent-B", "packet": packet,
        })
        assert r_recv.status_code == 200
        body = r_recv.json()
        assert body["success"] is True
        assert body["integrated_state"]["state_data"] == state

    # ── Step 6: B records positive outcome → A's trust rises ─────────────────

    def test_12_record_positive_collaboration_outcome(self):
        r = client.post("/context/outcome", json={
            "agent_id": "agent-A",
            "target_agent": "agent-B",
            "context_type": "task_knowledge",
            "outcome": "positive",
        })
        assert r.status_code == 200
        assert r.json()["recorded"] is True

    def test_13_repeated_positive_outcomes_raise_trust(self):
        base_trust = client.post("/context/trust", json={
            "agent_id": "agent-A",
            "target_agent": "agent-B",
            "context_type": "task_knowledge",
        }).json()["base_trust"]

        for _ in range(8):
            client.post("/context/outcome", json={
                "agent_id": "agent-A",
                "target_agent": "agent-B",
                "context_type": "task_knowledge",
                "outcome": "positive",
            })

        after_trust = client.post("/context/trust", json={
            "agent_id": "agent-A",
            "target_agent": "agent-B",
            "context_type": "task_knowledge",
        }).json()["base_trust"]

        assert after_trust > base_trust

    # ── Step 7: A checks its collaboration insights ───────────────────────────

    def test_14_insights_reflect_collaboration(self):
        client.post("/context/outcome", json={
            "agent_id": "agent-A",
            "target_agent": "agent-B",
            "context_type": "task_knowledge",
            "outcome": "positive",
        })
        r = client.get("/context/insights/agent-A")
        assert r.status_code == 200
        body = r.json()
        assert "agent-B" in body["trust_summary"]

    # ── Step 8: policy metrics reflect the full run ───────────────────────────

    def test_15_policy_metrics_after_full_run(self):
        # Align a decision so metrics are non-empty
        client.post("/policy/align", json={"decisions": [{
            "decision": "escalate",
            "confidence": "CONFIDENT",
            "reasoning": "exceeds authority",
            "agent_id": "agent-A",
            "domain": "billing",
        }]})
        r = client.get("/policy/metrics")
        assert r.status_code == 200
        assert r.json()["total_decisions"] >= 1

    def test_16_calibration_reflects_recorded_outcomes(self):
        # Record two correct outcomes so calibration has data
        for correct in [True, True]:
            client.post("/policy/learn/outcome", json={
                "decision": {
                    "decision": "escalate",
                    "confidence": "CONFIDENT",
                    "reasoning": "test",
                    "agent_id": "agent-A",
                    "domain": "billing",
                },
                "actual_outcome": "resolved",
                "was_correct": correct,
            })
        r = client.get("/policy/learn/calibration/agent-A")
        assert r.status_code == 200
        body = r.json()
        assert "accuracy" in body
        assert body["total_decisions"] == 2


# ── Negative path: cross-agent integrity ─────────────────────────────────────

class TestHandoffIntegrity:
    """Verify the handoff protocol rejects misuse."""

    def test_wrong_recipient_cannot_receive(self):
        r_prep = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": {"task": "billing"},
        })
        pid = r_prep.json()["packet_id"]
        r_send = client.post("/transfer/send", json={
            "agent_id": "agent-A", "packet_id": pid,
        })
        packet = r_send.json()

        r_recv = client.post("/transfer/receive", json={
            "agent_id": "agent-C",   # wrong recipient
            "packet": packet,
        })
        assert r_recv.json()["success"] is False

    def test_double_send_rejected(self):
        r_prep = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": {"task": "billing"},
        })
        pid = r_prep.json()["packet_id"]
        client.post("/transfer/send", json={"agent_id": "agent-A", "packet_id": pid})
        r2 = client.post("/transfer/send", json={"agent_id": "agent-A", "packet_id": pid})
        assert r2.status_code == 409

    def test_invalid_policy_decisions_blocked(self):
        client.post("/policy/constraint", json={
            "name": "high-confidence-required",
            "description": "Only confident decisions pass",
            "min_confidence": 0.9,
        })
        r = client.post("/policy/align", json={"decisions": [{
            "decision": "escalate",
            "confidence": "LOW",
            "reasoning": "unsure",
            "agent_id": "agent-A",
            "domain": "billing",
        }]})
        assert r.status_code == 200
        assert r.json()["final_decision"]["decision"] == "NO_VALID_DECISIONS"

    def test_rollback_after_receive(self):
        r_prep = client.post("/transfer/prepare", json={
            "from_agent_id": "agent-A",
            "to_agent_id": "agent-B",
            "state_data": {"task": "billing"},
        })
        pid = r_prep.json()["packet_id"]
        r_send = client.post("/transfer/send", json={"agent_id": "agent-A", "packet_id": pid})
        packet = r_send.json()
        client.post("/transfer/receive", json={"agent_id": "agent-B", "packet": packet})

        r_rollback = client.post("/transfer/rollback", json={
            "agent_id": "agent-B", "packet_id": pid,
        })
        assert r_rollback.status_code == 200
        assert r_rollback.json()["rolled_back"] is True
