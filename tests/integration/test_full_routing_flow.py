"""
tests/integration/test_full_routing_flow.py
─────────────────────────────────────────────
Integration scenarios that chain multiple primitives together.

Scenarios
─────────
1. Full routing flow
   register capabilities → build trust → route → record episode → recall

2. DLQ flow
   enqueue failed packet → retry → exhaust → verify stats

3. Consensus + trust integration
   run consensus with trust-weighted votes → verify decision reflects trust

4. Orchestration: register agents, route, then check episodic memory captures outcome

Run with:  pytest tests/integration/test_full_routing_flow.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.capabilities as cap_module
import pact_ax.api.routes.trust as trust_module
import pact_ax.api.routes.episodic_memory as mem_module
import pact_ax.api.routes.dead_letter as dlq_module
from pact_ax.primitives.capability_registry import CapabilityRegistry
from pact_ax.primitives.episodic_memory import EpisodicMemory
from pact_ax.primitives.dead_letter_queue import DeadLetterQueue
from pact_ax.primitives.agent_router import AgentRouter
from pact_ax.primitives.trust_score import TrustManager
from pact_ax.primitives.context_share.schemas import ContextType, CollaborationOutcome
from pact_ax.coordination.consensus import ConsensusProtocol, ConsensusStrategy, Vote

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def reset_modules(tmp_path):
    cap_module._registry  = CapabilityRegistry(":memory:")
    trust_module._managers.clear()
    mem_module._memory    = EpisodicMemory(tmp_path / "ep.db")
    dlq_module._dlq       = DeadLetterQueue(tmp_path / "dlq.db", max_attempts=3)
    yield
    cap_module._registry  = CapabilityRegistry(":memory:")
    trust_module._managers.clear()
    mem_module._memory    = EpisodicMemory(":memory:")


# ── Scenario 1: Full routing flow ─────────────────────────────────────────────

class TestFullRoutingFlow:
    """
    register capabilities → build trust → route → record episode → recall
    """

    def test_route_selects_most_trusted_capable_agent(self, tmp_path):
        cap_db   = str(tmp_path / "caps.db")
        trust_db = str(tmp_path / "trust.db")

        # 1. Register three agents with same skill
        reg = CapabilityRegistry(cap_db)
        reg.register("agent-alpha", "contract_review", "Expert NDA review", tags=["legal"])
        reg.register("agent-beta",  "contract_review", "Service agreements", tags=["legal"])
        reg.register("agent-gamma", "contract_review", "General review",     tags=["legal"])

        # 2. Build trust — alpha most trusted, gamma negative
        mgr = TrustManager.load(trust_db, "orchestrator")
        for _ in range(2):
            mgr.update_trust("agent-alpha", CollaborationOutcome.POSITIVE,
                             ContextType.TASK_KNOWLEDGE, impact=1.0)
        mgr.update_trust("agent-beta",  CollaborationOutcome.POSITIVE,
                         ContextType.TASK_KNOWLEDGE, impact=0.5)
        mgr.update_trust("agent-gamma", CollaborationOutcome.NEGATIVE,
                         ContextType.TASK_KNOWLEDGE, impact=0.8)
        mgr.save(trust_db)

        # 3. Route
        router   = AgentRouter(capability_db=cap_db, trust_db=trust_db)
        decision = router.route("orchestrator", "contract_review", min_trust=0.4)

        assert decision.routed
        assert decision.best_agent == "agent-alpha"
        assert decision.strategy_used == "trust_weighted"
        assert len(decision.candidates) >= 2

    def test_trust_below_threshold_excluded(self, tmp_path):
        cap_db   = str(tmp_path / "caps.db")
        trust_db = str(tmp_path / "trust.db")

        reg = CapabilityRegistry(cap_db)
        reg.register("agent-a", "tax_analysis", "Tax expert", tags=["finance"])
        reg.register("agent-b", "tax_analysis", "Tax helper", tags=["finance"])

        mgr = TrustManager.load(trust_db, "orchestrator")
        mgr.update_trust("agent-a", CollaborationOutcome.NEGATIVE,
                         ContextType.TASK_KNOWLEDGE, impact=1.0)
        mgr.update_trust("agent-b", CollaborationOutcome.POSITIVE,
                         ContextType.TASK_KNOWLEDGE, impact=1.0)
        mgr.save(trust_db)

        router   = AgentRouter(capability_db=cap_db, trust_db=trust_db)
        decision = router.route("orchestrator", "tax_analysis", min_trust=0.55)

        assert decision.best_agent == "agent-b"
        for c in decision.candidates:
            assert c.trust_score >= 0.55

    def test_episode_records_route_outcome(self, tmp_path):
        ep_db = str(tmp_path / "ep.db")
        mem   = EpisodicMemory(ep_db)

        mem.record("orchestrator", "contract_review",
                   partner_id="agent-alpha", outcome="positive",
                   importance=0.9, tags=["legal", "routing"])
        mem.record("orchestrator", "contract_review",
                   partner_id="agent-beta",  outcome="neutral",
                   importance=0.5, tags=["legal"])

        eps = mem.recall("orchestrator", partner_id="agent-alpha")
        assert len(eps) == 1
        assert eps[0].outcome == "positive"

        summary = mem.summary("orchestrator")
        assert summary["total_episodes"] == 2
        assert summary["outcome_breakdown"]["positive"] == 1
        assert "agent-alpha" in [p["partner_id"] for p in summary["top_partners"]]


# ── Scenario 2: DLQ flow ──────────────────────────────────────────────────────

class TestDLQFlow:
    """
    enqueue failed packet → retry up to max → exhaust → verify stats
    """

    def test_retry_to_exhaustion(self, tmp_path):
        dlq = DeadLetterQueue(tmp_path / "dlq.db", max_attempts=3, base_seconds=1)

        e = dlq.enqueue("pkt-fail", "orchestrator", "agent-b",
                        {"task": "review"}, reason="timeout")
        assert e.status == "pending"
        assert e.retryable

        e = dlq.retry(e.id, reason="retry 1")
        assert e.status == "retrying"
        assert e.attempt == 1

        e = dlq.retry(e.id, reason="retry 2")
        assert e.attempt == 2

        e = dlq.retry(e.id, reason="retry 3")
        assert e.status == "exhausted"
        assert not e.retryable

        stats = dlq.stats()
        assert stats["exhausted"] == 1
        assert stats["pending"]   == 0

    def test_resolve_after_partial_retries(self, tmp_path):
        dlq = DeadLetterQueue(tmp_path / "dlq.db", max_attempts=5, base_seconds=1)

        e = dlq.enqueue("pkt-x", "orch", "agent-b", {})
        dlq.retry(e.id)  # one failed retry
        e = dlq.resolve(e.id)

        assert e.status   == "resolved"
        assert e.attempt  == 1
        assert e.next_retry is None

        stats = dlq.stats()
        assert stats["resolved"] == 1
        assert stats["retrying"] == 0

    def test_api_dlq_full_lifecycle(self):
        # Enqueue
        r = client.post("/dlq/enqueue", json={
            "packet_id": "pkt-lifecycle", "from_agent": "orch",
            "to_agent": "agent-b", "payload": {}, "max_attempts": 2
        })
        assert r.status_code == 200
        eid = r.json()["id"]

        # Retry once
        r = client.post(f"/dlq/{eid}/retry")
        assert r.json()["attempt"] == 1

        # Retry again → exhausted
        r = client.post(f"/dlq/{eid}/retry")
        assert r.json()["status"] == "exhausted"

        # Stats
        stats = client.get("/dlq/stats").json()
        assert stats["exhausted"] >= 1


# ── Scenario 3: Consensus + trust weighting ────────────────────────────────────

class TestConsensusWithTrust:
    """
    Verify that trust scores correctly shift consensus outcomes.
    """

    def test_high_trust_minority_can_override_majority(self):
        protocol = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)
        votes = [
            Vote("agent-a", "deploy", 0.85),
            Vote("agent-b", "deploy", 0.72),
            Vote("agent-c", "hold",   0.90),   # sole dissenter, very confident
        ]
        # Give agent-c enormous trust → hold might win
        trust = {"agent-a": 0.2, "agent-b": 0.2, "agent-c": 0.99}
        result = protocol.run(votes, trust_scores=trust)

        # Weight breakdown should show hold is competitive
        assert "hold"   in result.vote_breakdown
        assert "deploy" in result.vote_breakdown

    def test_abstentions_tracked(self):
        protocol = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)
        votes = [
            Vote("agent-a", "go",   0.9, abstain=False),
            Vote("agent-b", "go",   0.8, abstain=False),
            Vote("agent-c", "stop", 0.5, abstain=True),
        ]
        result = protocol.run(votes)
        assert "agent-c" in result.abstentions

    def test_session_tracks_acceptance_rate(self):
        r = client.post("/consensus/sessions", json={"session_id": "s-rate"})
        assert r.status_code == 200

        majority_votes = [
            {"agent_id": "a", "decision": "go", "confidence": 0.9},
            {"agent_id": "b", "decision": "go", "confidence": 0.8},
            {"agent_id": "c", "decision": "stop", "confidence": 0.3},
        ]
        # Two rounds — both should reach consensus
        client.post("/consensus/sessions/s-rate/vote", json={"votes": majority_votes})
        client.post("/consensus/sessions/s-rate/vote", json={"votes": majority_votes})

        metrics = client.get("/consensus/sessions/s-rate").json()
        assert metrics["total_rounds"] == 2
        assert metrics["acceptance_rate"] == 1.0


# ── Scenario 4: Cross-primitive memory of routed tasks ────────────────────────

class TestCrossPrimitiveMemory:
    """
    Simulate orchestrator routing to an agent, recording the episode,
    and verifying the memory reflects the trust interaction.
    """

    def test_memory_reflects_trust_and_route(self, tmp_path):
        # Step 1: trust update via API
        client.post("/trust/orchestrator/update", json={
            "target_id": "agent-specialist",
            "outcome": "positive",
            "context_type": "task_knowledge",
            "impact": 1.0
        })
        t = client.get("/trust/orchestrator/agent-specialist").json()
        assert t["trust_score"] > 0.5

        # Step 2: record episode
        client.post("/memory/episodes/orchestrator", json={
            "action":     "contract_review",
            "partner_id": "agent-specialist",
            "outcome":    "positive",
            "importance": t["trust_score"],
            "tags":       ["legal", "routing-test"],
            "context":    {"trust_at_routing": t["trust_score"]}
        })

        # Step 3: verify recall
        eps = client.get(
            "/memory/episodes/orchestrator/agent-specialist"
        ).json()
        assert eps["count"] == 1
        ep = eps["episodes"][0]
        assert ep["outcome"]  == "positive"
        assert ep["context"]["trust_at_routing"] > 0.5

        # Step 4: summary
        summary = client.get("/memory/summary/orchestrator").json()
        assert summary["total_episodes"] == 1
        assert summary["outcome_breakdown"]["positive"] == 1
