"""
tests/unit/test_agent_router.py
─────────────────────────────────
Unit tests for AgentRouter.

Run with:  pytest tests/unit/test_agent_router.py -v
"""

import pytest
from pact_ax.primitives.capability_registry import CapabilityRegistry
from pact_ax.primitives.agent_router import AgentRouter, RouteDecision
from pact_ax.primitives.trust_score import TrustManager
from pact_ax.primitives.context_share.schemas import ContextType, CollaborationOutcome


@pytest.fixture
def cap_db(tmp_path):
    db = str(tmp_path / "caps.db")
    reg = CapabilityRegistry(db)
    reg.register("agent-alpha", "contract_review", "Expert NDA review",       tags=["legal"])
    reg.register("agent-beta",  "contract_review", "Service agreements",       tags=["legal"])
    reg.register("agent-gamma", "contract_review", "General contract review",  tags=["legal"])
    reg.register("agent-beta",  "tax_analysis",    "Federal + state tax",      tags=["finance"])
    reg.register("agent-delta", "ip_licensing",    "Patent licensing",         tags=["legal", "ip"])
    return db


@pytest.fixture
def trust_db(tmp_path):
    db = str(tmp_path / "trust.db")
    mgr = TrustManager.load(db, "orchestrator")
    # alpha: high trust; beta: medium; gamma: low
    for _ in range(2):
        mgr.update_trust("agent-alpha", CollaborationOutcome.POSITIVE,
                         ContextType.TASK_KNOWLEDGE, impact=1.0)
    mgr.update_trust("agent-beta",  CollaborationOutcome.POSITIVE,
                     ContextType.TASK_KNOWLEDGE, impact=0.6)
    mgr.update_trust("agent-gamma", CollaborationOutcome.NEGATIVE,
                     ContextType.TASK_KNOWLEDGE, impact=0.8)
    mgr.save(db)
    return db


@pytest.fixture
def router(cap_db, trust_db):
    return AgentRouter(capability_db=cap_db, trust_db=trust_db)


# ── route — basic ─────────────────────────────────────────────────────────────

class TestRoute:

    def test_returns_route_decision(self, router):
        d = router.route("orchestrator", "contract_review")
        assert isinstance(d, RouteDecision)

    def test_best_agent_is_most_trusted(self, router):
        d = router.route("orchestrator", "contract_review")
        assert d.best_agent == "agent-alpha"

    def test_candidates_sorted_by_trust_descending(self, router):
        d = router.route("orchestrator", "contract_review")
        scores = [c.trust_score for c in d.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_is_trust_weighted(self, router):
        d = router.route("orchestrator", "contract_review")
        assert d.strategy_used == "trust_weighted"

    def test_total_capable_counts_all(self, router):
        d = router.route("orchestrator", "contract_review")
        assert d.total_capable == 3

    def test_unknown_skill_returns_no_route(self, router):
        d = router.route("orchestrator", "nonexistent_skill")
        assert d.best_agent is None
        assert d.routed is False
        assert d.strategy_used == "none"

    def test_min_trust_filters_candidates(self, router):
        d = router.route("orchestrator", "contract_review", min_trust=0.6)
        for c in d.candidates:
            assert c.trust_score >= 0.6

    def test_min_trust_too_high_returns_no_route(self, router):
        d = router.route("orchestrator", "contract_review", min_trust=0.99)
        assert d.best_agent is None

    def test_top_k_limits_candidates(self, router):
        d = router.route("orchestrator", "contract_review", top_k=2)
        assert len(d.candidates) <= 2

    def test_does_not_route_to_self(self, router):
        d = router.route("agent-alpha", "contract_review")
        agent_ids = [c.agent_id for c in d.candidates]
        assert "agent-alpha" not in agent_ids


# ── route — no trust history ──────────────────────────────────────────────────

class TestRouteNoTrustHistory:

    def test_falls_back_to_capability_only(self, cap_db, tmp_path):
        empty_trust = str(tmp_path / "empty_trust.db")
        router = AgentRouter(capability_db=cap_db, trust_db=empty_trust)
        d = router.route("orchestrator", "contract_review")
        assert d.strategy_used == "capability_only"
        assert d.best_agent is not None  # still picks someone

    def test_default_trust_is_neutral(self, cap_db, tmp_path):
        empty_trust = str(tmp_path / "empty_trust.db")
        router = AgentRouter(capability_db=cap_db, trust_db=empty_trust)
        d = router.route("orchestrator", "contract_review")
        for c in d.candidates:
            assert abs(c.trust_score - 0.5) < 0.01


# ── route_any — fuzzy ─────────────────────────────────────────────────────────

class TestRouteAny:

    def test_finds_skill_by_keyword(self, router):
        d = router.route_any("orchestrator", "tax")
        assert d.best_agent is not None
        assert any(c.skill == "tax_analysis" for c in d.candidates)

    def test_finds_by_description_keyword(self, router):
        d = router.route_any("orchestrator", "NDA")
        assert d.best_agent is not None

    def test_finds_by_tag_keyword(self, router):
        d = router.route_any("orchestrator", "finance")
        assert d.best_agent is not None

    def test_no_match_returns_no_route(self, router):
        d = router.route_any("orchestrator", "zzznomatch")
        assert d.best_agent is None
        assert d.strategy_used == "none"

    def test_returns_route_decision(self, router):
        d = router.route_any("orchestrator", "legal")
        assert isinstance(d, RouteDecision)
