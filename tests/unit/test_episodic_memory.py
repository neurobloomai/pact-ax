"""
tests/unit/test_episodic_memory.py
────────────────────────────────────
Unit tests for EpisodicMemory.

Run with:  pytest tests/unit/test_episodic_memory.py -v
"""

import pytest
from pact_ax.primitives.episodic_memory import EpisodicMemory, Episode, Outcome, Valence


@pytest.fixture
def mem(tmp_path):
    return EpisodicMemory(tmp_path / "episodic.db")


@pytest.fixture
def populated(mem):
    mem.record("agent-a", "contract_review", partner_id="agent-b",
               outcome=Outcome.POSITIVE, importance=0.9, tags=["legal", "nda"])
    mem.record("agent-a", "handoff",         partner_id="agent-b",
               outcome=Outcome.NEUTRAL,  importance=0.5, tags=["ops"])
    mem.record("agent-a", "tax_analysis",    partner_id="agent-c",
               outcome=Outcome.NEGATIVE, importance=0.7, tags=["finance"])
    mem.record("agent-a", "solo_action",     partner_id="",
               outcome=Outcome.POSITIVE, importance=0.3, tags=[])
    return mem


# ── record ────────────────────────────────────────────────────────────────────

class TestRecord:

    def test_returns_episode(self, mem):
        ep = mem.record("agent-a", "contract_review")
        assert isinstance(ep, Episode)

    def test_fields_stored_correctly(self, mem):
        ep = mem.record("agent-a", "review",
                        partner_id="agent-b",
                        outcome=Outcome.POSITIVE,
                        importance=0.8,
                        valence=Valence.POSITIVE,
                        session_id="sess-1",
                        tags=["legal"],
                        context={"jurisdiction": "CA"})
        assert ep.agent_id   == "agent-a"
        assert ep.partner_id == "agent-b"
        assert ep.action     == "review"
        assert ep.outcome    == Outcome.POSITIVE
        assert ep.importance == 0.8
        assert ep.valence    == Valence.POSITIVE
        assert ep.session_id == "sess-1"
        assert "legal" in ep.tags
        assert ep.context["jurisdiction"] == "CA"

    def test_importance_clamped_to_zero(self, mem):
        ep = mem.record("agent-a", "action", importance=-0.5)
        assert ep.importance == 0.0

    def test_importance_clamped_to_one(self, mem):
        ep = mem.record("agent-a", "action", importance=1.5)
        assert ep.importance == 1.0

    def test_empty_partner_allowed(self, mem):
        ep = mem.record("agent-a", "solo")
        assert ep.partner_id == ""


# ── recall ────────────────────────────────────────────────────────────────────

class TestRecall:

    def test_returns_all_for_agent(self, populated):
        eps = populated.recall("agent-a")
        assert len(eps) == 4

    def test_filters_by_partner(self, populated):
        eps = populated.recall("agent-a", partner_id="agent-b")
        assert all(e.partner_id == "agent-b" for e in eps)
        assert len(eps) == 2

    def test_filters_by_outcome(self, populated):
        eps = populated.recall("agent-a", outcome=Outcome.POSITIVE)
        assert all(e.outcome == Outcome.POSITIVE for e in eps)

    def test_filters_by_min_importance(self, populated):
        eps = populated.recall("agent-a", min_importance=0.7)
        assert all(e.importance >= 0.7 for e in eps)

    def test_ordered_by_importance_desc(self, populated):
        eps = populated.recall("agent-a")
        scores = [e.importance for e in eps]
        assert scores == sorted(scores, reverse=True)

    def test_limit(self, populated):
        eps = populated.recall("agent-a", limit=2)
        assert len(eps) <= 2

    def test_empty_for_unknown_agent(self, populated):
        assert populated.recall("agent-z") == []

    def test_filters_by_tags(self, populated):
        eps = populated.recall("agent-a", tags=["legal"])
        assert all("legal" in e.tags for e in eps)


# ── recall_partner ────────────────────────────────────────────────────────────

class TestRecallPartner:

    def test_returns_partner_episodes(self, populated):
        eps = populated.recall_partner("agent-a", "agent-b")
        assert len(eps) == 2
        assert all(e.partner_id == "agent-b" for e in eps)

    def test_no_episodes_for_unknown_partner(self, populated):
        assert populated.recall_partner("agent-a", "agent-z") == []


# ── summary ───────────────────────────────────────────────────────────────────

class TestSummary:

    def test_total_count(self, populated):
        s = populated.summary("agent-a")
        assert s["total_episodes"] == 4

    def test_outcome_breakdown(self, populated):
        s = populated.summary("agent-a")
        assert s["outcome_breakdown"][Outcome.POSITIVE] == 2
        assert s["outcome_breakdown"][Outcome.NEUTRAL]  == 1
        assert s["outcome_breakdown"][Outcome.NEGATIVE] == 1

    def test_top_partners(self, populated):
        s = populated.summary("agent-a")
        partner_ids = [p["partner_id"] for p in s["top_partners"]]
        assert "agent-b" in partner_ids

    def test_avg_importance(self, populated):
        s = populated.summary("agent-a")
        assert 0.0 < s["avg_importance"] <= 1.0

    def test_empty_agent_summary(self, mem):
        s = mem.summary("nobody")
        assert s["total_episodes"] == 0
        assert s["outcome_breakdown"] == {}


# ── delete / clear ────────────────────────────────────────────────────────────

class TestDeleteAndClear:

    def test_delete_episode(self, populated):
        eps = populated.recall("agent-a")
        ep_id = eps[0].id
        assert populated.delete_episode(ep_id) is True
        remaining = populated.recall("agent-a")
        assert not any(e.id == ep_id for e in remaining)

    def test_delete_nonexistent_returns_false(self, mem):
        assert mem.delete_episode("nonexistent-id") is False

    def test_clear_removes_all_for_agent(self, populated):
        count = populated.clear("agent-a")
        assert count == 4
        assert populated.recall("agent-a") == []

    def test_clear_does_not_affect_other_agents(self, mem):
        mem.record("agent-a", "action")
        mem.record("agent-b", "action")
        mem.clear("agent-a")
        assert len(mem.recall("agent-b")) == 1


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_survives_new_instance(self, tmp_path):
        db = tmp_path / "ep.db"
        m1 = EpisodicMemory(db)
        m1.record("agent-a", "review", partner_id="agent-b",
                  outcome="positive", tags=["legal"], context={"x": 1})
        m2 = EpisodicMemory(db)
        eps = m2.recall("agent-a")
        assert len(eps) == 1
        assert eps[0].context == {"x": 1}
        assert "legal" in eps[0].tags
