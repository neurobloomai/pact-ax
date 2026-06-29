"""
Unit tests for TrustChain primitive.

Covers: scoring, recording, verification, drift detection,
        state transitions, persistence, and edge cases.
"""

import math
import pytest
from pact_ax.primitives.trust_chain import (
    TrustChainManager,
    TrustChain,
    ChainState,
    ChainScore,
    ChainVerification,
    _geometric_mean,
    _coherence,
    _chain_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_resolver(scores: dict):
    """Build a trust resolver from a {(from, to): score} dict."""
    def resolver(from_agent: str, to_agent: str) -> float:
        return scores.get((from_agent, to_agent), 0.5)
    return resolver


@pytest.fixture
def high_trust_resolver():
    return make_resolver({
        ("a", "b"): 0.9,
        ("b", "c"): 0.85,
        ("c", "d"): 0.88,
    })


@pytest.fixture
def weak_link_resolver():
    return make_resolver({
        ("a", "b"): 0.9,
        ("b", "c"): 0.3,   # weak link
        ("c", "d"): 0.9,
    })


@pytest.fixture
def mgr(high_trust_resolver):
    return TrustChainManager(trust_resolver=high_trust_resolver)


# ── Scoring helpers ───────────────────────────────────────────────────────────

class TestScoringHelpers:
    def test_geometric_mean_uniform(self):
        scores = [0.9, 0.9, 0.9]
        assert abs(_geometric_mean(scores) - 0.9) < 1e-6

    def test_geometric_mean_penalises_weak_link(self):
        uniform = _geometric_mean([0.8, 0.8, 0.8])
        weak    = _geometric_mean([0.9, 0.3, 0.9])
        assert weak < uniform

    def test_geometric_mean_empty(self):
        assert _geometric_mean([]) == 0.0

    def test_geometric_mean_zero_score(self):
        assert _geometric_mean([0.9, 0.0, 0.8]) == 0.0

    def test_coherence_uniform_is_one(self):
        assert _coherence([0.8, 0.8, 0.8]) == 1.0

    def test_coherence_single_hop(self):
        assert _coherence([0.7]) == 1.0

    def test_coherence_max_variance(self):
        # [0.0, 1.0] std=0.5 → coherence = 0.0
        assert _coherence([0.0, 1.0]) == 0.0

    def test_coherence_decreases_with_variance(self):
        high = _coherence([0.8, 0.85, 0.82])
        low  = _coherence([0.9, 0.1, 0.9])
        assert high > low

    def test_chain_state_active(self):
        assert _chain_state(0.85, 0.80) == ChainState.ACTIVE

    def test_chain_state_degraded_low_trust(self):
        assert _chain_state(0.55, 0.80) == ChainState.DEGRADED

    def test_chain_state_degraded_low_coherence(self):
        assert _chain_state(0.80, 0.50) == ChainState.DEGRADED

    def test_chain_state_broken(self):
        assert _chain_state(0.30, 0.90) == ChainState.BROKEN


# ── Score (no recording) ──────────────────────────────────────────────────────

class TestScore:
    def test_score_returns_chain_score(self, mgr):
        s = mgr.score(["a", "b", "c"])
        assert isinstance(s, ChainScore)
        assert s.agents == ["a", "b", "c"]
        assert len(s.hop_scores) == 2
        assert 0.0 <= s.chain_trust <= 1.0
        assert 0.0 <= s.coherence <= 1.0

    def test_score_two_agents(self, mgr):
        s = mgr.score(["a", "b"])
        assert len(s.hop_scores) == 1
        assert s.hop_scores[0] == pytest.approx(0.9)

    def test_score_identifies_weakest_hop(self):
        resolver = make_resolver({("a","b"): 0.9, ("b","c"): 0.3, ("c","d"): 0.9})
        mgr = TrustChainManager(trust_resolver=resolver)
        s = mgr.score(["a", "b", "c", "d"])
        assert s.weakest_pair == ("b", "c")
        assert s.weakest_index == 1

    def test_score_does_not_record(self, mgr):
        mgr.score(["a", "b", "c"])
        assert len(mgr.list_chains()) == 0

    def test_score_requires_at_least_two_agents(self, mgr):
        with pytest.raises(ValueError, match="at least 2 agents"):
            mgr.score(["a"])

    def test_score_rejects_cycles(self, mgr):
        with pytest.raises(ValueError, match="cycles"):
            mgr.score(["a", "b", "a"])

    def test_score_state_active_for_high_trust(self, mgr):
        s = mgr.score(["a", "b", "c"])
        assert s.state == ChainState.ACTIVE

    def test_score_state_broken_for_weak_chain(self):
        resolver = make_resolver({("a","b"): 0.2, ("b","c"): 0.2})
        mgr = TrustChainManager(trust_resolver=resolver)
        s = mgr.score(["a", "b", "c"])
        assert s.state == ChainState.BROKEN


# ── Record ────────────────────────────────────────────────────────────────────

class TestRecord:
    def test_record_returns_trust_chain(self, mgr):
        chain = mgr.record(["a", "b", "c"])
        assert isinstance(chain, TrustChain)
        assert chain.depth == 2
        assert chain.agents == ["a", "b", "c"]

    def test_record_persists_chain(self, mgr):
        chain = mgr.record(["a", "b"])
        retrieved = mgr.get(chain.chain_id)
        assert retrieved.chain_id == chain.chain_id

    def test_record_custom_chain_id(self, mgr):
        chain = mgr.record(["a", "b"], chain_id="my-chain-001")
        assert chain.chain_id == "my-chain-001"

    def test_record_baseline_scores_set(self, mgr):
        chain = mgr.record(["a", "b", "c"])
        assert chain.hops[0].baseline_score == pytest.approx(0.9)
        assert chain.hops[1].baseline_score == pytest.approx(0.85)

    def test_record_weakest_hop(self):
        resolver = make_resolver({("a","b"): 0.9, ("b","c"): 0.3})
        mgr = TrustChainManager(trust_resolver=resolver)
        chain = mgr.record(["a", "b", "c"])
        assert chain.weakest_hop.from_agent == "b"
        assert chain.weakest_hop.to_agent   == "c"

    def test_record_four_hop_chain(self, mgr):
        chain = mgr.record(["a", "b", "c", "d"])
        assert chain.depth == 3
        assert len(chain.hops) == 3


# ── Verify ────────────────────────────────────────────────────────────────────

class TestVerify:
    def test_verify_no_drift_returns_same_state(self, mgr):
        chain = mgr.record(["a", "b", "c"])
        v = mgr.verify(chain.chain_id)
        assert isinstance(v, ChainVerification)
        assert v.state_changed is False
        assert v.current_state == chain.state

    def test_verify_detects_trust_decay(self):
        scores = {("a","b"): 0.9, ("b","c"): 0.85}
        resolver = make_resolver(scores)
        mgr = TrustChainManager(trust_resolver=resolver, drift_threshold=0.05)
        chain = mgr.record(["a", "b", "c"])

        # Simulate trust decay on b→c
        scores[("b","c")] = 0.3
        v = mgr.verify(chain.chain_id)

        assert v.state_changed is True
        assert v.current_state in (ChainState.DEGRADED, ChainState.BROKEN)
        bc_drift = next(d for d in v.hop_drift if d.from_agent == "b")
        assert bc_drift.drifted is True
        assert bc_drift.drift < 0

    def test_verify_updates_chain_state_in_place(self):
        scores = {("a","b"): 0.9, ("b","c"): 0.85}
        resolver = make_resolver(scores)
        mgr = TrustChainManager(trust_resolver=resolver)
        chain = mgr.record(["a", "b", "c"])

        scores[("b","c")] = 0.2
        mgr.verify(chain.chain_id)

        updated = mgr.get(chain.chain_id)
        assert updated.state in (ChainState.DEGRADED, ChainState.BROKEN)

    def test_verify_no_false_positive_below_threshold(self):
        scores = {("a","b"): 0.9, ("b","c"): 0.85}
        resolver = make_resolver(scores)
        mgr = TrustChainManager(trust_resolver=resolver, drift_threshold=0.05)
        chain = mgr.record(["a", "b", "c"])

        # Tiny nudge — below drift_threshold
        scores[("b","c")] = 0.87
        v = mgr.verify(chain.chain_id)

        bc_drift = next(d for d in v.hop_drift if d.from_agent == "b")
        assert bc_drift.drifted is False

    def test_verify_unknown_chain_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.verify("nonexistent-id")


# ── Complete ──────────────────────────────────────────────────────────────────

class TestComplete:
    def test_complete_marks_chain_completed(self, mgr):
        chain = mgr.record(["a", "b"])
        mgr.complete(chain.chain_id)
        assert mgr.get(chain.chain_id).state == ChainState.COMPLETED

    def test_complete_unknown_chain_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.complete("ghost-id")


# ── List chains ───────────────────────────────────────────────────────────────

class TestListChains:
    def test_list_all_chains(self, mgr):
        mgr.record(["a", "b"])
        mgr.record(["b", "c"])
        assert len(mgr.list_chains()) == 2

    def test_list_chains_by_agent(self, mgr):
        mgr.record(["a", "b"])
        mgr.record(["c", "d"])
        chains = mgr.list_chains(agent_id="a")
        assert len(chains) == 1
        assert "a" in chains[0].agents


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, mgr, tmp_path):
        chain = mgr.record(["a", "b", "c"])
        db = str(tmp_path / "test_chains.db")
        mgr.save(db)

        resolver = make_resolver({("a","b"): 0.9, ("b","c"): 0.85})
        loaded = TrustChainManager.load(db, trust_resolver=resolver)
        restored = loaded.get(chain.chain_id)

        assert restored.chain_id    == chain.chain_id
        assert restored.chain_trust == pytest.approx(chain.chain_trust, abs=1e-4)
        assert restored.state       == chain.state
        assert len(restored.hops)   == len(chain.hops)

    def test_load_nonexistent_db_returns_empty_manager(self, tmp_path):
        resolver = make_resolver({})
        mgr = TrustChainManager.load(
            str(tmp_path / "ghost.db"), trust_resolver=resolver
        )
        assert mgr.list_chains() == []
