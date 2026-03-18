"""
tests/test_consensus.py
────────────────────────
Test suite for ConsensusProtocol.

Run with: pytest tests/test_consensus.py -v
"""

import pytest
from pact_ax.coordination.consensus import (
    ConsensusOutcome,
    ConsensusProtocol,
    ConsensusResult,
    ConsensusStrategy,
    Vote,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def unanimous_votes():
    return [
        Vote("agent-A", "deploy-v2", 0.9, "metrics stable"),
        Vote("agent-B", "deploy-v2", 0.8, "risk acceptable"),
        Vote("agent-C", "deploy-v2", 0.75, "checks pass"),
    ]

@pytest.fixture
def split_votes():
    return [
        Vote("agent-A", "deploy-v2", 0.85, "go"),
        Vote("agent-B", "deploy-v2", 0.80, "go"),
        Vote("agent-C", "hold",      0.65, "need more data"),
    ]

@pytest.fixture
def deadlock_votes():
    """Perfect 50/50 split — should fail to reach consensus."""
    return [
        Vote("agent-A", "option-X", 0.75, ""),
        Vote("agent-B", "option-Y", 0.75, ""),
    ]

@pytest.fixture
def trust_scores():
    return {"agent-A": 0.9, "agent-B": 0.85, "agent-C": 0.7}


# ──────────────────────────────────────────────────────────────────────────────
# Vote
# ──────────────────────────────────────────────────────────────────────────────

class TestVote:
    def test_valid_vote(self):
        v = Vote("agent-A", "deploy", 0.8, "reason")
        assert v.confidence == 0.8

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Vote("agent-A", "deploy", 1.5)

    def test_abstain_zeroes_weight(self):
        v = Vote("agent-A", "deploy", 0.9, abstain=True)
        assert v.effective_weight(trust_score=1.0) == 0.0

    def test_effective_weight_with_trust(self):
        v = Vote("agent-A", "deploy", 0.8)
        assert abs(v.effective_weight(0.9) - 0.72) < 1e-9

    def test_effective_weight_default_trust(self):
        v = Vote("agent-A", "deploy", 0.8)
        assert v.effective_weight() == 0.8


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTED_VOTE strategy
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightedVote:
    @pytest.fixture
    def proto(self):
        return ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)

    def test_unanimous_accepted(self, proto, unanimous_votes, trust_scores):
        result = proto.run(unanimous_votes, trust_scores=trust_scores)
        assert result.reached
        assert result.winning_decision == "deploy-v2"

    def test_majority_accepted(self, proto, split_votes, trust_scores):
        result = proto.run(split_votes, trust_scores=trust_scores)
        assert result.reached
        assert result.winning_decision == "deploy-v2"

    def test_result_has_confidence_score(self, proto, split_votes, trust_scores):
        result = proto.run(split_votes, trust_scores=trust_scores)
        assert 0.0 < result.confidence_score <= 1.0

    def test_vote_breakdown_sums_correctly(self, proto, split_votes, trust_scores):
        result = proto.run(split_votes, trust_scores=trust_scores)
        assert abs(sum(result.vote_breakdown.values()) - result.total_weight) < 1e-6

    def test_dissent_map_populated(self, proto, split_votes, trust_scores):
        result = proto.run(split_votes, trust_scores=trust_scores)
        assert "hold" in result.dissent_map
        assert "agent-C" in result.dissent_map["hold"]

    def test_result_serialisable(self, proto, split_votes, trust_scores):
        result = proto.run(split_votes, trust_scores=trust_scores)
        d = result.to_dict()
        assert d["outcome"] == ConsensusOutcome.ACCEPTED.value
        assert "vote_breakdown" in d

    def test_custom_round_id(self, proto, unanimous_votes):
        result = proto.run(unanimous_votes, round_id="my-round-42")
        assert result.round_id == "my-round-42"

    def test_abstentions_tracked(self, proto, trust_scores):
        votes = [
            Vote("agent-A", "deploy-v2", 0.9),
            Vote("agent-B", "deploy-v2", 0.8),
            Vote("agent-C", "hold",      0.6, abstain=True),
        ]
        result = proto.run(votes, trust_scores=trust_scores)
        assert "agent-C" in result.abstentions


# ──────────────────────────────────────────────────────────────────────────────
# QUORUM strategy
# ──────────────────────────────────────────────────────────────────────────────

class TestQuorum:
    @pytest.fixture
    def proto(self):
        return ConsensusProtocol(strategy=ConsensusStrategy.QUORUM, quorum_fraction=0.5)

    def test_majority_quorum_accepted(self, proto, split_votes):
        result = proto.run(split_votes)
        assert result.reached
        assert result.winning_decision == "deploy-v2"

    def test_exact_tie_fails_quorum(self, proto, deadlock_votes):
        result = proto.run(deadlock_votes)
        assert not result.reached

    def test_high_quorum_threshold_fails(self, split_votes):
        strict = ConsensusProtocol(strategy=ConsensusStrategy.QUORUM, quorum_fraction=0.9)
        result = strict.run(split_votes)
        assert not result.reached

    def test_unanimous_passes_any_quorum(self, unanimous_votes):
        proto = ConsensusProtocol(strategy=ConsensusStrategy.QUORUM, quorum_fraction=0.99)
        result = proto.run(unanimous_votes)
        assert result.reached


# ──────────────────────────────────────────────────────────────────────────────
# UNANIMOUS strategy
# ──────────────────────────────────────────────────────────────────────────────

class TestUnanimous:
    @pytest.fixture
    def proto(self):
        return ConsensusProtocol(strategy=ConsensusStrategy.UNANIMOUS)

    def test_unanimous_accepted(self, proto, unanimous_votes):
        result = proto.run(unanimous_votes)
        assert result.reached
        assert result.winning_decision == "deploy-v2"

    def test_one_dissenter_fails(self, proto, split_votes):
        result = proto.run(split_votes)
        assert not result.reached

    def test_two_way_split_escalates(self, proto, deadlock_votes):
        result = proto.run(deadlock_votes)
        assert result.outcome == ConsensusOutcome.ESCALATE_TO_HUMAN

    def test_three_way_split_deadlocks(self, proto):
        votes = [
            Vote("a", "X", 0.7),
            Vote("b", "Y", 0.7),
            Vote("c", "Z", 0.7),
        ]
        result = proto.run(votes)
        assert result.outcome == ConsensusOutcome.DEADLOCK


# ──────────────────────────────────────────────────────────────────────────────
# CONFIDENCE_THRESHOLD strategy
# ──────────────────────────────────────────────────────────────────────────────

class TestConfidenceThreshold:
    @pytest.fixture
    def proto(self):
        return ConsensusProtocol(
            strategy=ConsensusStrategy.CONFIDENCE_THRESHOLD,
            confidence_threshold=0.75,
        )

    def test_high_confidence_accepted(self, proto, unanimous_votes):
        result = proto.run(unanimous_votes)
        assert result.reached

    def test_low_confidence_fails(self, proto):
        votes = [
            Vote("a", "X", 0.4),
            Vote("b", "X", 0.45),
            Vote("c", "Y", 0.5),
        ]
        result = proto.run(votes)
        assert not result.reached


# ──────────────────────────────────────────────────────────────────────────────
# Insufficient votes
# ──────────────────────────────────────────────────────────────────────────────

class TestInsufficientVotes:
    def test_single_vote_insufficient(self):
        proto  = ConsensusProtocol(min_votes=2)
        result = proto.run([Vote("a", "X", 0.9)])
        assert result.outcome == ConsensusOutcome.INSUFFICIENT_VOTES

    def test_all_abstain_insufficient(self):
        proto = ConsensusProtocol(min_votes=1)
        votes = [Vote("a", "X", 0.9, abstain=True)]
        result = proto.run(votes)
        assert result.outcome == ConsensusOutcome.INSUFFICIENT_VOTES


# ──────────────────────────────────────────────────────────────────────────────
# History and metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_history_grows(self, unanimous_votes, split_votes):
        proto = ConsensusProtocol()
        proto.run(unanimous_votes)
        proto.run(split_votes)
        assert len(proto.history()) == 2

    def test_acceptance_rate(self, unanimous_votes):
        proto = ConsensusProtocol()
        proto.run(unanimous_votes)
        proto.run(unanimous_votes)
        assert proto.acceptance_rate() == 1.0

    def test_metrics_keys(self, unanimous_votes):
        proto = ConsensusProtocol()
        proto.run(unanimous_votes)
        m = proto.metrics()
        for key in ("strategy", "total_rounds", "acceptance_rate",
                    "escalation_rate", "avg_confidence", "outcome_breakdown"):
            assert key in m

    def test_reached_property(self, unanimous_votes):
        proto  = ConsensusProtocol()
        result = proto.run(unanimous_votes)
        assert result.reached is True

    def test_needs_human_property(self):
        proto = ConsensusProtocol(strategy=ConsensusStrategy.UNANIMOUS)
        votes = [Vote("a", "X", 0.7), Vote("b", "Y", 0.7)]
        result = proto.run(votes)
        assert result.needs_human is True

    @pytest.mark.parametrize("strategy", list(ConsensusStrategy))
    def test_all_strategies_return_result(self, strategy, unanimous_votes):
        proto  = ConsensusProtocol(strategy=strategy)
        result = proto.run(unanimous_votes)
        assert isinstance(result, ConsensusResult)
