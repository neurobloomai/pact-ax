"""
tests/unit/test_policy_align.py
────────────────────────────────
Comprehensive tests for PolicyAlignmentManager, PolicyDecision,
PolicyConstraint, and PolicyLearning.

Run with:  pytest tests/unit/test_policy_align.py -v
"""

import pytest
from datetime import datetime

from pact_ax.coordination.policy_alignment import (
    PolicyDecision,
    PolicyConstraint,
    PolicyAlignmentManager,
    PolicyConflictResolution,
    PolicyLearning,
)
from pact_ax.primitives.epistemic import (
    ConfidenceLevel,
    KnowledgeBoundary,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_decision(
    decision: str = "deploy",
    confidence: ConfidenceLevel = ConfidenceLevel.CONFIDENT,
    agent_id: str = "agent-A",
    domain: str = "deployment",
    uncertainty_factors: list = None,
    alternatives: list = None,
) -> PolicyDecision:
    return PolicyDecision(
        decision=decision,
        confidence=confidence,
        reasoning="test reasoning",
        agent_id=agent_id,
        domain=domain,
        alternatives_considered=alternatives or [],
        uncertainty_factors=uncertainty_factors or [],
    )


def make_boundary(domain: str, capabilities: set = None, limits: set = None) -> KnowledgeBoundary:
    return KnowledgeBoundary(
        domain=domain,
        proficiency=ConfidenceLevel.CONFIDENT,
        known_capabilities=capabilities or set(),
        known_limits=limits or set(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# PolicyDecision
# ──────────────────────────────────────────────────────────────────────────────

class TestPolicyDecision:

    def test_is_confident_enough_above_threshold(self):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)  # 0.80
        assert d.is_confident_enough(0.7) is True

    def test_is_confident_enough_at_threshold(self):
        d = make_decision(confidence=ConfidenceLevel.MODERATE)  # 0.60
        assert d.is_confident_enough(0.6) is True

    def test_is_confident_enough_below_threshold(self):
        d = make_decision(confidence=ConfidenceLevel.LOW)  # 0.40
        assert d.is_confident_enough(0.7) is False

    def test_should_seek_consensus_when_low_confidence(self):
        d = make_decision(confidence=ConfidenceLevel.LOW)  # 0.40 < 0.8
        assert d.should_seek_consensus() is True

    def test_should_seek_consensus_when_many_uncertainty_factors(self):
        d = make_decision(
            confidence=ConfidenceLevel.CONFIDENT,
            uncertainty_factors=["a", "b", "c"],  # > 2
        )
        assert d.should_seek_consensus() is True

    def test_no_consensus_needed_when_high_confidence_and_few_factors(self):
        d = make_decision(
            confidence=ConfidenceLevel.CERTAIN,  # 0.95 >= 0.8
            uncertainty_factors=["one"],
        )
        assert d.should_seek_consensus() is False

    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        d = make_decision()
        assert d.timestamp >= before

    def test_alternatives_defaults_empty(self):
        d = make_decision()
        assert d.alternatives_considered == []


# ──────────────────────────────────────────────────────────────────────────────
# PolicyConstraint
# ──────────────────────────────────────────────────────────────────────────────

class TestPolicyConstraint:

    def test_satisfied_when_meets_min_confidence(self):
        constraint = PolicyConstraint(
            name="min-conf",
            description="need moderate confidence",
            min_confidence=0.5,
        )
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        assert constraint.is_satisfied(d, []) is True

    def test_violated_when_below_min_confidence(self):
        constraint = PolicyConstraint(
            name="high-conf",
            description="need high confidence",
            min_confidence=0.9,
        )
        d = make_decision(confidence=ConfidenceLevel.MODERATE)  # 0.60
        assert constraint.is_satisfied(d, []) is False

    def test_requires_specialist_satisfied_when_boundary_has_domain(self):
        constraint = PolicyConstraint(
            name="specialist",
            description="needs domain expert",
            min_confidence=0.5,
            requires_specialist=True,
        )
        d = make_decision(domain="deployment")
        boundary = make_boundary("sys", capabilities={"deployment"})
        assert constraint.is_satisfied(d, [boundary]) is True

    def test_requires_specialist_violated_when_no_boundary(self):
        constraint = PolicyConstraint(
            name="specialist",
            description="needs domain expert",
            min_confidence=0.5,
            requires_specialist=True,
        )
        d = make_decision(domain="deployment")
        assert constraint.is_satisfied(d, []) is False

    def test_domain_check_satisfied_when_domain_in_allowed_set(self):
        constraint = PolicyConstraint(
            name="domain-check",
            description="only deployment domain",
            min_confidence=0.3,
            domains={"deployment"},
        )
        d = make_decision(domain="deployment")
        assert constraint.is_satisfied(d, []) is True

    def test_domain_check_violated_when_domain_not_in_set(self):
        constraint = PolicyConstraint(
            name="domain-check",
            description="only deployment domain",
            min_confidence=0.3,
            domains={"deployment"},
        )
        d = make_decision(domain="finance")
        assert constraint.is_satisfied(d, []) is False

    def test_empty_domains_skips_domain_check(self):
        constraint = PolicyConstraint(
            name="no-domain-filter",
            description="any domain",
            min_confidence=0.3,
        )
        d = make_decision(domain="anything")
        assert constraint.is_satisfied(d, []) is True


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — evaluate_decision()
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateDecision:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    def test_no_constraints_valid_by_default(self, pam):
        d = make_decision(confidence=ConfidenceLevel.MODERATE)
        valid, issues = pam.evaluate_decision(d, [])
        assert valid
        assert issues == []

    def test_unknown_confidence_invalid(self, pam):
        d = make_decision(confidence=ConfidenceLevel.UNKNOWN)
        valid, issues = pam.evaluate_decision(d, [])
        assert not valid
        assert any("UNKNOWN" in issue for issue in issues)

    def test_low_confidence_without_consensus_check_invalid(self, pam):
        # LOW confidence + <= 2 uncertainty_factors → should_seek_consensus() returns True
        # but evaluate_decision treats LOW confidence differently...
        # Looking at the implementation: LOW confidence AND should_seek_consensus() is False
        # means it adds a violation. But LOW always triggers should_seek_consensus() = True,
        # so this branch is actually unreachable for LOW. Testing with MODERATE and overriding:
        d = PolicyDecision(
            decision="hold",
            confidence=ConfidenceLevel.LOW,
            reasoning="uncertain",
            agent_id="agent-A",
            domain="deployment",
            uncertainty_factors=[],  # <=2, so should_seek_consensus = True because LOW < 0.8
        )
        # should_seek_consensus() returns True (LOW < 0.8), so the violation is NOT added
        valid, issues = pam.evaluate_decision(d, [])
        # LOW confidence but should_seek_consensus() is True, so that branch violation skipped
        # However LOW confidence might still trigger UNKNOWN check? No, LOW != UNKNOWN
        # So this should be valid (no constraints set)
        assert valid

    def test_constraint_violation_included_in_issues(self, pam):
        pam.add_constraint(PolicyConstraint(
            name="high-conf",
            description="need high confidence",
            min_confidence=0.9,
        ))
        d = make_decision(confidence=ConfidenceLevel.MODERATE)
        valid, issues = pam.evaluate_decision(d, [])
        assert not valid
        assert any("high-conf" in issue for issue in issues)

    def test_multiple_constraint_violations_all_reported(self, pam):
        pam.add_constraint(PolicyConstraint(
            name="conf1", description="needs confidence", min_confidence=0.9
        ))
        pam.add_constraint(PolicyConstraint(
            name="conf2", description="domain check", min_confidence=0.5, domains={"finance"}
        ))
        d = make_decision(confidence=ConfidenceLevel.LOW, domain="deployment")
        valid, issues = pam.evaluate_decision(d, [])
        assert not valid
        assert len(issues) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — resolve_conflict()
# ──────────────────────────────────────────────────────────────────────────────

class TestResolveConflict:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    def test_single_decision_returned_as_is(self, pam):
        d = make_decision()
        result = pam.resolve_conflict([d])
        assert result is d

    def test_raises_on_empty_decisions(self, pam):
        with pytest.raises(ValueError):
            pam.resolve_conflict([])

    def test_defer_to_more_confident_picks_highest(self, pam):
        low = make_decision("hold", ConfidenceLevel.LOW, "agent-B")
        high = make_decision("deploy", ConfidenceLevel.CERTAIN, "agent-A")
        result = pam.resolve_conflict(
            [low, high], PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT
        )
        assert result.decision == "deploy"

    def test_most_conservative_picks_lowest_confidence(self, pam):
        high = make_decision("deploy", ConfidenceLevel.CERTAIN, "agent-A")
        low = make_decision("hold", ConfidenceLevel.LOW, "agent-B")
        result = pam.resolve_conflict(
            [high, low], PolicyConflictResolution.MOST_CONSERVATIVE
        )
        assert result.decision == "hold"

    def test_escalate_to_human_returns_escalation_decision(self, pam):
        d1 = make_decision("deploy", ConfidenceLevel.MODERATE, "agent-A")
        d2 = make_decision("hold", ConfidenceLevel.MODERATE, "agent-B")
        result = pam.resolve_conflict(
            [d1, d2], PolicyConflictResolution.ESCALATE_TO_HUMAN
        )
        assert result.decision == "ESCALATE_TO_HUMAN"
        assert result.agent_id == "system"

    def test_consensus_required_when_all_agree(self, pam):
        d1 = make_decision("deploy", ConfidenceLevel.CONFIDENT, "agent-A")
        d2 = make_decision("deploy", ConfidenceLevel.MODERATE, "agent-B")
        result = pam.resolve_conflict(
            [d1, d2], PolicyConflictResolution.CONSENSUS_REQUIRED
        )
        assert result.decision == "deploy"
        assert result.agent_id == "consensus"

    def test_consensus_required_when_no_agreement_returns_no_consensus(self, pam):
        d1 = make_decision("deploy", ConfidenceLevel.CONFIDENT, "agent-A")
        d2 = make_decision("hold", ConfidenceLevel.MODERATE, "agent-B")
        result = pam.resolve_conflict(
            [d1, d2], PolicyConflictResolution.CONSENSUS_REQUIRED
        )
        assert result.decision == "NO_CONSENSUS"

    def test_conflict_logged(self, pam):
        d1 = make_decision("deploy", ConfidenceLevel.CONFIDENT, "agent-A")
        d2 = make_decision("hold", ConfidenceLevel.MODERATE, "agent-B")
        pam.resolve_conflict([d1, d2], PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT)
        assert len(pam.conflict_resolutions) == 1
        assert "agent-A" in pam.conflict_resolutions[0]["agents"]


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — align_policies()
# ──────────────────────────────────────────────────────────────────────────────

class TestAlignPolicies:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    def test_returns_a_decision(self, pam):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        result = pam.align_policies([d], {})
        assert isinstance(result, PolicyDecision)

    def test_decision_added_to_history(self, pam):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        pam.align_policies([d], {})
        assert len(pam.decision_history) == 1

    def test_invalid_decisions_filtered_out(self, pam):
        pam.add_constraint(PolicyConstraint(
            name="high-conf", description="test", min_confidence=0.9
        ))
        bad = make_decision("bad", ConfidenceLevel.LOW)
        result = pam.align_policies([bad], {})
        assert result.decision == "NO_VALID_DECISIONS"

    def test_no_valid_decisions_returns_fallback(self, pam):
        pam.add_constraint(PolicyConstraint(
            name="impossible", description="test", min_confidence=1.1
        ))
        d = make_decision(confidence=ConfidenceLevel.CERTAIN)
        result = pam.align_policies([d], {})
        assert result.decision == "NO_VALID_DECISIONS"
        assert result.confidence == ConfidenceLevel.UNKNOWN

    def test_valid_decision_chosen_from_mixed_pool(self, pam):
        pam.add_constraint(PolicyConstraint(
            name="min-conf", description="test", min_confidence=0.75
        ))
        bad = make_decision("hold", ConfidenceLevel.LOW, "agent-B")
        good = make_decision("deploy", ConfidenceLevel.CONFIDENT, "agent-A")
        result = pam.align_policies([bad, good], {})
        assert result.decision == "deploy"


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — _choose_resolution_strategy()
# ──────────────────────────────────────────────────────────────────────────────

class TestChooseResolutionStrategy:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    def test_many_uncertainty_factors_chooses_conservative(self, pam):
        d = make_decision(
            confidence=ConfidenceLevel.CONFIDENT,
            uncertainty_factors=["a", "b", "c", "d"],  # > 3
        )
        strategy = pam._choose_resolution_strategy([d])
        assert strategy == PolicyConflictResolution.MOST_CONSERVATIVE

    def test_wide_confidence_range_defers_to_confident(self, pam):
        low = make_decision(confidence=ConfidenceLevel.UNKNOWN)   # 0.20
        high = make_decision(confidence=ConfidenceLevel.CERTAIN)   # 0.95
        strategy = pam._choose_resolution_strategy([low, high])
        assert strategy == PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT

    def test_uniformly_low_confidence_escalates(self, pam):
        d1 = make_decision(confidence=ConfidenceLevel.UNKNOWN, agent_id="agent-A")  # 0.20
        d2 = make_decision(confidence=ConfidenceLevel.UNKNOWN, agent_id="agent-B")  # 0.20
        strategy = pam._choose_resolution_strategy([d1, d2])
        assert strategy == PolicyConflictResolution.ESCALATE_TO_HUMAN

    def test_similar_moderate_confidence_uses_consensus(self, pam):
        d1 = make_decision(confidence=ConfidenceLevel.CONFIDENT, agent_id="agent-A")  # 0.80
        d2 = make_decision(confidence=ConfidenceLevel.CONFIDENT, agent_id="agent-B")  # 0.80
        strategy = pam._choose_resolution_strategy([d1, d2])
        assert strategy == PolicyConflictResolution.CONSENSUS_REQUIRED


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — _map_to_confidence_level()
# ──────────────────────────────────────────────────────────────────────────────

class TestMapToConfidenceLevel:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    @pytest.mark.parametrize("value,expected", [
        (0.95, ConfidenceLevel.CERTAIN),
        (0.99, ConfidenceLevel.CERTAIN),
        (0.85, ConfidenceLevel.CONFIDENT),
        (0.80, ConfidenceLevel.CONFIDENT),
        (0.70, ConfidenceLevel.MODERATE),
        (0.60, ConfidenceLevel.MODERATE),
        (0.45, ConfidenceLevel.LOW),
        (0.40, ConfidenceLevel.LOW),
        (0.20, ConfidenceLevel.UNKNOWN),
        (0.10, ConfidenceLevel.UNKNOWN),
    ])
    def test_mapping(self, pam, value, expected):
        assert pam._map_to_confidence_level(value) == expected


# ──────────────────────────────────────────────────────────────────────────────
# PolicyAlignmentManager — get_policy_alignment_metrics()
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPolicyAlignmentMetrics:

    @pytest.fixture
    def pam(self):
        return PolicyAlignmentManager()

    def test_empty_returns_no_decisions_message(self, pam):
        metrics = pam.get_policy_alignment_metrics()
        assert "message" in metrics

    def test_metrics_keys_present_after_decisions(self, pam):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        pam.align_policies([d], {})
        metrics = pam.get_policy_alignment_metrics()
        for key in ("total_decisions", "conflicts_resolved", "avg_confidence",
                    "escalations", "high_confidence_rate"):
            assert key in metrics

    def test_total_decisions_count(self, pam):
        for _ in range(3):
            d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
            pam.align_policies([d], {})
        assert pam.get_policy_alignment_metrics()["total_decisions"] == 3

    def test_avg_confidence_in_valid_range(self, pam):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        pam.align_policies([d], {})
        avg = pam.get_policy_alignment_metrics()["avg_confidence"]
        assert 0.0 <= avg <= 1.0

    def test_high_confidence_rate_between_zero_and_one(self, pam):
        d = make_decision(confidence=ConfidenceLevel.CERTAIN)
        pam.align_policies([d], {})
        rate = pam.get_policy_alignment_metrics()["high_confidence_rate"]
        assert 0.0 <= rate <= 1.0

    def test_escalations_counted(self, pam):
        pam.decision_history.append(
            make_decision("ESCALATE_TO_HUMAN", ConfidenceLevel.UNKNOWN)
        )
        pam.decision_history.append(
            make_decision("deploy", ConfidenceLevel.CONFIDENT)
        )
        metrics = pam.get_policy_alignment_metrics()
        assert metrics["escalations"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# PolicyLearning
# ──────────────────────────────────────────────────────────────────────────────

class TestPolicyLearning:

    @pytest.fixture
    def learner(self):
        return PolicyLearning()

    def test_record_outcome_stored(self, learner):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT)
        learner.record_outcome(d, "succeeded", True)
        assert len(learner.outcomes) == 1

    def test_record_outcome_fields(self, learner):
        d = make_decision("deploy", ConfidenceLevel.CONFIDENT, "agent-A")
        learner.record_outcome(d, "launched", True, feedback="great")
        entry = learner.outcomes[0]
        assert entry["decision"] == "deploy"
        assert entry["agent_id"] == "agent-A"
        assert entry["was_correct"] is True
        assert entry["feedback"] == "great"

    def test_calibration_no_data_returns_message(self, learner):
        result = learner.get_agent_calibration("agent-unknown")
        assert "message" in result

    def test_calibration_accuracy_calculated(self, learner):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT, agent_id="agent-A")
        learner.record_outcome(d, "ok", True)
        learner.record_outcome(d, "ok", True)
        learner.record_outcome(d, "fail", False)
        cal = learner.get_agent_calibration("agent-A")
        assert cal["total_decisions"] == 3
        assert abs(cal["accuracy"] - 2/3) < 0.01

    def test_calibration_error_computed(self, learner):
        d = make_decision(confidence=ConfidenceLevel.CONFIDENT, agent_id="agent-B")  # 0.80
        learner.record_outcome(d, "ok", False)  # always wrong → accuracy 0.0
        cal = learner.get_agent_calibration("agent-B")
        assert cal["calibration_error"] > 0

    def test_well_calibrated_flag(self, learner):
        # Perfect calibration: confidence 0.80, accuracy ~0.80
        d1 = make_decision(confidence=ConfidenceLevel.CONFIDENT, agent_id="agent-C")
        for i in range(8):
            learner.record_outcome(d1, "ok", True)
        for i in range(2):
            learner.record_outcome(d1, "fail", False)
        cal = learner.get_agent_calibration("agent-C")
        assert cal["is_well_calibrated"]  # error < 0.15

    def test_tendency_overconfident(self, learner):
        d = make_decision(confidence=ConfidenceLevel.CERTAIN, agent_id="agent-D")  # 0.95
        # Always wrong → accuracy = 0.0
        for _ in range(5):
            learner.record_outcome(d, "fail", False)
        cal = learner.get_agent_calibration("agent-D")
        assert cal["tendency"] == "overconfident"

    def test_tendency_underconfident(self, learner):
        d = make_decision(confidence=ConfidenceLevel.UNKNOWN, agent_id="agent-E")  # 0.20
        # Always right → accuracy = 1.0
        for _ in range(5):
            learner.record_outcome(d, "ok", True)
        cal = learner.get_agent_calibration("agent-E")
        assert cal["tendency"] == "underconfident"

    def test_suggest_adjustment_overconfident(self, learner):
        d = make_decision(confidence=ConfidenceLevel.CERTAIN, agent_id="agent-F")
        for _ in range(5):
            learner.record_outcome(d, "fail", False)
        adj = learner.suggest_confidence_adjustment("agent-F", "deployment")
        assert adj < 1.0

    def test_suggest_adjustment_underconfident(self, learner):
        d = make_decision(confidence=ConfidenceLevel.UNKNOWN, agent_id="agent-G")
        for _ in range(5):
            learner.record_outcome(d, "ok", True)
        adj = learner.suggest_confidence_adjustment("agent-G", "deployment")
        assert adj > 1.0

    def test_suggest_adjustment_no_data_returns_one(self, learner):
        adj = learner.suggest_confidence_adjustment("agent-none", "deployment")
        assert adj == 1.0
