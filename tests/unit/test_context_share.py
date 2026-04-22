"""
tests/unit/test_context_share.py
─────────────────────────────────
Comprehensive tests for ContextShareManager.

Run with:  pytest tests/unit/test_context_share.py -v
"""

import pytest
from datetime import datetime, timezone

from pact_ax.primitives.context_share.manager import ContextShareManager
from pact_ax.primitives.context_share.schemas import (
    ContextPacket,
    ContextType,
    TrustLevel,
    Priority,
    CollaborationOutcome,
    AgentIdentity,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    return ContextShareManager(
        agent_id="agent-001",
        agent_type="support_specialist",
        capabilities=["nlp", "customer_support"],
        specializations=["billing"],
    )


@pytest.fixture
def bare_manager():
    return ContextShareManager(agent_id="agent-bare")


@pytest.fixture
def basic_payload():
    return {"current_task": "billing_support", "priority": "high"}


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestInit:

    def test_agent_id_stored(self, manager):
        assert manager.agent_id == "agent-001"

    def test_identity_is_agent_identity(self, manager):
        assert isinstance(manager.identity, AgentIdentity)

    def test_identity_reflects_init_args(self, manager):
        assert manager.identity.agent_type == "support_specialist"
        assert "nlp" in manager.identity.capabilities

    def test_empty_profiles_on_start(self, manager):
        assert manager.trust_profiles == {}

    def test_empty_capability_sensors_on_start(self, manager):
        assert manager.capability_sensors == {}

    def test_defaults_when_no_optional_args(self, bare_manager):
        assert bare_manager.identity.agent_type == "generic"
        assert bare_manager.identity.capabilities == []


# ──────────────────────────────────────────────────────────────────────────────
# create_context_packet()
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateContextPacket:

    def test_returns_context_packet(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert isinstance(pkt, ContextPacket)

    def test_correct_sender(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.from_agent.agent_id == "agent-001"

    def test_correct_recipient(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.to_agent == "agent-002"

    def test_payload_preserved(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.payload == basic_payload

    def test_context_type_stored(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.context_type == ContextType.TASK_KNOWLEDGE

    def test_priority_defaults_to_normal(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.priority == Priority.NORMAL

    def test_priority_high_respected(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
            priority=Priority.HIGH,
        )
        assert pkt.priority == Priority.HIGH

    def test_ttl_sets_expires_at(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
            ttl_seconds=300,
        )
        assert pkt.expires_at is not None
        assert pkt.expires_at > datetime.now(timezone.utc)

    def test_no_ttl_leaves_expires_at_none(self, manager, basic_payload):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload=basic_payload,
        )
        assert pkt.expires_at is None


# ──────────────────────────────────────────────────────────────────────────────
# Trust heuristics (_assess_required_trust)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrustHeuristics:

    def test_emotional_state_requires_strong_trust(self, manager):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.EMOTIONAL_STATE,
            payload={"mood": "anxious"},
        )
        assert pkt.trust_required == TrustLevel.STRONG

    def test_handoff_request_requires_building_trust(self, manager):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.HANDOFF_REQUEST,
            payload={"task": "billing"},
        )
        assert pkt.trust_required == TrustLevel.BUILDING

    def test_sensitive_payload_requires_deep_trust(self, manager):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload={"secret": "api_key_123"},
        )
        assert pkt.trust_required == TrustLevel.DEEP

    def test_normal_payload_requires_emerging_trust(self, manager):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            payload={"task": "revenue_analysis"},
        )
        assert pkt.trust_required == TrustLevel.EMERGING


# ──────────────────────────────────────────────────────────────────────────────
# assess_trust()
# ──────────────────────────────────────────────────────────────────────────────

class TestAssessTrust:

    def test_returns_dict_with_expected_keys(self, manager):
        result = manager.assess_trust(
            target_agent="agent-002",
            context_type=ContextType.TASK_KNOWLEDGE,
            current_situation={},
        )
        for key in ("agent_id", "context_type", "base_trust", "situation_adjustment",
                    "final_trust", "recommendation"):
            assert key in result

    def test_agent_id_in_result(self, manager):
        result = manager.assess_trust("agent-002", ContextType.TASK_KNOWLEDGE, {})
        assert result["agent_id"] == "agent-002"

    def test_context_type_as_string(self, manager):
        result = manager.assess_trust("agent-002", ContextType.TASK_KNOWLEDGE, {})
        assert result["context_type"] == ContextType.TASK_KNOWLEDGE.value

    def test_final_trust_in_valid_range(self, manager):
        result = manager.assess_trust("agent-002", ContextType.TASK_KNOWLEDGE, {})
        assert 0.0 <= result["final_trust"] <= 1.0

    def test_high_stakes_reduces_trust(self, manager):
        low_result = manager.assess_trust("agent-002", ContextType.TASK_KNOWLEDGE, {})
        high_result = manager.assess_trust(
            "agent-002", ContextType.TASK_KNOWLEDGE, {"stakes": "high"}
        )
        assert high_result["final_trust"] < low_result["final_trust"]

    def test_low_stakes_increases_trust(self, manager):
        base = manager.assess_trust("agent-002", ContextType.TASK_KNOWLEDGE, {})
        low_stakes = manager.assess_trust(
            "agent-002", ContextType.TASK_KNOWLEDGE, {"stakes": "low"}
        )
        assert low_stakes["final_trust"] > base["final_trust"]

    def test_recommendation_share_when_trust_high(self, manager):
        for _ in range(10):
            manager.record_collaboration_outcome(
                "agent-trusted", ContextType.TASK_KNOWLEDGE, "positive"
            )
        result = manager.assess_trust("agent-trusted", ContextType.TASK_KNOWLEDGE, {})
        assert result["recommendation"] == "share"

    def test_recommendation_caution_for_new_agent(self, manager):
        result = manager.assess_trust("brand-new-agent", ContextType.TASK_KNOWLEDGE, {})
        # New agent starts at neutral 0.5 which is <= 0.6
        assert result["recommendation"] == "caution"


# ──────────────────────────────────────────────────────────────────────────────
# sense_capability_limit()
# ──────────────────────────────────────────────────────────────────────────────

class TestSenseCapabilityLimit:

    def test_returns_dict_with_expected_keys(self, manager):
        result = manager.sense_capability_limit("billing_resolution")
        for key in ("task", "current_confidence", "threshold", "approaching_limit",
                    "limit_proximity", "recommendation"):
            assert key in result

    def test_unknown_task_defaults_to_full_confidence(self, manager):
        result = manager.sense_capability_limit("unknown_task")
        assert result["current_confidence"] == 1.0
        assert result["approaching_limit"] is False

    def test_low_confidence_approaching_limit(self, manager):
        manager.update_capability_confidence("hard_task", 0.3)
        result = manager.sense_capability_limit("hard_task", confidence_threshold=0.7)
        assert result["approaching_limit"] is True

    def test_high_confidence_not_approaching(self, manager):
        manager.update_capability_confidence("easy_task", 0.95)
        result = manager.sense_capability_limit("easy_task", confidence_threshold=0.7)
        assert result["approaching_limit"] is False

    def test_recommendation_continue(self, manager):
        manager.update_capability_confidence("strong_task", 0.95)
        result = manager.sense_capability_limit("strong_task", confidence_threshold=0.7)
        assert result["recommendation"] == "continue"

    def test_recommendation_monitor(self, manager):
        manager.update_capability_confidence("mid_task", 0.75)
        result = manager.sense_capability_limit("mid_task", confidence_threshold=0.7)
        assert result["recommendation"] == "monitor"

    def test_recommendation_prepare_handoff(self, manager):
        manager.update_capability_confidence("weak_task", 0.60)
        result = manager.sense_capability_limit("weak_task", confidence_threshold=0.7)
        assert result["recommendation"] == "prepare_handoff"

    def test_recommendation_immediate_handoff(self, manager):
        manager.update_capability_confidence("failing_task", 0.2)
        result = manager.sense_capability_limit("failing_task", confidence_threshold=0.7)
        assert result["recommendation"] == "immediate_handoff"

    def test_limit_proximity_zero_when_not_approaching(self, manager):
        manager.update_capability_confidence("strong_task2", 1.0)
        result = manager.sense_capability_limit("strong_task2", confidence_threshold=0.7)
        assert result["limit_proximity"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# update_capability_confidence()
# ──────────────────────────────────────────────────────────────────────────────

class TestUpdateCapabilityConfidence:

    def test_creates_sensor_for_new_task(self, manager):
        manager.update_capability_confidence("new_task", 0.8)
        assert "new_task" in manager.capability_sensors

    def test_confidence_stored(self, manager):
        manager.update_capability_confidence("my_task", 0.65)
        assert manager.capability_sensors["my_task"].current_confidence == 0.65

    def test_confidence_updated_on_repeat_call(self, manager):
        manager.update_capability_confidence("task_x", 0.9)
        manager.update_capability_confidence("task_x", 0.4)
        assert manager.capability_sensors["task_x"].current_confidence == 0.4

    def test_reflected_in_sense_capability_limit(self, manager):
        manager.update_capability_confidence("tracked_task", 0.5)
        result = manager.sense_capability_limit("tracked_task", confidence_threshold=0.7)
        assert result["current_confidence"] == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# prepare_handoff()
# ──────────────────────────────────────────────────────────────────────────────

class TestPrepareHandoff:

    def test_returns_context_packet(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution")
        assert isinstance(pkt, ContextPacket)

    def test_context_type_is_handoff_request(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution")
        assert pkt.context_type == ContextType.HANDOFF_REQUEST

    def test_priority_is_high(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution")
        assert pkt.priority == Priority.HIGH

    def test_target_agent_correct(self, manager):
        pkt = manager.prepare_handoff("agent-999", "billing_resolution")
        assert pkt.to_agent == "agent-999"

    def test_payload_contains_current_task(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution")
        assert pkt.payload["current_task"] == "billing_resolution"

    def test_payload_includes_emotional_context_by_default(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution")
        assert "emotional_context" in pkt.payload

    def test_payload_excludes_emotional_context_when_disabled(self, manager):
        pkt = manager.prepare_handoff(
            "agent-002", "billing_resolution", preserve_emotional_context=False
        )
        assert "emotional_context" not in pkt.payload

    def test_transfer_ownership_in_payload(self, manager):
        pkt = manager.prepare_handoff("agent-002", "billing_resolution", transfer_ownership=True)
        assert pkt.payload["transfer_ownership"] is True


# ──────────────────────────────────────────────────────────────────────────────
# record_collaboration_outcome()
# ──────────────────────────────────────────────────────────────────────────────

class TestRecordCollaborationOutcome:

    def test_creates_trust_profile_for_new_agent(self, manager):
        manager.record_collaboration_outcome(
            "agent-new", ContextType.TASK_KNOWLEDGE, "positive"
        )
        assert "agent-new" in manager.trust_profiles

    def test_positive_outcome_raises_trust(self, manager):
        before = manager.assess_trust("agent-collab", ContextType.TASK_KNOWLEDGE, {})["base_trust"]
        for _ in range(5):
            manager.record_collaboration_outcome(
                "agent-collab", ContextType.TASK_KNOWLEDGE, "positive"
            )
        after = manager.assess_trust("agent-collab", ContextType.TASK_KNOWLEDGE, {})["base_trust"]
        assert after > before

    def test_negative_outcome_lowers_trust(self, manager):
        for _ in range(5):
            manager.record_collaboration_outcome(
                "agent-bad", ContextType.TASK_KNOWLEDGE, "positive"
            )
        trust_mid = manager.assess_trust("agent-bad", ContextType.TASK_KNOWLEDGE, {})["base_trust"]
        for _ in range(5):
            manager.record_collaboration_outcome(
                "agent-bad", ContextType.TASK_KNOWLEDGE, "negative"
            )
        trust_after = manager.assess_trust("agent-bad", ContextType.TASK_KNOWLEDGE, {})["base_trust"]
        assert trust_after < trust_mid

    def test_invalid_outcome_does_not_raise(self, manager):
        manager.record_collaboration_outcome(
            "agent-002", ContextType.TASK_KNOWLEDGE, "completely_invalid_value"
        )

    def test_records_pattern_for_positive(self, manager):
        manager.record_collaboration_outcome(
            "agent-p", ContextType.TASK_KNOWLEDGE, "positive"
        )
        key = f"agent-p_{ContextType.TASK_KNOWLEDGE.value}"
        assert manager.collaboration_patterns[key]["successes"] == 1

    def test_records_pattern_for_negative(self, manager):
        manager.record_collaboration_outcome(
            "agent-n", ContextType.TASK_KNOWLEDGE, "negative"
        )
        key = f"agent-n_{ContextType.TASK_KNOWLEDGE.value}"
        assert manager.collaboration_patterns[key]["failures"] == 1

    def test_pattern_counts_accumulate(self, manager):
        for _ in range(3):
            manager.record_collaboration_outcome(
                "agent-acc", ContextType.TASK_KNOWLEDGE, "positive"
            )
        key = f"agent-acc_{ContextType.TASK_KNOWLEDGE.value}"
        assert manager.collaboration_patterns[key]["successes"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# get_collaboration_insights()
# ──────────────────────────────────────────────────────────────────────────────

class TestGetCollaborationInsights:

    def test_returns_dict_with_required_keys(self, manager):
        insights = manager.get_collaboration_insights()
        for key in ("trust_summary", "capability_status", "collaboration_patterns"):
            assert key in insights

    def test_empty_trust_summary_when_no_interactions(self, manager):
        insights = manager.get_collaboration_insights()
        assert insights["trust_summary"] == {}

    def test_capability_status_reflects_sensors(self, manager):
        manager.update_capability_confidence("task_a", 0.8)
        insights = manager.get_collaboration_insights()
        assert "task_a" in insights["capability_status"]
        assert insights["capability_status"]["task_a"] == 0.8

    def test_trust_summary_populated_after_outcomes(self, manager):
        manager.record_collaboration_outcome(
            "agent-x", ContextType.TASK_KNOWLEDGE, "positive"
        )
        insights = manager.get_collaboration_insights()
        assert "agent-x" in insights["trust_summary"]

    def test_trust_summary_contains_interaction_count(self, manager):
        for _ in range(3):
            manager.record_collaboration_outcome(
                "agent-y", ContextType.TASK_KNOWLEDGE, "positive"
            )
        insights = manager.get_collaboration_insights()
        assert insights["trust_summary"]["agent-y"]["interaction_count"] == 3

    def test_trust_summary_overall_trust_in_range(self, manager):
        manager.record_collaboration_outcome(
            "agent-z", ContextType.TASK_KNOWLEDGE, "positive"
        )
        insights = manager.get_collaboration_insights()
        ot = insights["trust_summary"]["agent-z"]["overall_trust"]
        assert 0.0 <= ot <= 1.0

    def test_collaboration_patterns_in_insights(self, manager):
        manager.record_collaboration_outcome(
            "agent-pat", ContextType.TASK_KNOWLEDGE, "positive"
        )
        insights = manager.get_collaboration_insights()
        key = f"agent-pat_{ContextType.TASK_KNOWLEDGE.value}"
        assert key in insights["collaboration_patterns"]


# ──────────────────────────────────────────────────────────────────────────────
# Multiple context types
# ──────────────────────────────────────────────────────────────────────────────

class TestMultipleContextTypes:

    @pytest.mark.parametrize("ctx_type", [
        ContextType.TASK_KNOWLEDGE,
        ContextType.EMOTIONAL_STATE,
        ContextType.HANDOFF_REQUEST,
        ContextType.TRUST_SIGNAL,
    ])
    def test_packet_created_for_all_context_types(self, manager, ctx_type):
        pkt = manager.create_context_packet(
            target_agent="agent-002",
            context_type=ctx_type,
            payload={"data": "test"},
        )
        assert pkt.context_type == ctx_type

    @pytest.mark.parametrize("ctx_type", [
        ContextType.TASK_KNOWLEDGE,
        ContextType.EMOTIONAL_STATE,
        ContextType.HANDOFF_REQUEST,
        ContextType.TRUST_SIGNAL,
    ])
    def test_assess_trust_works_for_all_context_types(self, manager, ctx_type):
        result = manager.assess_trust("agent-002", ctx_type, {})
        assert 0.0 <= result["final_trust"] <= 1.0
