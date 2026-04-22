"""
tests/unit/test_epistemic_transfer.py
──────────────────────────────────────
Tests for EpistemicStateTransfer — both the transfer() and receive() paths.

Run with:  pytest tests/unit/test_epistemic_transfer.py -v
"""

import pytest
from datetime import datetime

from pact_ax.state.epistemic_transfer import EpistemicStateTransfer
from pact_ax.primitives.epistemic import (
    EpistemicState,
    ConfidenceLevel,
    KnowledgeBoundary,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def est():
    return EpistemicStateTransfer()


@pytest.fixture
def confident_state():
    return EpistemicState(
        value="Q3 revenue grew 12%",
        confidence=ConfidenceLevel.CONFIDENT,
        uncertainty_reason=None,
        source="analyst-agent-A",
    )


@pytest.fixture
def uncertain_state():
    return EpistemicState(
        value="APAC growth is highest",
        confidence=ConfidenceLevel.MODERATE,
        uncertainty_reason="Sample size limited to 3 months",
        source="analyst-agent-B",
    )


@pytest.fixture
def state_with_boundary():
    boundary = KnowledgeBoundary(
        domain="financial_analysis",
        proficiency=ConfidenceLevel.CONFIDENT,
        known_capabilities={"revenue_analysis", "cost_breakdown"},
        known_limits={"derivatives_pricing"},
    )
    return EpistemicState(
        value="revenue up 12%",
        confidence=ConfidenceLevel.CERTAIN,
        source="specialist-agent",
        boundary=boundary,
    )


# ──────────────────────────────────────────────────────────────────────────────
# transfer()
# ──────────────────────────────────────────────────────────────────────────────

class TestTransfer:

    def test_returns_dict(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert isinstance(pkg, dict)

    def test_package_keys_present(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        for key in ("knowledge", "confidence", "confidence_value",
                    "uncertainty_reason", "boundaries", "source", "metadata"):
            assert key in pkg

    def test_knowledge_value_preserved(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["knowledge"] == confident_state.value

    def test_confidence_name_preserved(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["confidence"] == "CONFIDENT"

    def test_confidence_value_in_range(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert 0.0 < pkg["confidence_value"] <= 1.0

    def test_source_preserved(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["source"] == "analyst-agent-A"

    def test_metadata_agent_ids(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["metadata"]["from_agent"] == "agent-A"
        assert pkg["metadata"]["to_agent"]   == "agent-B"

    def test_metadata_state_id_matches(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["metadata"]["state_id"] == confident_state.id

    def test_transfer_logged(self, est, confident_state):
        assert len(est.transfer_log) == 0
        est.transfer(confident_state, "agent-A", "agent-B")
        assert len(est.transfer_log) == 1

    def test_multiple_transfers_all_logged(self, est, confident_state, uncertain_state):
        est.transfer(confident_state, "agent-A", "agent-B")
        est.transfer(uncertain_state, "agent-B", "agent-C")
        assert len(est.transfer_log) == 2

    def test_uncertainty_reason_in_package(self, est, uncertain_state):
        pkg = est.transfer(uncertain_state, "agent-A", "agent-B")
        assert pkg["uncertainty_reason"] == uncertain_state.uncertainty_reason

    def test_context_passed_through(self, est, confident_state):
        pkg = est.transfer(
            confident_state, "agent-A", "agent-B",
            context={"priority": "high"},
        )
        assert pkg["metadata"]["context"]["priority"] == "high"

    def test_no_context_defaults_to_empty_dict(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["metadata"]["context"] == {}

    def test_boundary_serialised_when_present(self, est, state_with_boundary):
        pkg = est.transfer(state_with_boundary, "agent-A", "agent-B")
        assert pkg["boundaries"] is not None
        assert pkg["boundaries"]["domain"] == "financial_analysis"

    def test_boundary_none_when_absent(self, est, confident_state):
        pkg = est.transfer(confident_state, "agent-A", "agent-B")
        assert pkg["boundaries"] is None


# ──────────────────────────────────────────────────────────────────────────────
# receive()
# ──────────────────────────────────────────────────────────────────────────────

class TestReceive:

    def _roundtrip(self, est, state):
        pkg = est.transfer(state, "agent-A", "agent-B")
        return est.receive(pkg, "agent-B")

    def test_returns_epistemic_state(self, est, confident_state):
        received = self._roundtrip(est, confident_state)
        assert isinstance(received, EpistemicState)

    def test_value_preserved_through_roundtrip(self, est, confident_state):
        received = self._roundtrip(est, confident_state)
        assert received.value == confident_state.value

    def test_confidence_preserved_through_roundtrip(self, est, confident_state):
        received = self._roundtrip(est, confident_state)
        assert received.confidence == confident_state.confidence

    def test_uncertainty_reason_preserved(self, est, uncertain_state):
        received = self._roundtrip(est, uncertain_state)
        assert received.uncertainty_reason == uncertain_state.uncertainty_reason

    def test_source_includes_originating_agent(self, est, confident_state):
        received = self._roundtrip(est, confident_state)
        assert "agent-A" in received.source

    def test_timestamp_preserved(self, est, confident_state):
        received = self._roundtrip(est, confident_state)
        assert isinstance(received.timestamp, datetime)

    def test_all_confidence_levels_survive_roundtrip(self, est):
        for level in ConfidenceLevel:
            state = EpistemicState(
                value=f"fact at {level.name}",
                confidence=level,
                source="test",
            )
            received = self._roundtrip(est, state)
            assert received.confidence == level

    def test_receive_with_none_uncertainty_reason(self, est):
        state = EpistemicState(
            value="clean fact",
            confidence=ConfidenceLevel.CERTAIN,
            source="test",
        )
        received = self._roundtrip(est, state)
        assert received.uncertainty_reason is None


# ──────────────────────────────────────────────────────────────────────────────
# _serialize_boundary()
# ──────────────────────────────────────────────────────────────────────────────

class TestSerializeBoundary:

    def test_none_boundary_returns_none(self, est):
        assert est._serialize_boundary(None) is None

    def test_boundary_dict_has_expected_keys(self, est):
        b = KnowledgeBoundary(
            domain="testing",
            proficiency=ConfidenceLevel.MODERATE,
            known_capabilities={"a", "b"},
            known_limits={"c"},
        )
        serialised = est._serialize_boundary(b)
        for key in ("domain", "proficiency", "known_limits",
                    "known_capabilities", "last_updated"):
            assert key in serialised

    def test_capabilities_serialised_as_list(self, est):
        b = KnowledgeBoundary(
            domain="x", proficiency=ConfidenceLevel.MODERATE,
            known_capabilities={"cap1", "cap2"},
        )
        s = est._serialize_boundary(b)
        assert isinstance(s["known_capabilities"], list)
        assert set(s["known_capabilities"]) == {"cap1", "cap2"}

    def test_limits_serialised_as_list(self, est):
        b = KnowledgeBoundary(
            domain="x", proficiency=ConfidenceLevel.LOW,
            known_limits={"lim1"},
        )
        s = est._serialize_boundary(b)
        assert isinstance(s["known_limits"], list)
        assert "lim1" in s["known_limits"]

    def test_proficiency_as_name_string(self, est):
        b = KnowledgeBoundary(domain="x", proficiency=ConfidenceLevel.CERTAIN)
        s = est._serialize_boundary(b)
        assert s["proficiency"] == "CERTAIN"
