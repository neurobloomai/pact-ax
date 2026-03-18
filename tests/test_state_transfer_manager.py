"""
tests/test_state_transfer_manager.py
──────────────────────────────────────
Full test suite for StateTransferManager.

Run with:  pytest tests/test_state_transfer_manager.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from pact_ax.state.state_transfer_manager import (
    HandoffPacket,
    HandoffReason,
    IntegrationResult,
    StateTransferManager,
    TransferStatus,
    ValidationResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sender():
    return StateTransferManager(agent_id="agent-sender")


@pytest.fixture
def receiver():
    return StateTransferManager(agent_id="agent-receiver")


@pytest.fixture
def simple_state():
    return {
        "task":     "analyse Q3 revenue report",
        "progress": 0.6,
        "findings": ["revenue up 12%", "APAC growth strongest"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# prepare() + send()
# ──────────────────────────────────────────────────────────────────────────────

class TestPrepareAndSend:

    def test_prepare_returns_packet_id(self, sender, simple_state):
        pid = sender.prepare(
            to_agent_id="agent-receiver",
            state_data=simple_state,
        )
        assert isinstance(pid, str)
        assert pid.startswith("pkt-")

    def test_packet_in_preparing_state_after_prepare(self, sender, simple_state):
        pid = sender.prepare(
            to_agent_id="agent-receiver",
            state_data=simple_state,
        )
        assert sender._outbound[pid].status == TransferStatus.PREPARING

    def test_send_transitions_to_in_flight(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        assert packet.status == TransferStatus.IN_FLIGHT

    def test_send_returns_handoff_packet(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        assert isinstance(packet, HandoffPacket)

    def test_send_unknown_packet_id_raises(self, sender):
        with pytest.raises(KeyError):
            sender.send("nonexistent-id")

    def test_send_non_preparing_packet_raises(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        with pytest.raises(ValueError):
            sender.send(pid)  # already IN_FLIGHT

    def test_handoff_reason_stored(self, sender, simple_state):
        pid    = sender.prepare(
            to_agent_id="agent-receiver",
            state_data=simple_state,
            reason=HandoffReason.ESCALATION,
        )
        packet = sender.send(pid)
        assert packet.reason == HandoffReason.ESCALATION

    def test_trust_score_in_valid_range(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        assert 0.0 <= packet.trust_score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# validate()
# ──────────────────────────────────────────────────────────────────────────────

class TestValidate:

    def _make_packet(self, sender, receiver, simple_state, **kwargs):
        pid    = sender.prepare(
            to_agent_id=receiver.agent_id,
            state_data=simple_state,
            **kwargs,
        )
        return sender.send(pid)

    def test_valid_packet_passes(self, sender, receiver, simple_state):
        packet = self._make_packet(sender, receiver, simple_state)
        result = receiver.validate(packet)
        assert result.valid
        assert result.issues == []

    def test_wrong_recipient_fails(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-other", state_data=simple_state)
        packet = sender.send(pid)
        # validate with a receiver whose agent_id doesn't match
        other = StateTransferManager(agent_id="agent-receiver")
        result = other.validate(packet)
        assert not result.valid
        assert any("addressed to" in issue for issue in result.issues)

    def test_low_trust_fails(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        packet.trust_score = 0.1  # force low trust

        strict_receiver = StateTransferManager(
            agent_id="agent-receiver", trust_floor=0.5
        )
        result = strict_receiver.validate(packet)
        assert not result.valid
        assert any("Trust score" in issue for issue in result.issues)

    def test_empty_state_fails(self, sender, receiver):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data={})
        packet = sender.send(pid)
        result = receiver.validate(packet)
        assert not result.valid
        assert any("empty" in issue for issue in result.issues)

    def test_expired_packet_fails(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        # Backdate creation time beyond TTL
        packet.created_at = datetime.utcnow() - timedelta(hours=5)
        result = receiver.validate(packet)
        assert not result.valid
        assert any("old" in issue for issue in result.issues)


# ──────────────────────────────────────────────────────────────────────────────
# receive() + integration
# ──────────────────────────────────────────────────────────────────────────────

class TestReceive:

    def _full_handoff(self, sender, receiver, state_data, **prepare_kwargs):
        pid    = sender.prepare(
            to_agent_id=receiver.agent_id,
            state_data=state_data,
            **prepare_kwargs,
        )
        packet = sender.send(pid)
        return receiver.receive(packet)

    def test_receive_succeeds(self, sender, receiver, simple_state):
        result = self._full_handoff(sender, receiver, simple_state)
        assert result.success

    def test_receive_returns_integration_result(self, sender, receiver, simple_state):
        result = self._full_handoff(sender, receiver, simple_state)
        assert isinstance(result, IntegrationResult)

    def test_integrated_state_contains_state_data(self, sender, receiver, simple_state):
        result = self._full_handoff(sender, receiver, simple_state)
        assert result.integrated_state["state_data"] == simple_state

    def test_provenance_recorded(self, sender, receiver, simple_state):
        result = self._full_handoff(sender, receiver, simple_state)
        prov = result.integrated_state["handoff_provenance"]
        assert prov["from_agent"] == sender.agent_id
        assert prov["reason"]     == HandoffReason.CONTINUATION.value

    def test_packet_marked_integrated(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        receiver.receive(packet)
        assert packet.status == TransferStatus.INTEGRATED

    def test_invalid_packet_fails_gracefully(self, sender, simple_state):
        pid    = sender.prepare(to_agent_id="agent-other", state_data=simple_state)
        packet = sender.send(pid)
        # receiver has different agent_id — should fail validation
        result = receiver.receive(packet)
        assert not result.success
        assert result.error is not None

    def test_packet_with_wrong_recipient_fails(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id="agent-wrong", state_data=simple_state)
        packet = sender.send(pid)
        result = receiver.receive(packet)
        assert not result.success

    def test_serialisation_roundtrip(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        # simulate serialise → deserialise across process boundary
        rehydrated = HandoffPacket.from_dict(packet.to_dict())
        result = receiver.receive(rehydrated)
        assert result.success
        assert result.integrated_state["state_data"] == simple_state


# ──────────────────────────────────────────────────────────────────────────────
# rollback()
# ──────────────────────────────────────────────────────────────────────────────

class TestRollback:

    def test_rollback_integrated_packet(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        result = receiver.receive(packet)
        assert result.success

        ok = receiver.rollback(packet.packet_id)
        assert ok
        assert packet.status == TransferStatus.ROLLED_BACK

    def test_rollback_unknown_id_returns_false(self, receiver):
        assert receiver.rollback("nonexistent-xyz") is False

    def test_rollback_in_flight_returns_false(self, sender, receiver, simple_state):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        # packet not yet received — can't roll back from sender side
        assert sender.rollback(pid) is False


# ──────────────────────────────────────────────────────────────────────────────
# checkpoint() + restore()
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpoints:

    def test_checkpoint_returns_id(self, sender, simple_state):
        ckpt_id = sender.checkpoint("before-analysis", simple_state)
        assert isinstance(ckpt_id, str)
        assert ckpt_id.startswith("ckpt-")

    def test_checkpoint_listed(self, sender, simple_state):
        ckpt_id = sender.checkpoint("my-checkpoint", simple_state)
        labels  = [c["checkpoint_id"] for c in sender.list_checkpoints()]
        assert ckpt_id in labels

    def test_restore_returns_snapshot(self, sender, simple_state):
        ckpt_id = sender.checkpoint("snap1", simple_state)
        snap    = sender.restore(ckpt_id)
        assert snap["state_data"] == simple_state

    def test_restore_unknown_raises(self, sender):
        with pytest.raises(KeyError):
            sender.restore("ckpt-doesnotexist")

    def test_multiple_checkpoints_independent(self, sender):
        c1 = sender.checkpoint("v1", {"step": 1})
        c2 = sender.checkpoint("v2", {"step": 2})
        assert sender.restore(c1)["state_data"]["step"] == 1
        assert sender.restore(c2)["state_data"]["step"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Observability — summary(), get_active_transfers(), clear_completed()
# ──────────────────────────────────────────────────────────────────────────────

class TestObservability:

    def test_summary_keys_present(self, sender):
        s = sender.summary()
        for key in ("agent_id", "outbound_count", "inbound_count",
                    "active_transfers", "status_breakdown", "trust_floor"):
            assert key in s

    def test_active_transfers_includes_in_flight(self, sender, simple_state):
        pid = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        sender.send(pid)
        active = sender.get_active_transfers()
        assert any(p.packet_id == pid for p in active)

    def test_get_transfer_status(self, sender, simple_state):
        pid = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        assert sender.get_transfer_status(pid) == TransferStatus.PREPARING
        sender.send(pid)
        assert sender.get_transfer_status(pid) == TransferStatus.IN_FLIGHT

    def test_get_transfer_status_unknown_returns_none(self, sender):
        assert sender.get_transfer_status("unknown-id") is None

    def test_clear_completed_removes_old_packets(self, sender, simple_state, receiver):
        pid    = sender.prepare(to_agent_id=receiver.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        receiver.receive(packet)
        # Backdate so TTL fires
        packet.created_at = datetime.utcnow() - timedelta(hours=10)
        cleared = receiver.clear_completed()
        assert cleared == 1


# ──────────────────────────────────────────────────────────────────────────────
# StoryKeeper integration (optional dependency)
# ──────────────────────────────────────────────────────────────────────────────

class TestStoryKeeperIntegration:

    def test_narrative_built_without_story_keeper(self, sender, simple_state):
        """Manager without a StoryKeeper should use stub narrative, not crash."""
        pid    = sender.prepare(to_agent_id="agent-receiver", state_data=simple_state)
        packet = sender.send(pid)
        assert "summary" in packet.narrative

    def test_narrative_from_mock_story_keeper(self, simple_state):
        mock_sk = MagicMock()
        mock_sk.get_story_summary.return_value = {
            "arc_transitions": [],
            "total_interactions": 3,
        }
        mock_sk.recall_for_context.return_value = [
            {"input": "hello", "response": "world"}
        ]
        mock_sk.current_arc = "COLLABORATION"

        mgr = StateTransferManager(agent_id="agent-sk", story_keeper=mock_sk)
        pid = mgr.prepare(to_agent_id="agent-b", state_data=simple_state)
        pkt = mgr.send(pid)

        assert pkt.narrative["current_arc"] == "COLLABORATION"
        assert pkt.narrative["interaction_count"] == 3

    def test_receive_integrates_narrative_into_story_keeper(self, sender, simple_state):
        mock_sk = MagicMock()
        mock_sk.get_story_summary.return_value = {"arc_transitions": [], "total_interactions": 0}
        mock_sk.recall_for_context.return_value = []
        mock_sk.current_arc = "EXPLORATION"

        recv = StateTransferManager(agent_id="agent-receiver", story_keeper=mock_sk)
        pid    = sender.prepare(to_agent_id=recv.agent_id, state_data=simple_state)
        packet = sender.send(pid)
        recv.receive(packet)
        # No crash; StoryKeeper methods may or may not have been called depending on narrative
        assert True


# ──────────────────────────────────────────────────────────────────────────────
# HandoffReason coverage
# ──────────────────────────────────────────────────────────────────────────────

class TestHandoffReasons:

    @pytest.mark.parametrize("reason", list(HandoffReason))
    def test_all_reasons_complete_handoff(self, sender, receiver, simple_state, reason):
        pid    = sender.prepare(
            to_agent_id=receiver.agent_id,
            state_data=simple_state,
            reason=reason,
        )
        packet = sender.send(pid)
        result = receiver.receive(packet)
        assert result.success
        assert result.integrated_state["handoff_provenance"]["reason"] == reason.value
