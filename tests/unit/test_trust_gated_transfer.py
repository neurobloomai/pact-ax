"""
Unit tests for trust-gated state transfer.

Covers:
  - TrustGateResult structure and repr
  - Heuristic fallback when no TrustManager is wired
  - Receiver-side TrustManager check (not just sender's stamped score)
  - Trust gate rejects when authoritative score below floor
  - Trust gate passes with degraded chain (warn, don't reject)
  - Trust gate rejects when chain is BROKEN
  - TrustChainManager failure is non-fatal (degraded to warn-only)
  - trust_gate field on IntegrationResult (success + failure paths)
  - Sender-side: _compute_trust_score uses TrustManager when wired
  - summary() reports trust_manager_wired / trust_chain_wired
  - Existing handoff lifecycle still works without any trust primitives
"""

import pytest
from pact_ax.state import (
    StateTransferManager,
    HandoffReason,
    IntegrationResult,
    TrustGateResult,
    TransferStatus,
)


# ── Minimal TrustManager stub ─────────────────────────────────────────────────

class StubTrustManager:
    """Minimal TrustManager double — maps (caller, target) → score."""

    def __init__(self, scores: dict):
        self._scores = scores

    def get_trust(self, target_id: str) -> float:
        return self._scores.get(target_id, 0.5)


class ErrorTrustManager:
    def get_trust(self, target_id: str) -> float:
        raise RuntimeError("DB connection lost")


# ── Minimal TrustChainManager stub ───────────────────────────────────────────

class _Score:
    def __init__(self, chain_trust, state_str):
        self.chain_trust = chain_trust

        class _State:
            def __init__(self, v):
                self.value = v
        self.state = _State(state_str)


class StubChainManager:
    def __init__(self, chain_trust: float, state: str):
        self._chain_trust = chain_trust
        self._state       = state

    def score(self, agents):
        return _Score(self._chain_trust, self._state)


class ErrorChainManager:
    def score(self, agents):
        raise RuntimeError("chain lookup failed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_packet(sender_id="agent-a", receiver_id="agent-b",
                 state_data=None, trust_score=0.8):
    """Build and send a packet from a plain sender."""
    sender = StateTransferManager(agent_id=sender_id)
    pid    = sender.prepare(
        to_agent_id  = receiver_id,
        state_data   = state_data or {"task": "analyse Q3"},
        reason       = HandoffReason.CONTINUATION,
    )
    packet        = sender.send(pid)
    # Override the heuristic-stamped score so tests can set it explicitly
    packet.trust_score = trust_score
    return packet


# ── TrustGateResult class ─────────────────────────────────────────────────────

class TestTrustGateResult:
    def test_passed_repr(self):
        r = TrustGateResult(sender_trust=0.8, sender_trust_source="trust_manager", passed=True)
        assert "PASS" in repr(r)
        assert "trust_manager" in repr(r)

    def test_failed_repr(self):
        r = TrustGateResult(
            sender_trust=0.2, sender_trust_source="heuristic",
            passed=False, rejection_reason="below floor"
        )
        assert "FAIL" in repr(r)
        assert "below floor" in repr(r)

    def test_to_dict(self):
        r = TrustGateResult(
            sender_trust=0.75,
            sender_trust_source="trust_manager",
            passed=True,
            chain_verified=True,
            chain_trust=0.70,
            chain_state="active",
        )
        d = r.to_dict()
        assert d["sender_trust"]        == 0.75
        assert d["sender_trust_source"] == "trust_manager"
        assert d["passed"]              is True
        assert d["chain_verified"]      is True
        assert d["chain_trust"]         == 0.70
        assert d["chain_state"]         == "active"

    def test_to_dict_chain_not_verified(self):
        r = TrustGateResult(sender_trust=0.5, sender_trust_source="heuristic", passed=True)
        d = r.to_dict()
        assert d["chain_verified"] is False
        assert d["chain_trust"]    is None
        assert d["chain_state"]    is None


# ── Heuristic fallback (no TrustManager) ─────────────────────────────────────

class TestHeuristicFallback:
    def test_receive_without_trust_manager_uses_packet_score(self):
        """When no TrustManager is wired, fall back to packet's own score."""
        packet   = _make_packet(trust_score=0.8)
        receiver = StateTransferManager(agent_id="agent-b", trust_floor=0.3)
        result   = receiver.receive(packet)

        assert result.success
        assert result.trust_gate is not None
        assert result.trust_gate.passed
        assert result.trust_gate.sender_trust_source == "heuristic"
        assert result.trust_gate.sender_trust        == pytest.approx(0.8)

    def test_receive_without_trust_manager_rejects_low_packet_score(self):
        packet   = _make_packet(trust_score=0.1)
        receiver = StateTransferManager(agent_id="agent-b", trust_floor=0.3)
        result   = receiver.receive(packet)

        assert not result.success
        # Can come from either validate() or trust gate
        assert result.error


# ── Receiver-side TrustManager gate ──────────────────────────────────────────

class TestTrustManagerGate:
    def test_gate_uses_receivers_trust_manager(self):
        """Receiver checks its own TrustManager, not the sender's stamped score."""
        tm = StubTrustManager({"agent-a": 0.85})
        # Packet has low stamped score — but receiver's TM says 0.85
        packet   = _make_packet(trust_score=0.2)
        receiver = StateTransferManager(
            agent_id="agent-b", trust_manager=tm, trust_floor=0.3
        )
        result = receiver.receive(packet)

        assert result.success, result.error
        assert result.trust_gate.sender_trust        == pytest.approx(0.85)
        assert result.trust_gate.sender_trust_source == "trust_manager"

    def test_gate_rejects_when_tm_score_below_floor(self):
        """Low TM score overrides a high stamped score."""
        tm = StubTrustManager({"agent-a": 0.1})
        # Packet says 0.9 — but receiver's TM says 0.1
        packet   = _make_packet(trust_score=0.9)
        receiver = StateTransferManager(
            agent_id="agent-b", trust_manager=tm, trust_floor=0.3
        )
        result = receiver.receive(packet)

        assert not result.success
        assert result.trust_gate is not None
        assert not result.trust_gate.passed
        assert result.trust_gate.sender_trust        == pytest.approx(0.1)
        assert result.trust_gate.sender_trust_source == "trust_manager"
        assert "0.10" in result.trust_gate.rejection_reason

    def test_gate_falls_back_to_heuristic_on_tm_error(self):
        """If TrustManager raises, use packet score as fallback (graceful degradation)."""
        tm       = ErrorTrustManager()
        packet   = _make_packet(trust_score=0.8)
        receiver = StateTransferManager(
            agent_id="agent-b", trust_manager=tm, trust_floor=0.3
        )
        result = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.sender_trust_source == "heuristic"

    def test_tm_score_in_integration_result_provenance(self):
        tm       = StubTrustManager({"agent-a": 0.9})
        packet   = _make_packet()
        receiver = StateTransferManager(agent_id="agent-b", trust_manager=tm)
        result   = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.sender_trust == pytest.approx(0.9)


# ── Sender-side TrustManager (compute_trust_score) ───────────────────────────

class TestSenderSideTrustManager:
    def test_sender_uses_tm_for_outbound_packet_score(self):
        """Sender's TrustManager score is stamped on the packet trust_score field."""
        tm     = StubTrustManager({"agent-b": 0.77})
        sender = StateTransferManager(agent_id="agent-a", trust_manager=tm)
        pid    = sender.prepare("agent-b", state_data={"task": "x"})
        packet = sender.send(pid)

        assert packet.trust_score == pytest.approx(0.77)

    def test_sender_falls_back_to_heuristic_when_tm_missing(self):
        sender = StateTransferManager(agent_id="agent-a")
        pid    = sender.prepare("agent-b", state_data={"task": "x"})
        packet = sender.send(pid)

        # Heuristic baseline is 0.7 for CONTINUATION with no epistemic payload
        assert 0.6 <= packet.trust_score <= 0.8

    def test_sender_falls_back_to_heuristic_on_tm_error(self):
        tm     = ErrorTrustManager()
        sender = StateTransferManager(agent_id="agent-a", trust_manager=tm)
        pid    = sender.prepare("agent-b", state_data={"task": "x"})
        packet = sender.send(pid)

        # Should not raise — heuristic baseline used
        assert 0.0 <= packet.trust_score <= 1.0


# ── TrustChainManager gate ────────────────────────────────────────────────────

class TestTrustChainGate:
    def test_active_chain_passes_and_is_recorded(self):
        tm    = StubTrustManager({"agent-a": 0.85})
        chain = StubChainManager(chain_trust=0.82, state="active")

        packet   = _make_packet()
        receiver = StateTransferManager(
            agent_id="agent-b",
            trust_manager=tm,
            trust_chain_manager=chain,
        )
        result = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.chain_verified
        assert result.trust_gate.chain_state  == "active"
        assert result.trust_gate.chain_trust  == pytest.approx(0.82)
        assert result.trust_gate.passed

    def test_broken_chain_rejects_packet(self):
        tm    = StubTrustManager({"agent-a": 0.85})
        chain = StubChainManager(chain_trust=0.2, state="broken")

        packet   = _make_packet()
        receiver = StateTransferManager(
            agent_id="agent-b",
            trust_manager=tm,
            trust_chain_manager=chain,
        )
        result = receiver.receive(packet)

        assert not result.success
        assert result.trust_gate is not None
        assert not result.trust_gate.passed
        assert "BROKEN" in result.trust_gate.rejection_reason
        assert result.trust_gate.chain_state == "broken"

    def test_degraded_chain_warns_but_does_not_reject(self):
        tm    = StubTrustManager({"agent-a": 0.85})
        chain = StubChainManager(chain_trust=0.55, state="degraded")

        packet   = _make_packet()
        receiver = StateTransferManager(
            agent_id="agent-b",
            trust_manager=tm,
            trust_chain_manager=chain,
        )
        result = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.chain_state == "degraded"
        assert any("DEGRADED" in w for w in result.warnings)

    def test_chain_manager_error_is_non_fatal(self):
        """TrustChainManager failure falls through — gate still passes on TM score alone."""
        tm    = StubTrustManager({"agent-a": 0.85})
        chain = ErrorChainManager()

        packet   = _make_packet()
        receiver = StateTransferManager(
            agent_id="agent-b",
            trust_manager=tm,
            trust_chain_manager=chain,
        )
        result = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.chain_verified is False  # error → not verified

    def test_chain_manager_without_trust_manager(self):
        """Chain check works even with heuristic trust fallback."""
        chain  = StubChainManager(chain_trust=0.80, state="active")
        packet = _make_packet(trust_score=0.8)

        receiver = StateTransferManager(
            agent_id="agent-b",
            trust_chain_manager=chain,
            trust_floor=0.3,
        )
        result = receiver.receive(packet)

        assert result.success
        assert result.trust_gate.chain_verified
        assert result.trust_gate.chain_state == "active"


# ── Trust gate field on IntegrationResult ─────────────────────────────────────

class TestIntegrationResultTrustGate:
    def test_trust_gate_present_on_success(self):
        packet   = _make_packet(trust_score=0.8)
        receiver = StateTransferManager(agent_id="agent-b")
        result   = receiver.receive(packet)

        assert result.trust_gate is not None
        assert isinstance(result.trust_gate, TrustGateResult)

    def test_trust_gate_present_on_gate_failure(self):
        tm       = StubTrustManager({"agent-a": 0.05})
        packet   = _make_packet(trust_score=0.9)
        receiver = StateTransferManager(agent_id="agent-b", trust_manager=tm, trust_floor=0.3)
        result   = receiver.receive(packet)

        assert not result.success
        assert result.trust_gate is not None
        assert not result.trust_gate.passed

    def test_trust_gate_none_on_structural_validation_failure(self):
        """Packets rejected by validate() before the gate have no trust_gate."""
        packet   = _make_packet(receiver_id="agent-c")  # addressed to wrong agent
        receiver = StateTransferManager(agent_id="agent-b")
        result   = receiver.receive(packet)

        assert not result.success
        assert result.trust_gate is None  # gate never ran


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_no_trust_primitives(self):
        mgr = StateTransferManager(agent_id="agent-x")
        s   = mgr.summary()
        assert s["trust_manager_wired"] is False
        assert s["trust_chain_wired"]   is False

    def test_summary_with_trust_manager(self):
        tm  = StubTrustManager({})
        mgr = StateTransferManager(agent_id="agent-x", trust_manager=tm)
        s   = mgr.summary()
        assert s["trust_manager_wired"] is True
        assert s["trust_chain_wired"]   is False

    def test_summary_with_both_primitives(self):
        tm    = StubTrustManager({})
        chain = StubChainManager(0.8, "active")
        mgr   = StateTransferManager(
            agent_id="agent-x", trust_manager=tm, trust_chain_manager=chain
        )
        s = mgr.summary()
        assert s["trust_manager_wired"] is True
        assert s["trust_chain_wired"]   is True


# ── Existing lifecycle still works without any trust primitives ───────────────

class TestBackwardCompat:
    def test_basic_prepare_send_receive_unchanged(self):
        sender   = StateTransferManager(agent_id="agent-a")
        receiver = StateTransferManager(agent_id="agent-b")

        pid    = sender.prepare("agent-b", state_data={"task": "x", "progress": 0.5})
        packet = sender.send(pid)
        result = receiver.receive(packet)

        assert result.success
        assert result.integrated_state["state_data"]["task"] == "x"
        assert result.trust_gate is not None  # always populated now

    def test_rollback_still_works(self):
        sender   = StateTransferManager(agent_id="agent-a")
        receiver = StateTransferManager(agent_id="agent-b")

        pid    = sender.prepare("agent-b", state_data={"task": "y"})
        packet = sender.send(pid)
        result = receiver.receive(packet)

        assert result.success
        rolled = receiver.rollback(result.packet_id)
        assert rolled is True
