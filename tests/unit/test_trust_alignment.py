"""
Unit tests for TrustAlignmentCheck and TrustContext primitives.

Covers:
  - n/n strict gate: 3 dimensions, all pass → gate_passed True
  - n/n strict gate: 3 dimensions, 1 fails → gate_passed False
  - Threshold gate: 3/4 aligned at 0.75 ratio → gate_passed True
  - Nested dimensions: parent with 2 children, evaluate tree
  - TrustContext propagation: orchestrator issues, sub-agent receives
  - TrustContext scope check: action within scope → permitted
  - TrustContext scope check: action exceeds scope → blocked
  - Dimension break signal: sub-agent signals, context flags
  - Auto re-gate trigger: valid_until exceeded → re-gate triggered
  - StateTransferManager trust gate: transfer blocked without valid context
"""

import pytest
from datetime import datetime, timezone, timedelta

from pact_ax.primitives.trust_alignment import (
    TrustAlignmentCheck,
    TrustDimension,
    DimensionScore,
    AlignmentResult,
    GateMode,
)
from pact_ax.primitives.trust_context import (
    TrustContext,
    TrustScope,
    TrustOperatingMode,
    DimensionBreak,
    Action,
    ActionLevel,
)
from pact_ax.state import StateTransferManager, HandoffReason


# ── Evaluator factories ───────────────────────────────────────────────────────

def _const(value: float):
    """Return a constant-score evaluator."""
    return lambda _agent_id: value


def _lookup(scores: dict, default: float = 0.5):
    """Return a lookup-based evaluator keyed on agent_id."""
    return lambda agent_id: scores.get(agent_id, default)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def three_passing_dims():
    return [
        TrustDimension("behavioral_consistency", "Stability",        _const(0.85), threshold=0.70),
        TrustDimension("capability_coherence",   "Declared vs real", _const(0.80), threshold=0.70),
        TrustDimension("relational_history",     "Interaction ledger",_const(0.75), threshold=0.70),
    ]


@pytest.fixture
def three_dims_one_failing():
    return [
        TrustDimension("behavioral_consistency", "Stability",        _const(0.85), threshold=0.70),
        TrustDimension("capability_coherence",   "Declared vs real", _const(0.40), threshold=0.70),  # fails
        TrustDimension("relational_history",     "Interaction ledger",_const(0.75), threshold=0.70),
    ]


@pytest.fixture
def four_dims_three_passing():
    """3/4 pass → alignment_ratio = 0.75."""
    return [
        TrustDimension("d1", "dim1", _const(0.9), threshold=0.7),
        TrustDimension("d2", "dim2", _const(0.9), threshold=0.7),
        TrustDimension("d3", "dim3", _const(0.9), threshold=0.7),
        TrustDimension("d4", "dim4", _const(0.3), threshold=0.7),  # fails
    ]


@pytest.fixture
def simple_check(three_passing_dims):
    return TrustAlignmentCheck(dimensions=three_passing_dims)


# ── TrustAlignmentCheck — strict gate ─────────────────────────────────────────

class TestStrictGate:
    def test_all_pass_strict(self, three_passing_dims):
        """3/3 passing → gate_passed True (STRICT)."""
        check  = TrustAlignmentCheck(dimensions=three_passing_dims)
        result = check.evaluate("agent-x")

        assert result.gate_passed is True
        assert result.aligned == 3
        assert result.total   == 3
        assert result.alignment_ratio == pytest.approx(1.0)

    def test_one_failing_strict(self, three_dims_one_failing):
        """3 dimensions, 1 fails → gate_passed False (STRICT)."""
        check  = TrustAlignmentCheck(dimensions=three_dims_one_failing)
        result = check.evaluate("agent-x")

        assert result.gate_passed is False
        assert result.aligned == 2
        assert result.total   == 3

    def test_strict_requires_all(self, four_dims_three_passing):
        """4 dimensions, 3 pass → still False under STRICT."""
        check  = TrustAlignmentCheck(dimensions=four_dims_three_passing, gate_mode=GateMode.STRICT)
        result = check.evaluate("agent-x")

        assert result.gate_passed is False

    def test_result_contains_breakdown(self, three_passing_dims):
        check  = TrustAlignmentCheck(dimensions=three_passing_dims)
        result = check.evaluate("agent-x")

        assert set(result.breakdown.keys()) == {
            "behavioral_consistency",
            "capability_coherence",
            "relational_history",
        }
        for ds in result.breakdown.values():
            assert isinstance(ds, DimensionScore)
            assert ds.passed is True

    def test_weakest_dimension_identified(self):
        dims = [
            TrustDimension("strong", "s", _const(0.99), threshold=0.5),
            TrustDimension("weak",   "w", _const(0.51), threshold=0.5),
        ]
        result = TrustAlignmentCheck(dimensions=dims).evaluate("agent-x")
        assert result.weakest_dimension.name == "weak"

    def test_result_timestamp_is_utc(self, simple_check):
        result = simple_check.evaluate("agent-x")
        assert result.timestamp.tzinfo is not None

    def test_to_dict_structure(self, simple_check):
        d = simple_check.evaluate("agent-x").to_dict()
        assert "aligned"           in d
        assert "gate_passed"       in d
        assert "alignment_ratio"   in d
        assert "weakest_dimension" in d
        assert "breakdown"         in d


# ── TrustAlignmentCheck — threshold gate ─────────────────────────────────────

class TestThresholdGate:
    def test_three_of_four_at_075_passes(self, four_dims_three_passing):
        """3/4 aligned → 0.75 ratio; THRESHOLD at 0.75 → gate_passed True."""
        check  = TrustAlignmentCheck(
            dimensions       = four_dims_three_passing,
            gate_mode        = GateMode.THRESHOLD,
            threshold_ratio  = 0.75,
        )
        result = check.evaluate("agent-x")

        assert result.gate_passed is True
        assert result.alignment_ratio == pytest.approx(0.75)

    def test_below_threshold_fails(self, four_dims_three_passing):
        check = TrustAlignmentCheck(
            dimensions      = four_dims_three_passing,
            gate_mode       = GateMode.THRESHOLD,
            threshold_ratio = 0.80,   # needs 80%, only 75% aligned
        )
        result = check.evaluate("agent-x")
        assert result.gate_passed is False

    def test_weighted_gate(self):
        dims = [
            TrustDimension("d1", "high weight pass",  _const(0.9), threshold=0.5, weight=2.0),
            TrustDimension("d2", "low weight fail",   _const(0.1), threshold=0.5, weight=0.5),
        ]
        check  = TrustAlignmentCheck(dimensions=dims, gate_mode=GateMode.WEIGHTED, threshold_ratio=0.6)
        result = check.evaluate("agent-x")
        # weighted_score = (0.9*2 + 0.1*0.5) / 2.5 = 1.85/2.5 = 0.74 → passes 0.6
        assert result.gate_passed is True


# ── Nested dimensions ─────────────────────────────────────────────────────────

class TestNestedDimensions:
    def test_parent_with_two_children_evaluated_as_tree(self):
        """
        Parent dimension contains 2 children.  Engine traverses the tree;
        parent's evaluator is never called — only leaf children score.
        """
        parent_evaluator_called = []

        def parent_eval(agent_id):
            parent_evaluator_called.append(agent_id)
            return 0.0  # should never be called

        child1 = TrustDimension("child_a", "Child A", _const(0.8), threshold=0.7)
        child2 = TrustDimension("child_b", "Child B", _const(0.9), threshold=0.7)
        parent = TrustDimension(
            "parent_group", "Group",
            evaluator  = parent_eval,
            threshold  = 0.7,
            children   = [child1, child2],
        )

        result = TrustAlignmentCheck(dimensions=[parent]).evaluate("agent-x")

        assert parent_evaluator_called == [], "Parent evaluator must not be called when children exist"
        assert result.total   == 2
        assert result.aligned == 2
        assert result.gate_passed is True
        assert "child_a" in result.breakdown
        assert "child_b" in result.breakdown
        assert "parent_group" not in result.breakdown

    def test_nested_one_child_fails(self):
        child1 = TrustDimension("c1", "C1", _const(0.9), threshold=0.7)
        child2 = TrustDimension("c2", "C2", _const(0.3), threshold=0.7)
        parent = TrustDimension("p", "Parent", _const(0.9), threshold=0.7, children=[child1, child2])

        result = TrustAlignmentCheck(dimensions=[parent]).evaluate("agent-x")

        assert result.gate_passed is False   # STRICT: c2 fails


# ── TrustContext — propagation ────────────────────────────────────────────────

class TestTrustContextPropagation:
    def _make_context(self, ratio=1.0, valid_for=3600):
        score = ratio  # all passing dims at this level
        dims  = [TrustDimension("d", "D", _const(score), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("orch-1")
        return TrustContext.establish(
            established_by   = "orch-1",
            alignment_result = result,
            alignment_check  = check,
            valid_for_seconds = valid_for,
        )

    def test_establish_creates_establishment_mode(self):
        ctx = self._make_context()
        assert ctx.operating_mode     == TrustOperatingMode.ESTABLISHMENT
        assert ctx.established_by     == "orch-1"
        assert ctx.propagation_depth  == 0

    def test_propagate_increments_depth(self):
        ctx   = self._make_context()
        child = ctx.propagate("agent-x")
        assert child.propagation_depth == 1
        assert child.operating_mode    == TrustOperatingMode.CONTINUITY
        assert child.established_by    == "orch-1"

    def test_propagate_twice_depth_two(self):
        ctx    = self._make_context()
        child  = ctx.propagate("agent-x")
        grandchild = child.propagate("agent-y")
        assert grandchild.propagation_depth == 2

    def test_receive_context_sets_depth(self):
        ctx   = self._make_context()
        sub   = TrustContext.__new__(TrustContext)
        sub.break_signals   = []
        sub._regate_required = False
        sub.receive_context(ctx)
        assert sub.propagation_depth == 1
        assert sub.operating_mode    == TrustOperatingMode.CONTINUITY

    def test_max_depth_exceeded_flags_regate(self):
        ctx   = self._make_context()
        child = TrustContext.establish(
            established_by    = "orch-1",
            alignment_result  = ctx.alignment_at_entry,
            alignment_check   = ctx._alignment_check,
            max_propagation_depth = 1,
        )
        grandchild = child.propagate("agent-x").propagate("agent-y")
        assert grandchild.regate_required is True


# ── TrustContext — scope checks ───────────────────────────────────────────────

class TestTrustContextScope:
    def _full_context(self):
        """Returns a context with ratio=1.0 (all dims pass → CRITICAL permitted)."""
        dims  = [TrustDimension("d", "D", _const(1.0), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-x")
        return TrustContext.establish(
            established_by           = "orch-1",
            alignment_result         = result,
            alignment_check          = check,
            critical_action_threshold = ActionLevel.CRITICAL,
        )

    def _low_context(self):
        """Returns a context with alignment_ratio=0.0 → only LOW permitted."""
        dims  = [TrustDimension("d", "D", _const(0.0), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims, gate_mode=GateMode.THRESHOLD, threshold_ratio=0.0)
        result = check.evaluate("agent-x")
        # Manually set alignment_ratio for scope derivation
        result.alignment_ratio = 0.0
        return TrustContext.establish(
            established_by           = "orch-1",
            alignment_result         = result,
            alignment_check          = check,
            critical_action_threshold = ActionLevel.CRITICAL,
        )

    def test_low_action_within_scope_permitted(self):
        ctx    = self._full_context()
        action = Action("read_state", ActionLevel.LOW)
        assert ctx.check_action_permitted(action) is True

    def test_medium_action_within_scope_permitted(self):
        ctx    = self._full_context()
        action = Action("write_state", ActionLevel.MEDIUM)
        assert ctx.check_action_permitted(action) is True

    def test_high_action_within_scope_permitted(self):
        ctx    = self._full_context()
        action = Action("external_call", ActionLevel.HIGH)
        assert ctx.check_action_permitted(action) is True

    def test_critical_action_triggers_regate_and_blocks(self):
        """CRITICAL action at CRITICAL threshold → auto re-gate, returns False."""
        ctx    = self._full_context()
        action = Action("drop_database", ActionLevel.CRITICAL)
        result = ctx.check_action_permitted(action)
        assert result is False
        assert ctx.regate_required is True

    def test_high_action_blocked_when_scope_is_low_only(self):
        ctx    = self._low_context()
        action = Action("write", ActionLevel.HIGH)
        assert ctx.check_action_permitted(action) is False


# ── Dimension break signal ────────────────────────────────────────────────────

class TestDimensionBreakSignal:
    def _context(self):
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("a")
        return TrustContext.establish("orch", result, check)

    def test_signal_break_appends_to_break_signals(self):
        ctx = self._context()
        ctx.signal_break("behavioral_consistency", "anomalous output", signaled_by="sub-1")

        assert len(ctx.break_signals) == 1
        brk = ctx.break_signals[0]
        assert isinstance(brk, DimensionBreak)
        assert brk.dimension   == "behavioral_consistency"
        assert brk.reason      == "anomalous output"
        assert brk.signaled_by == "sub-1"

    def test_signal_break_sets_regate_required(self):
        ctx = self._context()
        assert ctx.regate_required is False
        ctx.signal_break("d", "unexpected value")
        assert ctx.regate_required is True

    def test_action_blocked_after_break(self):
        ctx = self._context()
        ctx.signal_break("d", "something wrong")
        action = Action("read", ActionLevel.LOW)
        assert ctx.check_action_permitted(action) is False

    def test_multiple_breaks_accumulated(self):
        ctx = self._context()
        ctx.signal_break("d1", "reason A")
        ctx.signal_break("d2", "reason B")
        assert len(ctx.break_signals) == 2


# ── Auto re-gate trigger: expiry ──────────────────────────────────────────────

class TestAutoRegate:
    def test_expired_context_blocks_action(self):
        """valid_until in the past → check_action_permitted returns False."""
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-x")
        ctx = TrustContext.establish(
            established_by    = "orch",
            alignment_result  = result,
            alignment_check   = check,
            valid_for_seconds = -1,   # already expired
        )

        action = Action("read", ActionLevel.LOW)
        assert ctx.check_action_permitted(action) is False

    def test_expired_context_sets_regate_required(self):
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-x")
        ctx = TrustContext.establish(
            established_by    = "orch",
            alignment_result  = result,
            alignment_check   = check,
            valid_for_seconds = -1,
        )
        assert ctx.regate_required is True

    def test_request_regate_returns_alignment_check(self):
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-x")
        ctx = TrustContext.establish("orch", result, check)

        returned_check = ctx.request_regate()
        assert returned_check is check
        assert ctx.operating_mode == TrustOperatingMode.REGATE

    def test_regate_check_can_re_evaluate(self):
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-x")
        ctx = TrustContext.establish("orch", result, check)

        regate_check = ctx.request_regate()
        new_result   = regate_check.evaluate("agent-x")
        assert new_result.gate_passed is True


# ── StateTransferManager — TrustContext gate ──────────────────────────────────

class TestStateTransferTrustContext:
    def _packet(self, sender="agent-a", receiver="agent-b"):
        sender_mgr = StateTransferManager(agent_id=sender)
        pid = sender_mgr.prepare(receiver, state_data={"task": "test"})
        return sender_mgr.send(pid)

    def _valid_context(self):
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("agent-b")
        return TrustContext.establish(
            established_by           = "orch",
            alignment_result         = result,
            alignment_check          = check,
            critical_action_threshold = ActionLevel.HIGH,  # MEDIUM receive_handoff passes
        )

    def test_transfer_succeeds_with_valid_context(self):
        ctx      = self._valid_context()
        packet   = self._packet()
        receiver = StateTransferManager(
            agent_id      = "agent-b",
            trust_context = ctx,
            trust_floor   = 0.0,
        )
        result = receiver.receive(packet)
        assert result.success, result.error

    def test_transfer_blocked_when_context_expired(self):
        """An expired TrustContext blocks state transfer."""
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        res   = check.evaluate("agent-b")
        expired_ctx = TrustContext.establish(
            "orch", res, check, valid_for_seconds=-1   # already expired
        )
        packet   = self._packet()
        receiver = StateTransferManager(
            agent_id      = "agent-b",
            trust_context = expired_ctx,
            trust_floor   = 0.0,
        )
        result = receiver.receive(packet)
        assert result.success is False
        assert result.trust_gate is not None
        assert result.trust_gate.passed is False

    def test_transfer_blocked_when_break_signaled(self):
        """A dimension break on the context blocks transfer."""
        ctx = self._valid_context()
        ctx.signal_break("d", "detected anomaly", signaled_by="sub-agent")

        packet   = self._packet()
        receiver = StateTransferManager(
            agent_id      = "agent-b",
            trust_context = ctx,
            trust_floor   = 0.0,
        )
        result = receiver.receive(packet)
        assert result.success is False

    def test_summary_reports_context_wired(self):
        ctx = self._valid_context()
        mgr = StateTransferManager(agent_id="agent-b", trust_context=ctx)
        assert mgr.summary()["trust_context_wired"] is True

    def test_summary_reports_context_not_wired(self):
        mgr = StateTransferManager(agent_id="agent-b")
        assert mgr.summary()["trust_context_wired"] is False
