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
  - TrustIntent: intent payload survives propagation
  - TrustIntent: load-bearing constraint violation detected at hop
  - TrustIntent: non-load-bearing constraint may be dropped
  - TrustIntent: no intent in child violates all load-bearing constraints
  - TrustIntent: MASI-class failure pattern (constraint lived in delegator's head)
  - OrientationDrift: mode_of_engagement drift fails n/n strict gate
  - OrientationDrift: stable mode never triggers spurious failures
  - OrientationDrift: full lifecycle — drift detected → signal_break → regate
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
    TrustIntent,
    TrustConstraint,
    ScopeDelta,
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


# ── TrustIntent ────────────────────────────────────────────────────────────────

def _intent_fixture() -> TrustIntent:
    """Shared intent used across TestTrustIntent cases."""
    return TrustIntent(
        purpose="scan UNIVERSE for MA proximity setups",
        constraints=[
            TrustConstraint(
                key="delisted_tickers_invalid",
                description="Delisted tickers must not appear in scan results",
                load_bearing=True,
            ),
            TrustConstraint(
                key="band_width_3pct",
                description="MA proximity band is -3% to +3%",
                load_bearing=False,
            ),
        ],
    )


def _context_with_intent(intent: TrustIntent) -> TrustContext:
    """Issue a TrustContext carrying the given intent."""
    dims = [
        TrustDimension("behavioral_consistency", "Stability", _const(0.85), threshold=0.70),
        TrustDimension("capability_coherence",   "Coherence", _const(0.80), threshold=0.70),
        TrustDimension("relational_history",     "History",   _const(0.75), threshold=0.70),
    ]
    check  = TrustAlignmentCheck(dimensions=dims)
    result = check.evaluate("orchestrator-1")
    return TrustContext.establish(
        established_by    = "orchestrator-1",
        alignment_result  = result,
        alignment_check   = check,
        intent            = intent,
    )


class TestTrustIntent:
    def test_intent_survives_propagation(self):
        """Intent on parent appears unchanged in child after propagate()."""
        intent  = _intent_fixture()
        parent  = _context_with_intent(intent)
        child   = parent.propagate(to_agent="scanner-agent")

        assert child.intent is not None
        assert child.intent.purpose == intent.purpose
        assert len(child.intent.constraints) == 2

    def test_intent_survives_receive_context(self):
        """Intent is copied when a sub-agent receives via receive_context()."""
        intent   = _intent_fixture()
        parent   = _context_with_intent(intent)
        receiver = _context_with_intent(TrustIntent(purpose="placeholder"))
        receiver.receive_context(parent)

        assert receiver.intent is not None
        assert receiver.intent.purpose == intent.purpose

    def test_load_bearing_constraint_violation_detected(self):
        """
        Child intent is missing a load-bearing constraint →
        verify_intent_integrity() returns its key.
        """
        parent_intent = _intent_fixture()
        parent        = _context_with_intent(parent_intent)
        child         = parent.propagate(to_agent="scanner-agent")

        # Simulate a hop that dropped the load-bearing constraint
        child.intent = TrustIntent(
            purpose="scan UNIVERSE",
            constraints=[
                TrustConstraint(
                    key="band_width_3pct",
                    description="MA proximity band is -3% to +3%",
                    load_bearing=False,
                ),
            ],
        )

        violations = child.verify_intent_integrity(parent_intent)
        assert "delisted_tickers_invalid" in violations

    def test_non_load_bearing_constraint_can_be_dropped(self):
        """
        Child intent drops a non-load-bearing constraint →
        verify_intent_integrity() reports no violations.
        """
        parent_intent = _intent_fixture()
        parent        = _context_with_intent(parent_intent)
        child         = parent.propagate(to_agent="scanner-agent")

        # Sub-agent narrowed scope: dropped the non-load-bearing band constraint
        child.intent = TrustIntent(
            purpose="scan UNIVERSE for MA proximity setups",
            constraints=[
                TrustConstraint(
                    key="delisted_tickers_invalid",
                    description="Delisted tickers must not appear in scan results",
                    load_bearing=True,
                ),
            ],
        )

        violations = child.verify_intent_integrity(parent_intent)
        assert violations == []

    def test_no_intent_in_child_violates_all_load_bearing(self):
        """
        Child carries no intent at all → every load-bearing constraint is
        reported as violated.
        """
        parent_intent = _intent_fixture()
        parent        = _context_with_intent(parent_intent)
        child         = parent.propagate(to_agent="scanner-agent")
        child.intent  = None

        violations = child.verify_intent_integrity(parent_intent)
        assert "delisted_tickers_invalid" in violations
        assert len(violations) == 1  # only one load-bearing constraint in fixture

    def test_intent_in_to_dict(self):
        """intent block appears in to_dict() output when set."""
        intent  = _intent_fixture()
        context = _context_with_intent(intent)
        d       = context.to_dict()

        assert d["intent"] is not None
        assert d["intent"]["purpose"] == intent.purpose
        assert len(d["intent"]["constraints"]) == 2

    def test_no_intent_to_dict_is_none(self):
        """intent is None in to_dict() when not provided."""
        dims = [
            TrustDimension("behavioral_consistency", "Stability", _const(0.85), threshold=0.70),
        ]
        check   = TrustAlignmentCheck(dimensions=dims)
        result  = check.evaluate("orchestrator-1")
        context = TrustContext.establish(
            established_by   = "orchestrator-1",
            alignment_result = result,
            alignment_check  = check,
        )
        assert context.to_dict()["intent"] is None

    def test_masi_class_failure_pattern(self):
        """
        MASI-class failure: a constraint lived only in the delegator's head,
        never in anything propagatable — the downstream agent had no mechanism
        to know it existed.

        This test proves the fix: the constraint is declared in the intent,
        propagated to the child, and its absence is detectable.
        """
        # Orchestrator knows: delisted tickers must not appear in results
        parent_intent = TrustIntent(
            purpose="run MA scanner on UNIVERSE",
            constraints=[
                TrustConstraint(
                    key="delisted_tickers_invalid",
                    description="Delisted tickers must not appear in scan results",
                    load_bearing=True,
                ),
            ],
        )
        parent = _context_with_intent(parent_intent)

        # Sub-agent receives context — intent arrives intact
        child = parent.propagate(to_agent="scanner-agent")
        assert child.verify_intent_integrity(parent_intent) == []

        # Sub-agent (or a further hop) silently drops the constraint
        child.intent = TrustIntent(
            purpose="run MA scanner on UNIVERSE",
            constraints=[],   # constraint dropped — constraint lived only in prose
        )

        # Now detectable — this is what was invisible before
        violations = child.verify_intent_integrity(parent_intent)
        assert violations == ["delisted_tickers_invalid"]

        # Sub-agent should signal a break when it detects the violation
        child.signal_break(
            "intent_integrity",
            f"load-bearing constraints dropped: {violations}",
            signaled_by="scanner-agent",
        )
        assert child.regate_required is True


# ── OrientationDrift ───────────────────────────────────────────────────────────

class TestOrientationDrift:
    """
    3b: mode_of_engagement as a declared dimension in TrustAlignmentCheck.

    Orientation (advisory vs. executive, read-only vs. write, etc.) is one
    of the n dimensions the caller declares. When the agent's observed mode
    drifts from the declared mode at establishment time, that single dimension
    scores 0.0. Under STRICT n/n, one failure fails the whole gate.

    The evaluator is a closure — it captures the declared mode and reads the
    agent's observable state. No new types needed; TrustDimension's evaluator
    interface is the extension point.
    """

    def test_mode_drift_fails_strict_gate(self):
        """
        n=3 check: behavioral + capability + mode_of_engagement.
        Declared mode is "advisory". When agent shifts to "executive",
        mode_of_engagement scores 0.0 → gate_passed False, and
        weakest_dimension is mode_of_engagement.
        """
        declared_mode = "advisory"
        agent_state   = {"mode": "advisory"}   # mutable: simulates observable state

        def mode_evaluator(agent_id: str) -> float:
            return 1.0 if agent_state["mode"] == declared_mode else 0.0

        dims = [
            TrustDimension("behavioral_consistency", "Stability",        _const(0.85), threshold=0.70),
            TrustDimension("capability_coherence",   "Coherence",        _const(0.80), threshold=0.70),
            TrustDimension("mode_of_engagement",     "Advisory vs exec", mode_evaluator, threshold=0.90),
        ]
        check = TrustAlignmentCheck(dimensions=dims)

        # Initial gate: mode is advisory — all 3 pass
        result = check.evaluate("agent-x")
        assert result.gate_passed is True
        assert result.aligned == 3

        # Mode drifts while task is in flight
        agent_state["mode"] = "executive"

        # Re-gate (or continuity monitor re-runs the same check):
        result_after = check.evaluate("agent-x")
        assert result_after.gate_passed is False
        assert result_after.aligned == 2                                          # behavioral + capability pass
        assert result_after.breakdown["mode_of_engagement"].passed is False
        assert result_after.weakest_dimension.name == "mode_of_engagement"

    def test_mode_alignment_survives_same_mode(self):
        """
        Mode never drifts → mode_of_engagement continues to pass on every
        re-evaluation. No spurious failures.
        """
        declared_mode = "advisory"
        agent_state   = {"mode": "advisory"}

        def mode_evaluator(agent_id: str) -> float:
            return 1.0 if agent_state["mode"] == declared_mode else 0.0

        dims = [
            TrustDimension("behavioral_consistency", "Stability",        _const(0.85), threshold=0.70),
            TrustDimension("mode_of_engagement",     "Advisory vs exec", mode_evaluator, threshold=0.90),
        ]
        check = TrustAlignmentCheck(dimensions=dims)

        r1 = check.evaluate("agent-x")
        r2 = check.evaluate("agent-x")
        assert r1.gate_passed is True
        assert r2.gate_passed is True

    def test_mode_drift_triggers_regate_via_signal_break(self):
        """
        Full lifecycle: gate established, sub-agent monitors mode,
        drift detected on re-evaluation → signal_break() → regate_required.

        The alignment check is not a live monitor — it re-evaluates on demand.
        TrustContext.signal_break() is the continuity hook that links detected
        drift to the re-gate lifecycle.
        """
        declared_mode = "advisory"
        agent_state   = {"mode": "advisory"}

        def mode_evaluator(agent_id: str) -> float:
            return 1.0 if agent_state["mode"] == declared_mode else 0.0

        dims = [
            TrustDimension("behavioral_consistency", "Stability",        _const(0.85), threshold=0.70),
            TrustDimension("capability_coherence",   "Coherence",        _const(0.80), threshold=0.70),
            TrustDimension("mode_of_engagement",     "Advisory vs exec", mode_evaluator, threshold=0.90),
        ]
        check  = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("orchestrator")
        assert result.gate_passed is True

        ctx   = TrustContext.establish("orchestrator", result, check)
        child = ctx.propagate(to_agent="agent-x")
        assert child.regate_required is False

        # Mode drifts mid-task — sub-agent re-runs check and detects it
        agent_state["mode"] = "executive"
        continuity_result   = check.evaluate("agent-x")
        if not continuity_result.breakdown["mode_of_engagement"].passed:
            child.signal_break(
                "mode_of_engagement",
                f"mode drifted from '{declared_mode}' to '{agent_state['mode']}'",
                signaled_by="agent-x",
            )

        assert child.regate_required is True
        assert len(child.break_signals) == 1
        assert child.break_signals[0].dimension == "mode_of_engagement"


# ── Intent preservation: origin_hash + ScopeDelta ─────────────────────────────

def _origin_intent_fixture() -> TrustIntent:
    return TrustIntent(
        purpose="validate market scanner output",
        constraints=[
            TrustConstraint(
                key="delisted_tickers_invalid",
                description="Delisted tickers must not appear in scan results",
                load_bearing=True,
            ),
            TrustConstraint(
                key="band_width_3pct",
                description="MA proximity band is -3% to +3%",
                load_bearing=False,
            ),
        ],
    )


def _ctx_with_origin(intent: TrustIntent) -> TrustContext:
    dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
    check = TrustAlignmentCheck(dimensions=dims)
    result = check.evaluate("orch")
    return TrustContext.establish(
        established_by   = "orch",
        alignment_result = result,
        alignment_check  = check,
        intent           = intent,
    )


class TestOriginHash:
    def test_origin_hash_set_at_establish(self):
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        assert ctx.origin_hash is not None
        assert ctx.origin_hash == intent.content_hash()

    def test_origin_hash_survives_3_hops_byte_identical(self):
        """Doctrine: origin_hash is unchanged across 3+ propagation hops."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        h0     = ctx.origin_hash

        hop1 = ctx.propagate("agent-1")
        hop2 = hop1.propagate("agent-2")
        hop3 = hop2.propagate("agent-3")

        assert hop1.origin_hash == h0
        assert hop2.origin_hash == h0
        assert hop3.origin_hash == h0

    def test_origin_intent_survives_3_hops_byte_identical(self):
        """origin_intent object identity carries through — no copying of fields."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)

        hop1 = ctx.propagate("agent-1")
        hop2 = hop1.propagate("agent-2")
        hop3 = hop2.propagate("agent-3")

        # Re-hash at each hop must equal the original — content unchanged
        assert hop1.origin_intent.content_hash() == intent.content_hash()
        assert hop2.origin_intent.content_hash() == intent.content_hash()
        assert hop3.origin_intent.content_hash() == intent.content_hash()

    def test_reconstructed_intent_fails_hash_verification(self):
        """
        Doctrine test: if a hop re-serialises intent (even with same meaning
        but different description wording), the hash fails.

        This is the mechanical detection that makes reconstruction impossible
        to hide — you cannot paraphrase your way through the gate.
        """
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        hop    = ctx.propagate("agent-1")

        # Simulate a hop that re-serialised/paraphrased the load_bearing constraint
        reconstructed = TrustIntent(
            purpose="validate market scanner output",
            constraints=[
                TrustConstraint(
                    key="delisted_tickers_invalid",
                    description="Do not include delisted tickers",  # paraphrase
                    load_bearing=True,
                ),
                TrustConstraint(
                    key="band_width_3pct",
                    description="MA proximity band is -3% to +3%",
                    load_bearing=False,
                ),
            ],
        )
        hop._origin_intent = reconstructed  # tamper with private field

        # Hash of reconstructed != original hash → fidelity violation
        assert hop.origin_intent.content_hash() != ctx.origin_hash

    def test_tampered_origin_hash_mismatch_detectable(self):
        """A hop that changes origin_hash directly is detectable via recompute."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        hop    = ctx.propagate("agent-1")

        original_hash = hop.origin_hash
        hop._origin_hash = "deadbeef" * 8  # tamper

        # The recomputed hash from origin_intent still matches the true hash
        assert hop.origin_intent.content_hash() == original_hash
        # But origin_hash field no longer matches — detectable
        assert hop.origin_hash != hop.origin_intent.content_hash()

    def test_no_origin_intent_gives_none_hash(self):
        """Backward compat: context without intent has None origin_hash."""
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("orch")
        ctx = TrustContext.establish(
            established_by   = "orch",
            alignment_result = result,
            alignment_check  = check,
        )
        assert ctx.origin_hash   is None
        assert ctx.origin_intent is None


class TestScopeDelta:
    def test_non_load_bearing_drop_accepted(self):
        """Dropping a non-load_bearing constraint via delta is legitimate narrowing."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)

        delta = ScopeDelta(
            hop_id              = "agent-1",
            narrowed_scope      = "sub-agent only needs liquid names",
            constraints_dropped = ["band_width_3pct"],  # non-load_bearing
        )
        ctx.add_scope_delta(delta)  # must not raise
        assert len(ctx._scope_deltas) == 1

    def test_load_bearing_drop_rejected_at_append(self):
        """Dropping a load_bearing constraint is rejected immediately at add_scope_delta()."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)

        delta = ScopeDelta(
            hop_id              = "agent-1",
            narrowed_scope      = "attempting to drop a load_bearing constraint",
            constraints_dropped = ["delisted_tickers_invalid"],  # load_bearing!
        )
        with pytest.raises(ValueError, match="load_bearing"):
            ctx.add_scope_delta(delta)

        # Delta must not have been appended
        assert len(ctx._scope_deltas) == 0

    def test_load_bearing_drop_detectable_at_gate(self):
        """
        Belt-and-suspenders: even if add_scope_delta is bypassed and a
        load_bearing key appears in _scope_deltas, effective_constraints()
        would exclude it — and the structural fidelity dimension catches it.

        We test effective_constraints() directly here (the belt).
        The structural_fidelity_dimension test covers the gate (the suspenders).
        """
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)

        # Force the delta in without going through the guard (simulates bypass)
        ctx._scope_deltas.append(
            ScopeDelta(
                hop_id              = "rogue-hop",
                narrowed_scope      = "dropped load_bearing",
                constraints_dropped = ["delisted_tickers_invalid"],
            )
        )

        effective_keys = {c.key for c in ctx.effective_constraints()}
        assert "delisted_tickers_invalid" not in effective_keys

    def test_effective_constraints_removes_dropped(self):
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)

        ctx.add_scope_delta(ScopeDelta(
            hop_id="agent-1",
            narrowed_scope="narrow band constraint",
            constraints_dropped=["band_width_3pct"],
        ))

        effective = ctx.effective_constraints()
        keys = [c.key for c in effective]
        assert "delisted_tickers_invalid" in keys
        assert "band_width_3pct"          not in keys

    def test_scope_deltas_in_to_dict(self):
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        ctx.add_scope_delta(ScopeDelta(
            hop_id="agent-1",
            narrowed_scope="narrowing",
            constraints_dropped=["band_width_3pct"],
        ))
        d = ctx.to_dict()
        assert len(d["scope_deltas"]) == 1
        assert d["scope_deltas"][0]["hop_id"] == "agent-1"

    def test_scope_deltas_propagate_to_child(self):
        """Accumulated deltas carry forward on propagate()."""
        intent = _origin_intent_fixture()
        ctx    = _ctx_with_origin(intent)
        ctx.add_scope_delta(ScopeDelta(
            hop_id="agent-1",
            narrowed_scope="first narrowing",
            constraints_dropped=[],
        ))
        child = ctx.propagate("agent-2")
        assert len(child._scope_deltas) == 1


class TestStructuralFidelityDimension:
    def _fidelity_ctx(self, intent):
        """Issue a fresh context with intent; return (ctx, dim)."""
        dims  = [TrustDimension("d", "D", _const(0.9), threshold=0.5)]
        check = TrustAlignmentCheck(dimensions=dims)
        result = check.evaluate("orch")
        ctx = TrustContext.establish(
            established_by   = "orch",
            alignment_result = result,
            alignment_check  = check,
            intent           = intent,
        )
        dim = TrustAlignmentCheck.structural_fidelity_dimension(
            origin_intent = intent,
            origin_hash   = ctx.origin_hash,
        )
        return ctx, dim

    def test_intact_origin_passes_gate(self):
        """Hash matches + all load_bearing present → fidelity dimension scores 1.0."""
        intent = _origin_intent_fixture()
        ctx, dim = self._fidelity_ctx(intent)
        hop = ctx.propagate("agent-1")

        evaluator = dim.make_bound_evaluator(hop)
        assert evaluator("agent-1") == pytest.approx(1.0)

    def test_tampered_hash_fails_gate(self):
        """origin_hash tampered → fidelity dimension scores 0.0."""
        intent = _origin_intent_fixture()
        ctx, dim = self._fidelity_ctx(intent)
        hop = ctx.propagate("agent-1")
        hop._origin_hash = "0" * 64  # tamper

        evaluator = dim.make_bound_evaluator(hop)
        assert evaluator("agent-1") == pytest.approx(0.0)

    def test_reconstructed_intent_fails_gate(self):
        """
        Doctrine test via the gate: a hop that re-serialised intent instead of
        referencing it is mechanically detectable — the hash fails.
        """
        intent = _origin_intent_fixture()
        ctx, dim = self._fidelity_ctx(intent)
        hop = ctx.propagate("agent-1")

        # Re-serialise with paraphrased description
        hop._origin_intent = TrustIntent(
            purpose="validate market scanner output",
            constraints=[
                TrustConstraint(
                    key="delisted_tickers_invalid",
                    description="No delisted tickers allowed",  # paraphrase
                    load_bearing=True,
                ),
            ],
        )

        evaluator = dim.make_bound_evaluator(hop)
        assert evaluator("agent-1") == pytest.approx(0.0)

    def test_missing_load_bearing_fails_gate(self):
        """Load_bearing constraint dropped from effective scope → fidelity 0.0."""
        intent = _origin_intent_fixture()
        ctx, dim = self._fidelity_ctx(intent)
        hop = ctx.propagate("agent-1")

        # Force-drop load_bearing key from effective scope
        hop._scope_deltas.append(ScopeDelta(
            hop_id="rogue",
            narrowed_scope="dropped load_bearing",
            constraints_dropped=["delisted_tickers_invalid"],
        ))

        evaluator = dim.make_bound_evaluator(hop)
        assert evaluator("agent-1") == pytest.approx(0.0)

    def test_legitimate_narrowing_passes_gate(self):
        """Non-load_bearing constraint dropped via delta → fidelity still 1.0."""
        intent = _origin_intent_fixture()
        ctx, dim = self._fidelity_ctx(intent)
        hop = ctx.propagate("agent-1")
        hop.add_scope_delta(ScopeDelta(
            hop_id="agent-1",
            narrowed_scope="sub-agent drops band constraint",
            constraints_dropped=["band_width_3pct"],
        ))

        evaluator = dim.make_bound_evaluator(hop)
        assert evaluator("agent-1") == pytest.approx(1.0)
