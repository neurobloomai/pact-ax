"""
TrustContext: Scoped, propagatable trust contract for agent chains.

An orchestrator runs TrustAlignmentCheck (n/n gate), receives an
AlignmentResult, and issues a TrustContext.  Sub-agents receive the
context — they do NOT re-run TrustAlignmentCheck.  They operate within
it, monitor within it, and escalate when it is exceeded.

Key concepts
------------
TrustScope
    Derived from the AlignmentResult — not hardcoded.  Which dimensions
    aligned, and at what strength, determines which ActionLevels are
    permitted in this context.

TrustOperatingMode
    ESTABLISHMENT — heavyweight, full n/n, at task assignment
    CONTINUITY    — lightweight, dimension monitoring only
    REGATE        — event-triggered, full n/n re-run

DimensionBreak
    A signal raised by a sub-agent when a monitored dimension diverges
    from the alignment established at entry.

TrustContext
    The live contract.  Knows which operating mode it was created in.
    Triggers REGATE automatically on:
      - a dimension break signal
      - valid_until exceeded
      - a critical action threshold crossed
      - propagation_depth exceeding max_propagation_depth

Usage
-----
    # Orchestrator side
    from pact_ax.primitives.trust_alignment import TrustAlignmentCheck
    from pact_ax.primitives.trust_context   import TrustContext, Action, ActionLevel

    check   = TrustAlignmentCheck(dimensions=my_dims)
    result  = check.evaluate("agent-x")
    context = TrustContext.establish(
        established_by    = "orchestrator-1",
        alignment_result  = result,
        alignment_check   = check,
    )

    # Sub-agent side
    child = context.propagate(to_agent="agent-x")
    if child.check_action_permitted(Action("write_to_store", ActionLevel.HIGH)):
        ...
    else:
        child.signal_break("behavioral_consistency", "unexpected output pattern")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .trust_alignment import TrustAlignmentCheck, AlignmentResult

AgentID = str


# ── Intent payload ─────────────────────────────────────────────────────────────

@dataclass
class TrustConstraint:
    """
    A single constraint within a TrustIntent.

    load_bearing=True  — must survive every propagation hop verbatim.
                         Omission or modification in a child context is a
                         detectable violation at any downstream gate.
    load_bearing=False — may be narrowed or dropped by a sub-agent performing
                         legitimate re-scoping.

    Part of PACT-AX trust infrastructure.
    Prose is lossy. Contracts propagate.
    """

    key:          str   # machine-readable identifier
    description:  str   # human-readable statement of the constraint
    load_bearing: bool  # True = must survive every hop; False = may be narrowed


@dataclass
class TrustIntent:
    """
    Structured intent block carried by a TrustContext.

    Carries not just permission but purpose and load-bearing constraints,
    so intent travels with the delegation instead of being reconstructed
    from prose at each hop.  Fidelity, not just authorization.

    Usage
    -----
        intent = TrustIntent(
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
                    load_bearing=False,   # sub-agent may narrow this
                ),
            ],
        )

    Part of PACT-AX trust infrastructure.
    Prose is lossy. Contracts propagate.
    """

    purpose:     str
    constraints: List[TrustConstraint] = field(default_factory=list)

    def load_bearing_constraints(self) -> List[TrustConstraint]:
        """Return only constraints marked load_bearing=True."""
        return [c for c in self.constraints if c.load_bearing]

    def verify_preserved_in(self, child_intent: "TrustIntent") -> List[str]:
        """
        Check that every load-bearing constraint in self appears verbatim
        in child_intent.

        Returns a list of violated constraint keys — empty means intact.
        A violation occurs when a load-bearing constraint is absent from
        child_intent or its description has been modified.

        Part of PACT-AX trust infrastructure.
        Prose is lossy. Contracts propagate.
        """
        child_by_key = {c.key: c for c in child_intent.constraints}
        violations = []
        for c in self.load_bearing_constraints():
            if c.key not in child_by_key:
                violations.append(c.key)
            elif child_by_key[c.key].description != c.description:
                violations.append(c.key)
        return violations

    def to_dict(self) -> dict:
        """
        Serialisable snapshot.

        Part of PACT-AX trust infrastructure.
        Prose is lossy. Contracts propagate.
        """
        return {
            "purpose": self.purpose,
            "constraints": [
                {
                    "key":          c.key,
                    "description":  c.description,
                    "load_bearing": c.load_bearing,
                }
                for c in self.constraints
            ],
        }


# ── Enums ──────────────────────────────────────────────────────────────────────

class ActionLevel(str, Enum):
    """
    Severity tier for agent actions.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class TrustOperatingMode(str, Enum):
    """
    Which phase of the trust lifecycle a TrustContext was created in.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    ESTABLISHMENT = "establishment"  # full n/n gate at task assignment
    CONTINUITY    = "continuity"     # lightweight dimension monitoring
    REGATE        = "regate"         # full n/n re-run after break/expiry


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class Action:
    """
    An action a sub-agent wants to perform, carrying its severity level.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    name:     str
    level:    ActionLevel
    metadata: Dict = field(default_factory=dict)


@dataclass
class DimensionBreak:
    """
    A break signal raised by a sub-agent for a monitored trust dimension.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    dimension:   str
    reason:      str
    signaled_by: AgentID
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TrustScope:
    """
    The permitted action surface derived from an AlignmentResult.

    Not hardcoded — alignment ratio and dimension strengths determine
    which ActionLevels are permitted in this context.

    Mapping (ratio → permitted ActionLevels):
        1.0        → LOW, MEDIUM, HIGH, CRITICAL
        [0.75, 1)  → LOW, MEDIUM, HIGH
        [0.5, 0.75)→ LOW, MEDIUM
        < 0.5      → LOW only

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    _RATIO_THRESHOLDS = [
        (1.0,  [ActionLevel.LOW, ActionLevel.MEDIUM, ActionLevel.HIGH, ActionLevel.CRITICAL]),
        (0.75, [ActionLevel.LOW, ActionLevel.MEDIUM, ActionLevel.HIGH]),
        (0.5,  [ActionLevel.LOW, ActionLevel.MEDIUM]),
        (0.0,  [ActionLevel.LOW]),
    ]

    def __init__(
        self,
        permitted_levels:    List[ActionLevel],
        dimension_strengths: Dict[str, float],
    ) -> None:
        """
        Initialise a TrustScope.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        self.permitted_levels    = set(permitted_levels)
        self.dimension_strengths = dimension_strengths

    @classmethod
    def from_alignment_result(cls, result: "AlignmentResult") -> "TrustScope":
        """
        Derive a TrustScope from an AlignmentResult.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        ratio = result.alignment_ratio
        levels: List[ActionLevel] = [ActionLevel.LOW]
        for threshold, permitted in cls._RATIO_THRESHOLDS:
            if ratio >= threshold:
                levels = permitted
                break

        strengths = {
            name: ds.score
            for name, ds in result.breakdown.items()
        }
        return cls(permitted_levels=levels, dimension_strengths=strengths)

    def permits(self, action: Action) -> bool:
        """
        Return True if the action's level is within this scope.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        return action.level in self.permitted_levels

    def to_dict(self) -> dict:
        """
        Serialisable snapshot.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        return {
            "permitted_levels":    [lvl.value for lvl in self.permitted_levels],
            "dimension_strengths": {k: round(v, 4) for k, v in self.dimension_strengths.items()},
        }


# ── Core contract ──────────────────────────────────────────────────────────────

class TrustContext:
    """
    Scoped, propagatable trust contract that flows through agent chains.

    Created by an orchestrator after a successful TrustAlignmentCheck.
    Sub-agents receive it via propagate(), operate within its scope, signal
    breaks when they detect divergence, and request a re-gate when a break
    or expiry demands a fresh n/n evaluation.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.

    Parameters
    ----------
    context_id : str
        UUID identifying this specific context instance.
    established_by : str
        AgentID of the orchestrator that ran the n/n gate.
    alignment_at_entry : AlignmentResult
        The n/n result that established this context.
    scope : TrustScope
        Derived from alignment_at_entry — determines permitted ActionLevels.
    operating_mode : TrustOperatingMode
        Which phase this context was created in.
    valid_until : datetime
        Expiry timestamp.  check_action_permitted() returns False after expiry.
    critical_action_threshold : ActionLevel
        Actions at or above this level trigger an automatic re-gate request.
        Default CRITICAL.
    propagation_depth : int
        How many hops this context is from its origin (0 = orchestrator).
    max_propagation_depth : int or None
        Maximum allowed propagation depth.  None = unlimited.
    """

    def __init__(
        self,
        context_id:                  str,
        established_by:              AgentID,
        alignment_at_entry:          "AlignmentResult",
        scope:                       TrustScope,
        alignment_check:             "TrustAlignmentCheck",
        operating_mode:              TrustOperatingMode = TrustOperatingMode.ESTABLISHMENT,
        valid_until:                 Optional[datetime] = None,
        critical_action_threshold:   ActionLevel = ActionLevel.CRITICAL,
        propagation_depth:           int = 0,
        max_propagation_depth:       Optional[int] = None,
        intent:                      Optional[TrustIntent] = None,
    ) -> None:
        """
        Initialise a TrustContext.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        self.context_id                = context_id
        self.established_by            = established_by
        self.alignment_at_entry        = alignment_at_entry
        self.scope                     = scope
        self._alignment_check          = alignment_check
        self.operating_mode            = operating_mode
        self.valid_until               = valid_until or datetime.now(timezone.utc) + timedelta(hours=1)
        self.critical_action_threshold = critical_action_threshold
        self.propagation_depth         = propagation_depth
        self.max_propagation_depth     = max_propagation_depth
        self.intent:                   Optional[TrustIntent] = intent
        self.break_signals: List[DimensionBreak] = []
        self._regate_required: bool = False

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def establish(
        cls,
        established_by:            AgentID,
        alignment_result:          "AlignmentResult",
        alignment_check:           "TrustAlignmentCheck",
        valid_for_seconds:         int = 3600,
        critical_action_threshold: ActionLevel = ActionLevel.CRITICAL,
        max_propagation_depth:     Optional[int] = None,
        intent:                    Optional[TrustIntent] = None,
    ) -> "TrustContext":
        """
        Issue a new TrustContext after a successful n/n alignment gate.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Parameters
        ----------
        established_by : str
            AgentID of the orchestrator.
        alignment_result : AlignmentResult
            The passing n/n result.
        alignment_check : TrustAlignmentCheck
            The check object (needed for re-gate).
        valid_for_seconds : int
            TTL for this context.  Default 3600.
        critical_action_threshold : ActionLevel
            Auto re-gate trigger level.  Default CRITICAL.
        max_propagation_depth : int or None
            Maximum hops before depth-based re-gate.  None = unlimited.

        Returns
        -------
        TrustContext
        """
        scope = TrustScope.from_alignment_result(alignment_result)
        return cls(
            context_id                = str(uuid.uuid4()),
            established_by            = established_by,
            alignment_at_entry        = alignment_result,
            scope                     = scope,
            alignment_check           = alignment_check,
            operating_mode            = TrustOperatingMode.ESTABLISHMENT,
            valid_until               = datetime.now(timezone.utc) + timedelta(seconds=valid_for_seconds),
            critical_action_threshold = critical_action_threshold,
            propagation_depth         = 0,
            max_propagation_depth     = max_propagation_depth,
            intent                    = intent,
        )

    # ── Sub-agent API ──────────────────────────────────────────────────────────

    def propagate(self, to_agent: AgentID) -> "TrustContext":
        """
        Produce a child TrustContext for a sub-agent.

        Increments propagation_depth.  If max_propagation_depth is reached,
        the child context immediately marks itself as requiring a re-gate.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Parameters
        ----------
        to_agent : str
            AgentID receiving the context (informational).

        Returns
        -------
        TrustContext
            A new child context operating in CONTINUITY mode.
        """
        new_depth = self.propagation_depth + 1
        child = TrustContext(
            context_id                = str(uuid.uuid4()),
            established_by            = self.established_by,
            alignment_at_entry        = self.alignment_at_entry,
            scope                     = self.scope,
            alignment_check           = self._alignment_check,
            operating_mode            = TrustOperatingMode.CONTINUITY,
            valid_until               = self.valid_until,
            critical_action_threshold = self.critical_action_threshold,
            propagation_depth         = new_depth,
            max_propagation_depth     = self.max_propagation_depth,
            intent                    = self.intent,
        )
        if self.max_propagation_depth is not None and new_depth > self.max_propagation_depth:
            child._regate_required = True
        return child

    def receive_context(self, context: "TrustContext") -> None:
        """
        Accept a context propagated from an orchestrator or parent agent.

        Copies scope, alignment, and validity from *context* into self,
        then increments propagation depth and validates depth limits.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        self.established_by        = context.established_by
        self.alignment_at_entry    = context.alignment_at_entry
        self.scope                 = context.scope
        self._alignment_check      = context._alignment_check
        self.operating_mode        = TrustOperatingMode.CONTINUITY
        self.valid_until           = context.valid_until
        self.propagation_depth     = context.propagation_depth + 1
        self.max_propagation_depth = context.max_propagation_depth
        self.intent                = context.intent

        if (
            self.max_propagation_depth is not None
            and self.propagation_depth > self.max_propagation_depth
        ):
            self._regate_required = True

    def signal_break(self, dimension: str, reason: str, signaled_by: AgentID = "") -> None:
        """
        Flag that a monitored dimension has diverged from its aligned state.

        Records a DimensionBreak and marks the context as requiring a re-gate.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Parameters
        ----------
        dimension : str
            The name of the dimension that broke.
        reason : str
            Human-readable description of the deviation.
        signaled_by : str
            AgentID raising the signal.  May be empty (anonymous sub-agent).
        """
        self.break_signals.append(
            DimensionBreak(
                dimension   = dimension,
                reason      = reason,
                signaled_by = signaled_by,
                timestamp   = datetime.now(timezone.utc),
            )
        )
        self._regate_required = True

    def check_action_permitted(self, action: Action) -> bool:
        """
        Return True if *action* is permitted within this context.

        Blocks if any of the following are true:
          - valid_until has passed
          - a dimension break has been signaled (re-gate required)
          - propagation_depth exceeds max_propagation_depth
          - action.level is above the critical_action_threshold
            (auto re-gate trigger — marks regate required and returns False)
          - action.level is not in scope.permitted_levels

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Parameters
        ----------
        action : Action
            The action the sub-agent wants to perform.

        Returns
        -------
        bool
        """
        # Expiry check
        if datetime.now(timezone.utc) >= self.valid_until:
            self._regate_required = True
            return False

        # Depth check
        if (
            self.max_propagation_depth is not None
            and self.propagation_depth > self.max_propagation_depth
        ):
            self._regate_required = True
            return False

        # Break signal check
        if self._regate_required:
            return False

        # Critical action threshold — auto re-gate trigger
        _level_order = {
            ActionLevel.LOW:      0,
            ActionLevel.MEDIUM:   1,
            ActionLevel.HIGH:     2,
            ActionLevel.CRITICAL: 3,
        }
        if _level_order[action.level] >= _level_order[self.critical_action_threshold]:
            self._regate_required = True
            return False

        # Scope check
        return self.scope.permits(action)

    def verify_intent_integrity(self, parent_intent: TrustIntent) -> List[str]:
        """
        Check that this context's intent preserves all load-bearing constraints
        from parent_intent.

        Returns a list of violated constraint keys — empty means intact.
        If this context carries no intent, every load-bearing constraint from
        parent_intent is reported as violated.

        Typical call site: immediately after receive_context(), to detect
        intent decay introduced at this hop before acting on the context.

        Example
        -------
            violations = child.verify_intent_integrity(parent.intent)
            if violations:
                child.signal_break(
                    "intent_integrity",
                    f"load-bearing constraints dropped: {violations}",
                    signaled_by=self_agent_id,
                )

        Part of PACT-AX trust infrastructure.
        Prose is lossy. Contracts propagate.
        """
        if self.intent is None:
            return [c.key for c in parent_intent.load_bearing_constraints()]
        return parent_intent.verify_preserved_in(self.intent)

    def request_regate(self) -> "TrustAlignmentCheck":
        """
        Return the original TrustAlignmentCheck, ready for re-evaluation.

        The caller must call .evaluate(agent_id) on the returned check and,
        if it passes, issue a new TrustContext via TrustContext.establish().
        This context's operating_mode is updated to REGATE in place.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Returns
        -------
        TrustAlignmentCheck
            The configured check object from initial establishment.
        """
        self.operating_mode = TrustOperatingMode.REGATE
        return self._alignment_check

    # ── Observability ──────────────────────────────────────────────────────────

    @property
    def regate_required(self) -> bool:
        """
        True when the context has been invalidated and a re-gate is needed.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        if datetime.now(timezone.utc) >= self.valid_until:
            return True
        if self.max_propagation_depth is not None and self.propagation_depth > self.max_propagation_depth:
            return True
        return self._regate_required

    def to_dict(self) -> dict:
        """
        Serialisable snapshot of the context state.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        return {
            "context_id":                self.context_id,
            "established_by":            self.established_by,
            "operating_mode":            self.operating_mode.value,
            "valid_until":               self.valid_until.isoformat(),
            "regate_required":           self.regate_required,
            "propagation_depth":         self.propagation_depth,
            "max_propagation_depth":     self.max_propagation_depth,
            "break_signals":             len(self.break_signals),
            "scope":                     self.scope.to_dict(),
            "alignment_at_entry":        self.alignment_at_entry.to_dict(),
            "intent":                    self.intent.to_dict() if self.intent else None,
        }

    def __repr__(self) -> str:
        """
        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        tag = "REGATE" if self.regate_required else self.operating_mode.value.upper()
        return (
            f"TrustContext({self.context_id[:8]}…, "
            f"by={self.established_by!r}, "
            f"mode={tag}, "
            f"depth={self.propagation_depth}, "
            f"breaks={len(self.break_signals)})"
        )
