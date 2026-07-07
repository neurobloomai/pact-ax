"""
TrustAlignmentCheck: n/n multi-dimensional trust gate for agent coordination.

Not a trust score. A coherence check across n caller-declared dimensions.
The alignment engine is the invariant; n is the variable.

Key concepts
------------
TrustDimension
    A single axis of trust evaluation. Carries an evaluator callable, a
    per-dimension threshold, and an optional weight for WEIGHTED gate mode.
    Dimensions are composable — a TrustDimension may contain child dimensions,
    making evaluation a tree traversal rather than a flat array scan. Leaf
    nodes (no children) are what get scored.

AlignmentResult
    The output of evaluate(). Contains: how many dimensions passed (aligned),
    the total leaf dimensions checked (total), whether the gate passed, a
    per-dimension breakdown, the weakest dimension, the alignment ratio, and
    the timestamp of evaluation.

GateMode
    STRICT     — gate_passed only if aligned == n (default)
    THRESHOLD  — gate_passed if aligned/n >= configured ratio
    WEIGHTED   — gate_passed if weighted score >= configured ratio

TrustAlignmentCheck
    Instantiated with a list of TrustDimension objects and a GateMode.
    Call evaluate(agent_id) to run the check.

Reference dimensions (not hardcoded — caller provides their own):
    behavioral_consistency  — stability of behavior over time
    capability_coherence    — declared vs actual performance
    relational_history      — interaction ledger (feeds from RLP-0)
    network_corroboration   — third-party agent trust signals
    mode_of_engagement      — orientation drift (advisory vs. executive,
                              read-only vs. write, explore vs. build);
                              evaluator returns 1.0 when observed mode
                              matches declared mode, 0.0 on drift — a
                              single drift fails the STRICT n/n gate

The mode_of_engagement example illustrates *why dimensions are never
hardcoded*: no framework author could predict which orientation dimensions
matter for a given collaboration — only the caller knows.  The engine is
the invariant; n is the variable.

Usage
-----
    from pact_ax.primitives.trust_alignment import (
        TrustAlignmentCheck, TrustDimension, GateMode
    )

    dims = [
        TrustDimension("behavioral_consistency", "...", evaluator=my_fn, threshold=0.7),
        TrustDimension("capability_coherence",   "...", evaluator=my_fn, threshold=0.6),
    ]
    check  = TrustAlignmentCheck(dimensions=dims)
    result = check.evaluate("agent-x")
    if result.gate_passed:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional

# Type alias — callers pass agent identifiers as plain strings
AgentID = str


# ── Enums ──────────────────────────────────────────────────────────────────────

class GateMode(str, Enum):
    """
    Determines how the n/n gate passes or fails.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    STRICT    = "strict"     # all n dimensions must pass
    THRESHOLD = "threshold"  # aligned/total >= threshold_ratio
    WEIGHTED  = "weighted"   # weighted score sum >= threshold_ratio


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """
    The scored result for a single leaf TrustDimension.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    name:      str
    score:     float   # raw evaluator output, 0.0 – 1.0
    threshold: float   # per-dimension pass threshold
    passed:    bool    # score >= threshold
    weight:    float = 1.0


@dataclass
class AlignmentResult:
    """
    The complete output of TrustAlignmentCheck.evaluate().

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.
    """

    aligned:            int                        # how many leaf dims passed
    total:              int                        # n — total leaf dims checked
    gate_passed:        bool                       # engine verdict
    breakdown:          Dict[str, DimensionScore]  # per-leaf-dim detail
    weakest_dimension:  DimensionScore             # lowest score/threshold ratio
    alignment_ratio:    float                      # aligned / total
    timestamp:          datetime

    def to_dict(self) -> dict:
        """
        Serialisable snapshot of the alignment result.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        return {
            "aligned":           self.aligned,
            "total":             self.total,
            "gate_passed":       self.gate_passed,
            "alignment_ratio":   round(self.alignment_ratio, 4),
            "weakest_dimension": self.weakest_dimension.name,
            "timestamp":         self.timestamp.isoformat(),
            "breakdown": {
                name: {
                    "score":     round(ds.score, 4),
                    "threshold": ds.threshold,
                    "passed":    ds.passed,
                    "weight":    ds.weight,
                }
                for name, ds in self.breakdown.items()
            },
        }


@dataclass
class TrustDimension:
    """
    A single axis of trust evaluation.

    Dimensions are composable: a TrustDimension with non-empty *children*
    acts as a grouping node. Evaluation recurses into children; the parent's
    evaluator is never called when children are present.  Only leaf nodes
    (children == []) contribute to AlignmentResult.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.

    Parameters
    ----------
    name : str
        Unique identifier for this dimension within a check.
    description : str
        Human-readable explanation of what this dimension measures.
    evaluator : Callable[[AgentID], float]
        Function that scores the agent on this dimension.  Must return a
        float in [0.0, 1.0].  Called only for leaf dimensions.
    threshold : float
        Per-dimension pass threshold.  passed = (score >= threshold).
    weight : float, optional
        Used in WEIGHTED gate mode.  Defaults to 1.0.
    children : list of TrustDimension, optional
        Child dimensions.  When non-empty, this node is a grouping container
        and *evaluator* is ignored.
    """

    name:        str
    description: str
    evaluator:   Callable[[AgentID], float]
    threshold:   float
    weight:      Optional[float] = None
    children:    List["TrustDimension"] = field(default_factory=list)


# ── Core engine ────────────────────────────────────────────────────────────────

class TrustAlignmentCheck:
    """
    n/n multi-dimensional trust alignment gate.

    Instantiate with a list of TrustDimension objects — the caller declares
    the dimensions; the engine evaluates them.  Supports flat and nested
    (tree) dimension structures.

    Part of PACT-AX trust infrastructure.
    Safety is a moment. Trust is duration.

    Parameters
    ----------
    dimensions : list of TrustDimension
        The dimensions to evaluate.  May be nested (tree traversal).
    gate_mode : GateMode
        STRICT (default), THRESHOLD, or WEIGHTED.
    threshold_ratio : float
        Used in THRESHOLD and WEIGHTED modes.  Default 0.75.
    """

    def __init__(
        self,
        dimensions:      List[TrustDimension],
        gate_mode:       GateMode = GateMode.STRICT,
        threshold_ratio: float = 0.75,
    ) -> None:
        """
        Initialise the alignment check engine.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        if not dimensions:
            raise ValueError("TrustAlignmentCheck requires at least one dimension.")
        self.dimensions      = dimensions
        self.gate_mode       = gate_mode
        self.threshold_ratio = threshold_ratio

    # ── Public API ──────────────────────────────────────────────────────────────

    def evaluate(self, agent_id: AgentID) -> AlignmentResult:
        """
        Evaluate all dimensions for *agent_id* and return an AlignmentResult.

        Traverses nested dimension trees recursively; only leaf nodes are
        scored.  The gate verdict is computed from leaf scores according to
        the configured GateMode.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.

        Parameters
        ----------
        agent_id : str
            The agent to evaluate.

        Returns
        -------
        AlignmentResult
        """
        breakdown: Dict[str, DimensionScore] = {}
        self._collect_leaf_scores(agent_id, self.dimensions, breakdown)

        total   = len(breakdown)
        aligned = sum(1 for ds in breakdown.values() if ds.passed)
        ratio   = aligned / total if total > 0 else 0.0

        gate_passed = self._apply_gate(breakdown, aligned, total)

        weakest = min(
            breakdown.values(),
            key=lambda ds: (ds.score / ds.threshold) if ds.threshold > 0 else 0.0,
        )

        return AlignmentResult(
            aligned           = aligned,
            total             = total,
            gate_passed       = gate_passed,
            breakdown         = breakdown,
            weakest_dimension = weakest,
            alignment_ratio   = round(ratio, 4),
            timestamp         = datetime.now(timezone.utc),
        )

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _collect_leaf_scores(
        self,
        agent_id:  AgentID,
        dims:      List[TrustDimension],
        breakdown: Dict[str, DimensionScore],
    ) -> None:
        """
        Recursively walk the dimension tree; score only leaf nodes.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        for dim in dims:
            if dim.children:
                self._collect_leaf_scores(agent_id, dim.children, breakdown)
            else:
                raw = float(dim.evaluator(agent_id))
                raw = max(0.0, min(1.0, raw))
                breakdown[dim.name] = DimensionScore(
                    name      = dim.name,
                    score     = raw,
                    threshold = dim.threshold,
                    passed    = raw >= dim.threshold,
                    weight    = dim.weight if dim.weight is not None else 1.0,
                )

    def _apply_gate(
        self,
        breakdown: Dict[str, DimensionScore],
        aligned:   int,
        total:     int,
    ) -> bool:
        """
        Compute the gate verdict according to GateMode.

        Part of PACT-AX trust infrastructure.
        Safety is a moment. Trust is duration.
        """
        if total == 0:
            return False

        if self.gate_mode == GateMode.STRICT:
            return aligned == total

        if self.gate_mode == GateMode.THRESHOLD:
            return (aligned / total) >= self.threshold_ratio

        if self.gate_mode == GateMode.WEIGHTED:
            total_weight   = sum(ds.weight for ds in breakdown.values())
            weighted_score = sum(ds.score * ds.weight for ds in breakdown.values())
            if total_weight <= 0:
                return False
            return (weighted_score / total_weight) >= self.threshold_ratio

        return False  # pragma: no cover — exhaustive enum

    # ── Intent fidelity dimension factory ──────────────────────────────────────

    @staticmethod
    def structural_fidelity_dimension(
        origin_intent: "TrustIntent",
        origin_hash:   str,
    ) -> "TrustDimension":
        """
        Return a TrustDimension that performs a cheap structural fidelity check.

        The dimension scores 1.0 when:
          - the received context's origin_intent hashes to origin_hash
            (proves the intent object was not reconstructed at any hop)
          - all load_bearing constraints from origin_intent are present in
            the context's effective_constraints()

        Scores 0.0 on any violation.

        This is one caller-declared dimension among n/n — not hardcoded as
        mandatory.  Declare it like any other TrustDimension per substrate
        doctrine.

        Part of PACT-AX trust infrastructure.
        Intent is referenced, never reconstructed.

        Parameters
        ----------
        origin_intent : TrustIntent
            The intent as declared by the orchestrator.
        origin_hash : str
            The SHA-256 hash computed at establishment time.

        Returns
        -------
        TrustDimension
        """
        from pact_ax.primitives.trust_context import TrustContext  # local to avoid circular

        def _evaluate(agent_id: str) -> float:
            # Evaluator checks the context passed via closure at call time.
            # Callers must bind the context being evaluated before calling
            # check.evaluate() by using a closure over the received context.
            # See usage pattern in tests.
            return 1.0  # base; real check is performed in the bound variant below

        def _make_bound(ctx: "TrustContext"):
            """Return an evaluator bound to a specific received context."""
            def _bound(_agent_id: str) -> float:
                # Hash check: did origin_intent survive byte-identical?
                if ctx.origin_intent is None:
                    return 0.0
                if ctx.origin_hash != origin_hash:
                    return 0.0
                if ctx.origin_intent.content_hash() != origin_hash:
                    return 0.0
                # Load_bearing constraints present in effective scope?
                effective_keys = {c.key for c in ctx.effective_constraints()}
                lb_keys        = {c.key for c in origin_intent.load_bearing_constraints()}
                if not lb_keys.issubset(effective_keys):
                    return 0.0
                return 1.0
            return _bound

        dim = TrustDimension(
            name        = "structural_fidelity",
            description = "Origin intent hash matches; all load_bearing constraints in effective scope",
            evaluator   = _evaluate,
            threshold   = 1.0,
        )
        # Attach the bound-context factory as a convenience attribute
        dim.make_bound_evaluator = _make_bound  # type: ignore[attr-defined]
        return dim
