"""
pact_ax/api/routes/policy_align.py
────────────────────────────────────
REST endpoints for PolicyAlignmentManager and PolicyLearning.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pact_ax.coordination.policy_alignment import (
    PolicyAlignmentManager,
    PolicyDecision,
    PolicyConstraint,
    PolicyConflictResolution,
    PolicyLearning,
)
from pact_ax.primitives.epistemic import ConfidenceLevel, KnowledgeBoundary

router = APIRouter(prefix="/policy", tags=["policy-alignment"])

# ── Singletons ─────────────────────────────────────────────────────────────────
_manager = PolicyAlignmentManager()
_learner  = PolicyLearning()


def _parse_confidence(name: str) -> ConfidenceLevel:
    try:
        return ConfidenceLevel[name.upper()]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown confidence level: {name!r}")


def _decision_from_dict(d: Dict[str, Any]) -> PolicyDecision:
    return PolicyDecision(
        decision=d["decision"],
        confidence=_parse_confidence(d.get("confidence", "MODERATE")),
        reasoning=d.get("reasoning", ""),
        agent_id=d["agent_id"],
        domain=d.get("domain", "general"),
        alternatives_considered=d.get("alternatives_considered", []),
        uncertainty_factors=d.get("uncertainty_factors", []),
    )


def _decision_to_dict(d: PolicyDecision) -> Dict[str, Any]:
    return {
        "decision":               d.decision,
        "confidence":             d.confidence.name,
        "confidence_value":       d.confidence.value,
        "reasoning":              d.reasoning,
        "agent_id":               d.agent_id,
        "domain":                 d.domain,
        "alternatives_considered": d.alternatives_considered,
        "uncertainty_factors":    d.uncertainty_factors,
        "is_confident_enough":    d.is_confident_enough(),
        "should_seek_consensus":  d.should_seek_consensus(),
    }


# ── Request / Response models ─────────────────────────────────────────────────

class AddConstraintRequest(BaseModel):
    name: str
    description: str
    min_confidence: float = 0.6
    requires_specialist: bool = False
    safety_critical: bool = False
    domains: List[str] = []


class EvaluateRequest(BaseModel):
    decision: Dict[str, Any]
    agent_boundaries: List[Dict[str, Any]] = []


class ResolveRequest(BaseModel):
    decisions: List[Dict[str, Any]]
    strategy: str = "defer_confident"


class AlignRequest(BaseModel):
    decisions: List[Dict[str, Any]]
    agent_boundaries: Dict[str, List[Dict[str, Any]]] = {}


class RecordOutcomeRequest(BaseModel):
    decision: Dict[str, Any]
    actual_outcome: str
    was_correct: bool
    feedback: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/constraint", summary="Add a policy constraint")
def add_constraint(req: AddConstraintRequest) -> Dict[str, Any]:
    """Register a constraint that policy decisions must satisfy."""
    constraint = PolicyConstraint(
        name=req.name,
        description=req.description,
        min_confidence=req.min_confidence,
        requires_specialist=req.requires_specialist,
        safety_critical=req.safety_critical,
        domains=set(req.domains),
    )
    _manager.add_constraint(constraint)
    return {"added": True, "constraint": req.name}


@router.post("/evaluate", summary="Evaluate a policy decision against constraints")
def evaluate_decision(req: EvaluateRequest) -> Dict[str, Any]:
    """Return whether a decision passes all registered constraints."""
    try:
        decision = _decision_from_dict(req.decision)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing decision field: {exc}")

    boundaries = [
        KnowledgeBoundary(
            domain=b.get("domain", ""),
            proficiency=_parse_confidence(b.get("proficiency", "MODERATE")),
            known_capabilities=set(b.get("known_capabilities", [])),
            known_limits=set(b.get("known_limits", [])),
        )
        for b in req.agent_boundaries
    ]

    is_valid, issues = _manager.evaluate_decision(decision, boundaries)
    return {
        "valid":    is_valid,
        "issues":   issues,
        "decision": _decision_to_dict(decision),
    }


@router.post("/resolve", summary="Resolve conflict between multiple decisions")
def resolve_conflict(req: ResolveRequest) -> Dict[str, Any]:
    """Apply a conflict resolution strategy and return the winning decision."""
    if not req.decisions:
        raise HTTPException(status_code=422, detail="decisions list is empty")

    try:
        strategy = PolicyConflictResolution(req.strategy)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {req.strategy!r}")

    try:
        decisions = [_decision_from_dict(d) for d in req.decisions]
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}")

    result = _manager.resolve_conflict(decisions, strategy)
    return {
        "resolved_decision": _decision_to_dict(result),
        "strategy_used":     strategy.value,
        "input_count":       len(decisions),
    }


@router.post("/align", summary="Align multiple policy decisions")
def align_policies(req: AlignRequest) -> Dict[str, Any]:
    """Filter invalid decisions, auto-select strategy, return final decision."""
    if not req.decisions:
        raise HTTPException(status_code=422, detail="decisions list is empty")

    try:
        decisions = [_decision_from_dict(d) for d in req.decisions]
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}")

    agent_boundaries: Dict[str, List[KnowledgeBoundary]] = {}
    for agent_id, boundary_list in req.agent_boundaries.items():
        agent_boundaries[agent_id] = [
            KnowledgeBoundary(
                domain=b.get("domain", ""),
                proficiency=_parse_confidence(b.get("proficiency", "MODERATE")),
                known_capabilities=set(b.get("known_capabilities", [])),
                known_limits=set(b.get("known_limits", [])),
            )
            for b in boundary_list
        ]

    final = _manager.align_policies(decisions, agent_boundaries)
    return {
        "final_decision": _decision_to_dict(final),
        "input_count":    len(decisions),
    }


@router.get("/metrics", summary="Policy alignment quality metrics")
def get_metrics() -> Dict[str, Any]:
    """Return aggregate metrics across all policy decisions."""
    return _manager.get_policy_alignment_metrics()


@router.post("/learn/outcome", summary="Record outcome for calibration learning")
def record_outcome(req: RecordOutcomeRequest) -> Dict[str, Any]:
    """Teach the system how well a decision turned out."""
    try:
        decision = _decision_from_dict(req.decision)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}")

    _learner.record_outcome(
        decision=decision,
        actual_outcome=req.actual_outcome,
        was_correct=req.was_correct,
        feedback=req.feedback,
    )
    return {"recorded": True, "agent_id": decision.agent_id}


@router.get("/learn/calibration/{agent_id}", summary="Agent confidence calibration")
def get_calibration(agent_id: str) -> Dict[str, Any]:
    """Return how well-calibrated an agent's confidence predictions are."""
    return _learner.get_agent_calibration(agent_id)
