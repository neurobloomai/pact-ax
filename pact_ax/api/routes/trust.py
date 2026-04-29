"""
pact_ax/api/routes/trust.py
────────────────────────────
REST endpoints for TrustManager.

Every agent owns a TrustManager keyed by agent_id. Managers are held in
a module-level registry and can be persisted/restored from SQLite via the
save/load endpoints, enabling cross-process and cross-system trust sharing.

Cross-system trust sharing works through two endpoints:
  POST /trust/{agent_id}/external   — ingest a trust signal from another system
  GET  /trust/{agent_id}/network/{target_id} — query transitive network trust

These two together let external systems propagate and consume trust without
sharing implementation details.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.primitives.trust_score import TrustManager
from pact_ax.primitives.context_share.schemas import ContextType, CollaborationOutcome

router = APIRouter(prefix="/trust", tags=["trust"])

# ── In-memory registry ────────────────────────────────────────────────────────
_managers: Dict[str, TrustManager] = {}


def _get_manager(agent_id: str) -> TrustManager:
    if agent_id not in _managers:
        _managers[agent_id] = TrustManager(agent_id=agent_id)
    return _managers[agent_id]


# ── Request / Response models ─────────────────────────────────────────────────

class UpdateTrustRequest(BaseModel):
    target_id: str = Field(..., min_length=1)
    outcome: str = Field(..., description="positive | negative | neutral | partial")
    context_type: str = Field("task_knowledge")
    impact: float = Field(1.0, ge=0.0, le=1.0)


class RecordOutcomeRequest(BaseModel):
    target_id: str = Field(..., min_length=1)
    outcome: str = Field(..., description="positive | negative | neutral | partial")
    context_type: str = Field("task_knowledge")
    impact: float = Field(1.0, ge=0.0, le=1.0)


class DecayRequest(BaseModel):
    target_id: Optional[str] = None
    days_inactive: Optional[float] = None


class ExternalTrustRequest(BaseModel):
    """
    Ingest a trust signal from an external system.

    source_agent    — the agent whose trust opinion we are recording
    target_agent    — the agent being rated
    score           — trust score in [0.0, 1.0] from the source system
    """
    source_agent: str = Field(..., min_length=1)
    target_agent: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)


class TrustedAgentsRequest(BaseModel):
    min_trust: float = Field(0.6, ge=0.0, le=1.0)
    context_type: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register/{agent_id}", summary="Register a TrustManager for an agent")
def register_agent(agent_id: str) -> Dict[str, Any]:
    """Create (or reset) a TrustManager for the given agent (idempotent)."""
    _managers[agent_id] = TrustManager(agent_id=agent_id)
    return {"registered": True, "agent_id": agent_id}


# NOTE: specific two-segment GET paths must come before /{agent_id}/{target_id}
# or FastAPI will swallow them as target_id values.

@router.get("/{agent_id}/insights", summary="Get full trust relationship insights")
def get_insights(agent_id: str) -> Dict[str, Any]:
    """Return per-agent trust summary, trends, and interaction counts."""
    mgr = _get_manager(agent_id)
    return mgr.get_trust_insights()


@router.get("/{agent_id}/{target_id}", summary="Get trust score for a target agent")
def get_trust(
    agent_id: str,
    target_id: str,
    context_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the direct trust score between agent and target."""
    mgr = _get_manager(agent_id)
    ctx = None
    if context_type:
        try:
            ctx = ContextType(context_type)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown context_type: {context_type!r}")

    score = mgr.get_trust(target_id, ctx)
    return {
        "agent_id":    agent_id,
        "target_id":   target_id,
        "score":       round(score, 4),
        "context_type": context_type,
    }


@router.post("/{agent_id}/update", summary="Record a collaboration outcome")
def update_trust(agent_id: str, req: UpdateTrustRequest) -> Dict[str, Any]:
    """Update trust based on the result of a collaboration."""
    mgr = _get_manager(agent_id)
    try:
        outcome = CollaborationOutcome(req.outcome.lower())
        ctx_type = ContextType(req.context_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    new_score = mgr.update_trust(
        target_id=req.target_id,
        outcome=outcome,
        context_type=ctx_type,
        impact=req.impact,
    )
    return {
        "agent_id":    agent_id,
        "target_id":   req.target_id,
        "new_score":   round(new_score, 4),
        "outcome":     req.outcome,
        "context_type": req.context_type,
    }


@router.post("/{agent_id}/decay", summary="Apply time-based trust decay")
def decay_trust(agent_id: str, req: DecayRequest) -> Dict[str, Any]:
    """Drift inactive trust relationships back toward neutral (0.5)."""
    mgr = _get_manager(agent_id)
    mgr.decay_trust(target_id=req.target_id, days_inactive=req.days_inactive)
    return {"decayed": True, "agent_id": agent_id, "target_id": req.target_id}


@router.get("/{agent_id}/network/{target_id}", summary="Get transitive network trust")
def get_network_trust(agent_id: str, target_id: str) -> Dict[str, Any]:
    """
    Infer trust for an agent we haven't interacted with directly,
    using weighted scores from agents we do trust.
    """
    mgr = _get_manager(agent_id)
    score = mgr.get_network_trust(target_id)
    return {
        "agent_id":  agent_id,
        "target_id": target_id,
        "network_trust": round(score, 4),
        "source": "direct" if target_id in mgr._profiles else "transitive",
    }


@router.post("/{agent_id}/external", summary="Ingest external trust signal (cross-system sharing)")
def register_external_trust(agent_id: str, req: ExternalTrustRequest) -> Dict[str, Any]:
    """
    Ingest a trust score from an external system.

    This is the cross-system trust sharing endpoint: a remote service that has
    observed interactions with req.target_agent can POST its trust score here.
    The receiving TrustManager incorporates it as a TRUST_SIGNAL so it can
    inform network trust inference without requiring direct interaction history.
    """
    mgr = _get_manager(agent_id)
    mgr.register_external_trust(
        source_agent=req.source_agent,
        target_agent=req.target_agent,
        score=req.score,
    )
    return {
        "recorded":     True,
        "agent_id":     agent_id,
        "source_agent": req.source_agent,
        "target_agent": req.target_agent,
        "score":        req.score,
    }


@router.post("/{agent_id}/agents", summary="List agents above a trust threshold")
def get_trusted_agents(agent_id: str, req: TrustedAgentsRequest) -> Dict[str, Any]:
    """Return agents whose trust score meets or exceeds min_trust."""
    mgr = _get_manager(agent_id)
    ctx = None
    if req.context_type:
        try:
            ctx = ContextType(req.context_type)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown context_type: {req.context_type!r}")

    agents = mgr.get_trusted_agents(min_trust=req.min_trust, context_type=ctx)
    return {"agent_id": agent_id, "trusted_agents": agents, "min_trust": req.min_trust}


@router.delete("/{agent_id}/{target_id}", summary="Reset trust for a target agent")
def reset_trust(agent_id: str, target_id: str) -> Dict[str, Any]:
    """Remove all trust history for target_id from this agent's TrustManager."""
    mgr = _get_manager(agent_id)
    mgr.reset_trust(target_id)
    return {"reset": True, "agent_id": agent_id, "target_id": target_id}
