"""
pact_ax/api/routes/context_share.py
─────────────────────────────────────
REST endpoints for ContextShareManager.

All operations are scoped to an agent_id supplied in the request body.
Managers are created on demand and held in a module-level in-memory registry
(replaced by a persistent store when the REST API layer is fully productionised).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pact_ax.primitives.context_share.manager import ContextShareManager
from pact_ax.primitives.context_share.schemas import ContextType, Priority

router = APIRouter(prefix="/context", tags=["context-share"])

# ── In-memory registry ────────────────────────────────────────────────────────
_managers: Dict[str, ContextShareManager] = {}


def _get_manager(agent_id: str) -> ContextShareManager:
    if agent_id not in _managers:
        _managers[agent_id] = ContextShareManager(agent_id=agent_id)
    return _managers[agent_id]


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterAgentRequest(BaseModel):
    agent_id: str
    agent_type: str = "generic"
    version: str = "1.0.0"
    capabilities: List[str] = []
    specializations: List[str] = []


class CreatePacketRequest(BaseModel):
    agent_id: str
    target_agent: str
    context_type: str
    payload: Dict[str, Any]
    priority: str = "normal"
    ttl_seconds: Optional[int] = None


class AssessTrustRequest(BaseModel):
    agent_id: str
    target_agent: str
    context_type: str
    current_situation: Dict[str, Any] = {}


class SenseCapabilityRequest(BaseModel):
    agent_id: str
    current_task: str
    confidence_threshold: float = 0.7


class UpdateConfidenceRequest(BaseModel):
    agent_id: str
    task: str
    confidence: float


class RecordOutcomeRequest(BaseModel):
    agent_id: str
    target_agent: str
    context_type: str
    outcome: str
    impact: float = 1.0


class PrepareHandoffRequest(BaseModel):
    agent_id: str
    target_agent: str
    current_task: str
    preserve_emotional_context: bool = True
    transfer_ownership: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register or update an agent")
def register_agent(req: RegisterAgentRequest) -> Dict[str, Any]:
    """Create a ContextShareManager for the given agent (idempotent)."""
    _managers[req.agent_id] = ContextShareManager(
        agent_id=req.agent_id,
        agent_type=req.agent_type,
        version=req.version,
        capabilities=req.capabilities,
        specializations=req.specializations,
    )
    return {"registered": True, "agent_id": req.agent_id}


@router.post("/packet", summary="Create a context packet")
def create_packet(req: CreatePacketRequest) -> Dict[str, Any]:
    """Create a trust-aware context packet from one agent to another."""
    try:
        ctx_type = ContextType(req.context_type)
        priority = Priority[req.priority.upper()]
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mgr = _get_manager(req.agent_id)
    packet = mgr.create_context_packet(
        target_agent=req.target_agent,
        context_type=ctx_type,
        payload=req.payload,
        priority=priority,
        ttl_seconds=req.ttl_seconds,
    )
    return {
        "packet_id":     str(packet.metadata.packet_id),
        "from_agent":    packet.from_agent.agent_id,
        "to_agent":      packet.to_agent,
        "context_type":  packet.context_type.value,
        "priority":      packet.priority.value,
        "trust_required": packet.trust_required.name.lower(),
        "expires_at":    packet.expires_at.isoformat() if packet.expires_at else None,
    }


@router.post("/trust", summary="Assess trust for a target agent")
def assess_trust(req: AssessTrustRequest) -> Dict[str, Any]:
    """Return trust assessment and recommendation for sharing context."""
    try:
        ctx_type = ContextType(req.context_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mgr = _get_manager(req.agent_id)
    return mgr.assess_trust(
        target_agent=req.target_agent,
        context_type=ctx_type,
        current_situation=req.current_situation,
    )


@router.post("/capability", summary="Sense capability limit for a task")
def sense_capability(req: SenseCapabilityRequest) -> Dict[str, Any]:
    """Return approaching-limit assessment and handoff recommendation."""
    mgr = _get_manager(req.agent_id)
    return mgr.sense_capability_limit(
        current_task=req.current_task,
        confidence_threshold=req.confidence_threshold,
    )


@router.post("/capability/update", summary="Update task confidence")
def update_confidence(req: UpdateConfidenceRequest) -> Dict[str, Any]:
    """Record a new confidence level for a specific task."""
    if not (0.0 <= req.confidence <= 1.0):
        raise HTTPException(status_code=422, detail="confidence must be in [0.0, 1.0]")
    mgr = _get_manager(req.agent_id)
    mgr.update_capability_confidence(req.task, req.confidence)
    return {"updated": True, "task": req.task, "confidence": req.confidence}


@router.post("/outcome", summary="Record a collaboration outcome")
def record_outcome(req: RecordOutcomeRequest) -> Dict[str, Any]:
    """Record the result of a collaboration to evolve trust over time."""
    try:
        ctx_type = ContextType(req.context_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mgr = _get_manager(req.agent_id)
    mgr.record_collaboration_outcome(
        target_agent=req.target_agent,
        context_type=ctx_type,
        outcome=req.outcome,
        impact=req.impact,
    )
    return {"recorded": True, "target_agent": req.target_agent, "outcome": req.outcome}


@router.post("/handoff", summary="Prepare a handoff context packet")
def prepare_handoff(req: PrepareHandoffRequest) -> Dict[str, Any]:
    """Prepare a high-priority HANDOFF_REQUEST packet for smooth agent transition."""
    mgr = _get_manager(req.agent_id)
    packet = mgr.prepare_handoff(
        target_agent=req.target_agent,
        current_task=req.current_task,
        preserve_emotional_context=req.preserve_emotional_context,
        transfer_ownership=req.transfer_ownership,
    )
    return {
        "packet_id":    str(packet.metadata.packet_id),
        "from_agent":   packet.from_agent.agent_id,
        "to_agent":     packet.to_agent,
        "context_type": packet.context_type.value,
        "payload_keys": list(packet.payload.keys()),
    }


@router.get("/insights/{agent_id}", summary="Get collaboration insights")
def get_insights(agent_id: str) -> Dict[str, Any]:
    """Return trust summary, capability status, and collaboration patterns."""
    mgr = _get_manager(agent_id)
    return mgr.get_collaboration_insights()
