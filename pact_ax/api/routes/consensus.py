"""
pact_ax/api/routes/consensus.py
─────────────────────────────────
REST endpoints for ConsensusProtocol.

Endpoints
─────────
  POST /consensus/run                  — run a single consensus round (stateless)
  POST /consensus/sessions             — create a named session (tracks history)
  POST /consensus/sessions/{sid}/vote  — run a round inside a session
  GET  /consensus/sessions/{sid}       — session metrics
  GET  /consensus/sessions             — list active sessions
  DELETE /consensus/sessions/{sid}     — destroy session
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.coordination.consensus import (
    ConsensusProtocol,
    ConsensusStrategy,
    Vote,
)
from pact_ax.observability.event_bus import get_bus

router = APIRouter(prefix="/consensus", tags=["consensus"])

# ── In-memory session registry ────────────────────────────────────────────────
_sessions: Dict[str, ConsensusProtocol] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class VoteRequest(BaseModel):
    agent_id:   str   = Field(..., min_length=1)
    decision:   str   = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning:  str   = Field("")
    abstain:    bool  = Field(False)


class RunRoundRequest(BaseModel):
    round_id:    Optional[str] = Field(None)
    strategy:    str           = Field("weighted_vote",
                                       description="weighted_vote | quorum | unanimous | confidence_threshold")
    votes:       List[VoteRequest] = Field(..., min_items=1)
    trust_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional {agent_id: trust_score} weights. Defaults to 1.0 for all agents."
    )
    quorum_fraction:       float = Field(0.5, ge=0.0, le=1.0)
    confidence_threshold:  float = Field(0.7, ge=0.0, le=1.0)
    escalation_threshold:  float = Field(0.4, ge=0.0, le=1.0)
    min_votes:             int   = Field(2, ge=1)
    metadata:              Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    session_id:            Optional[str] = Field(None, description="Auto-generated if omitted")
    strategy:              str           = Field("weighted_vote")
    quorum_fraction:       float = Field(0.5, ge=0.0, le=1.0)
    confidence_threshold:  float = Field(0.7, ge=0.0, le=1.0)
    escalation_threshold:  float = Field(0.4, ge=0.0, le=1.0)
    min_votes:             int   = Field(2, ge=1)


class SessionVoteRequest(BaseModel):
    round_id:     Optional[str]        = None
    votes:        List[VoteRequest]    = Field(..., min_items=1)
    trust_scores: Dict[str, float]     = Field(default_factory=dict)
    metadata:     Dict[str, Any]       = Field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_protocol(req: RunRoundRequest | CreateSessionRequest) -> ConsensusProtocol:
    try:
        strategy = ConsensusStrategy(req.strategy)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy {req.strategy!r}. "
                   f"Valid: {[s.value for s in ConsensusStrategy]}"
        )
    return ConsensusProtocol(
        strategy=strategy,
        quorum_fraction=req.quorum_fraction,
        confidence_threshold=req.confidence_threshold,
        escalation_threshold=req.escalation_threshold,
        min_votes=req.min_votes,
    )


def _to_votes(vote_reqs: List[VoteRequest]) -> List[Vote]:
    return [
        Vote(
            agent_id=v.agent_id,
            decision=v.decision,
            confidence=v.confidence,
            reasoning=v.reasoning,
            abstain=v.abstain,
        )
        for v in vote_reqs
    ]


# ── stateless endpoint ────────────────────────────────────────────────────────

@router.post("/run", summary="Run a single consensus round (stateless — no history kept)")
def run_round(req: RunRoundRequest) -> Dict[str, Any]:
    """
    One-shot consensus round. No session created; history is not retained.
    Use /sessions for multi-round scenarios.
    """
    protocol = _build_protocol(req)
    votes    = _to_votes(req.votes)
    result   = protocol.run(
        votes=votes,
        round_id=req.round_id,
        trust_scores=req.trust_scores,
        metadata=req.metadata,
    )
    get_bus().emit("consensus_run",
                   round_id=result.round_id,
                   outcome=result.outcome.value,
                   strategy=req.strategy,
                   vote_count=len(votes))
    return result.to_dict()


# ── session endpoints ─────────────────────────────────────────────────────────

@router.post("/sessions", summary="Create a named consensus session (tracks history)")
def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    sid = req.session_id or f"session-{uuid.uuid4().hex[:8]}"
    if sid in _sessions:
        raise HTTPException(status_code=409, detail=f"Session {sid!r} already exists")
    _sessions[sid] = _build_protocol(req)
    get_bus().emit("consensus_session_created", session_id=sid, strategy=req.strategy)
    return {"session_id": sid, "strategy": req.strategy, "created": True}


@router.get("/sessions", summary="List active consensus sessions")
def list_sessions() -> Dict[str, Any]:
    return {
        "sessions": [
            {"session_id": sid, **p.metrics()}
            for sid, p in _sessions.items()
        ],
        "count": len(_sessions),
    }


@router.post("/sessions/{session_id}/vote", summary="Run a consensus round inside a session")
def session_vote(session_id: str, req: SessionVoteRequest) -> Dict[str, Any]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    protocol = _sessions[session_id]
    votes    = _to_votes(req.votes)
    result   = protocol.run(
        votes=votes,
        round_id=req.round_id,
        trust_scores=req.trust_scores,
        metadata=req.metadata,
    )
    get_bus().emit("consensus_vote",
                   session_id=session_id,
                   round_id=result.round_id,
                   outcome=result.outcome.value,
                   vote_count=len(votes))
    return {**result.to_dict(), "session_id": session_id}


@router.get("/sessions/{session_id}", summary="Get metrics for a consensus session")
def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    p = _sessions[session_id]
    history = [r.to_dict() for r in p.history()]
    return {"session_id": session_id, **p.metrics(), "history": history}


@router.delete("/sessions/{session_id}", summary="Destroy a consensus session")
def delete_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    del _sessions[session_id]
    get_bus().emit("consensus_session_deleted", session_id=session_id)
    return {"deleted": True, "session_id": session_id}
