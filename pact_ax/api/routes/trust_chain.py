"""
pact_ax/api/routes/trust_chain.py
──────────────────────────────────
REST endpoints for TrustChainManager.

Chains are keyed by chain_id (UUID). The trust resolver is wired to the
existing TrustManager registry so each hop score comes from the initiating
agent's real trust history.

Routes
------
POST /trust-chain/score          Score a path without recording
POST /trust-chain/record         Record a chain and return chain_id
GET  /trust-chain/{chain_id}     Get chain state
POST /trust-chain/{chain_id}/verify   Re-verify against current trust
POST /trust-chain/{chain_id}/complete Mark chain completed
GET  /trust-chain/agent/{agent_id}    List chains involving an agent
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.primitives.trust_chain import (
    TrustChainManager,
    ChainState,
)
from pact_ax.primitives.trust_score import TrustManager

router = APIRouter(prefix="/trust-chain", tags=["trust-chain"])

# ── Shared state ──────────────────────────────────────────────────────────────

# TrustManager registry (mirrors trust.py — chains resolve via the same managers)
_trust_managers: Dict[str, TrustManager] = {}

def _get_trust_manager(agent_id: str) -> TrustManager:
    if agent_id not in _trust_managers:
        _trust_managers[agent_id] = TrustManager(agent_id=agent_id)
    return _trust_managers[agent_id]

def _resolve_trust(from_agent: str, to_agent: str) -> float:
    """Resolve hop trust: from_agent's TrustManager scores to_agent."""
    return _get_trust_manager(from_agent).get_trust(to_agent)

_chain_manager = TrustChainManager(trust_resolver=_resolve_trust)


# ── Request / Response models ─────────────────────────────────────────────────

class ChainPathRequest(BaseModel):
    agents: List[str] = Field(..., min_length=2, description="Ordered list of agent IDs — at least 2")


class RecordChainRequest(BaseModel):
    agents:   List[str]       = Field(..., min_length=2)
    chain_id: Optional[str]   = Field(None, description="Optional — auto-generated if omitted")


# ── Serialisers ───────────────────────────────────────────────────────────────

def _hop_dict(hop) -> Dict[str, Any]:
    return {
        "from_agent":     hop.from_agent,
        "to_agent":       hop.to_agent,
        "baseline_score": round(hop.baseline_score, 4),
        "current_score":  round(hop.current_score, 4),
        "drift":          hop.drift,
        "recorded_at":    hop.recorded_at.isoformat(),
        "last_verified":  hop.last_verified.isoformat() if hop.last_verified else None,
    }

def _chain_dict(chain) -> Dict[str, Any]:
    return {
        "chain_id":      chain.chain_id,
        "agents":        chain.agents,
        "depth":         chain.depth,
        "chain_trust":   chain.chain_trust,
        "coherence":     chain.coherence,
        "state":         chain.state.value,
        "weakest_hop":   _hop_dict(chain.weakest_hop) if chain.weakest_hop else None,
        "hops":          [_hop_dict(h) for h in chain.hops],
        "created_at":    chain.created_at.isoformat(),
        "last_verified": chain.last_verified.isoformat() if chain.last_verified else None,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/score", summary="Score a chain of agents without recording it")
def score_chain(req: ChainPathRequest) -> Dict[str, Any]:
    try:
        s = _chain_manager.score(req.agents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "agents":        s.agents,
        "hop_scores":    [round(x, 4) for x in s.hop_scores],
        "chain_trust":   s.chain_trust,
        "coherence":     s.coherence,
        "state":         s.state.value,
        "weakest_pair":  list(s.weakest_pair),
        "weakest_index": s.weakest_index,
    }


@router.post("/record", summary="Record a trust chain and return its chain_id")
def record_chain(req: RecordChainRequest) -> Dict[str, Any]:
    try:
        chain = _chain_manager.record(req.agents, chain_id=req.chain_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _chain_dict(chain)


@router.get("/{chain_id}", summary="Get chain state")
def get_chain(chain_id: str) -> Dict[str, Any]:
    try:
        return _chain_dict(_chain_manager.get(chain_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")


@router.post("/{chain_id}/verify", summary="Re-verify chain against current trust scores")
def verify_chain(chain_id: str) -> Dict[str, Any]:
    try:
        v = _chain_manager.verify(chain_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")
    return {
        "chain_id":        v.chain_id,
        "previous_state":  v.previous_state.value,
        "current_state":   v.current_state.value,
        "state_changed":   v.state_changed,
        "chain_trust_was": v.chain_trust_was,
        "chain_trust_now": v.chain_trust_now,
        "coherence_was":   v.coherence_was,
        "coherence_now":   v.coherence_now,
        "hop_drift": [
            {
                "from_agent": d.from_agent,
                "to_agent":   d.to_agent,
                "baseline":   round(d.baseline, 4),
                "current":    round(d.current, 4),
                "drift":      d.drift,
                "drifted":    d.drifted,
            }
            for d in v.hop_drift
        ],
        "verified_at": v.verified_at.isoformat(),
    }


@router.post("/{chain_id}/complete", summary="Mark a chain as completed")
def complete_chain(chain_id: str) -> Dict[str, Any]:
    try:
        return _chain_dict(_chain_manager.complete(chain_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")


@router.get("/agent/{agent_id}", summary="List all chains involving an agent")
def list_agent_chains(agent_id: str) -> Dict[str, Any]:
    chains = _chain_manager.list_chains(agent_id=agent_id)
    return {
        "agent_id": agent_id,
        "count":    len(chains),
        "chains":   [_chain_dict(c) for c in chains],
    }
