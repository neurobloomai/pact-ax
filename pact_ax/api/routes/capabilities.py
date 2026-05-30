"""
pact_ax/api/routes/capabilities.py
────────────────────────────────────
REST endpoints for CapabilityRegistry.

Agents declare what they can do; other agents discover capable peers.
Optionally pairs with /trust to filter candidates by minimum trust score.

Endpoints
─────────
  POST   /capabilities/register               — declare a skill
  DELETE /capabilities/{agent_id}/{skill}     — remove a skill
  DELETE /capabilities/{agent_id}             — remove all skills for an agent
  GET    /capabilities/{agent_id}             — list an agent's skills
  POST   /capabilities/find                   — find agents for a skill
  POST   /capabilities/search                 — fuzzy search by keyword
  GET    /capabilities/skills                 — all known skill names
  GET    /capabilities                        — full registry dump
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.observability.event_bus import get_bus
from pact_ax.primitives.capability_registry import CapabilityRegistry

router = APIRouter(prefix="/capabilities", tags=["capabilities"])

_CAP_DB = os.getenv("PACT_CAP_DB", "capabilities.db")
_registry = CapabilityRegistry(_CAP_DB)


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterCapabilityRequest(BaseModel):
    agent_id:    str = Field(..., min_length=1)
    skill:       str = Field(..., min_length=1)
    description: str = Field("", description="Human-readable description of what this skill does")
    tags:        List[str] = Field(default_factory=list)
    version:     str = Field("1.0")


class FindCapableRequest(BaseModel):
    skill:     str = Field(..., min_length=1, description="Exact skill name to look up")
    min_trust: Optional[float] = Field(None, ge=0.0, le=1.0,
                                       description="If set, cross-check /trust and filter below threshold")
    requester: Optional[str] = Field(None,
                                     description="Agent ID of requester — required if min_trust is set")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Keyword to search in skill names, descriptions, and tags")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register a capability for an agent")
def register_capability(req: RegisterCapabilityRequest) -> Dict[str, Any]:
    cap = _registry.register(
        agent_id=req.agent_id,
        skill=req.skill,
        description=req.description,
        tags=req.tags,
        version=req.version,
    )
    get_bus().emit("capability_registered",
                   agent_id=req.agent_id, skill=req.skill, tags=req.tags)
    return {"registered": True, **cap.to_dict()}


@router.delete("/{agent_id}/{skill}", summary="Remove a specific capability")
def deregister_capability(agent_id: str, skill: str) -> Dict[str, Any]:
    removed = _registry.deregister(agent_id, skill)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No capability {skill!r} for agent {agent_id!r}")
    get_bus().emit("capability_deregistered", agent_id=agent_id, skill=skill)
    return {"removed": True, "agent_id": agent_id, "skill": skill}


@router.delete("/{agent_id}", summary="Remove all capabilities for an agent")
def deregister_agent(agent_id: str) -> Dict[str, Any]:
    count = _registry.deregister_agent(agent_id)
    get_bus().emit("agent_capabilities_cleared", agent_id=agent_id, count=count)
    return {"removed": count, "agent_id": agent_id}


@router.get("/skills", summary="List all known skill names")
def list_skills() -> Dict[str, Any]:
    skills = _registry.all_skills()
    return {"skills": skills, "count": len(skills)}


@router.get("", summary="Dump full capability registry")
def list_all() -> Dict[str, Any]:
    caps = _registry.all_capabilities()
    return {"capabilities": [c.to_dict() for c in caps], "count": len(caps)}


@router.get("/{agent_id}", summary="List capabilities for an agent")
def get_agent_capabilities(agent_id: str) -> Dict[str, Any]:
    caps = _registry.get_agent_capabilities(agent_id)
    return {
        "agent_id":     agent_id,
        "capabilities": [c.to_dict() for c in caps],
        "count":        len(caps),
    }


@router.post("/find", summary="Find agents capable of a skill")
def find_capable(req: FindCapableRequest) -> Dict[str, Any]:
    """
    Returns agents registered for *skill*.

    If min_trust + requester are provided, also calls /trust to filter
    candidates whose trust score (from requester's perspective) is below
    the threshold. This requires the trust routes to be wired up and the
    requester to have prior interactions recorded.
    """
    caps = _registry.find_capable(req.skill)

    candidates = [c.to_dict() for c in caps]

    if req.min_trust is not None and req.requester:
        from pact_ax.api.routes.trust import _get_manager
        mgr = _get_manager(req.requester)
        candidates = [
            {**c, "trust_score": round(mgr.get_trust(c["agent_id"]), 4)}
            for c in candidates
            if mgr.get_trust(c["agent_id"]) >= req.min_trust
        ]

    get_bus().emit("capability_query",
                   skill=req.skill, matches=len(candidates),
                   requester=req.requester or "")
    return {
        "skill":      req.skill,
        "candidates": candidates,
        "count":      len(candidates),
    }


@router.post("/search", summary="Fuzzy search capabilities by keyword")
def search_capabilities(req: SearchRequest) -> Dict[str, Any]:
    caps = _registry.search(req.query)
    get_bus().emit("capability_search", query=req.query, matches=len(caps))
    return {
        "query":   req.query,
        "results": [c.to_dict() for c in caps],
        "count":   len(caps),
    }
