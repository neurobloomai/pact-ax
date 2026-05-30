"""
pact_ax/api/routes/agent_router.py
────────────────────────────────────
REST endpoints for AgentRouter.

  POST /route          — exact skill match, ranked by trust
  POST /route/any      — fuzzy query match, ranked by trust
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from pact_ax.observability.event_bus import get_bus
from pact_ax.primitives.agent_router import AgentRouter

router = APIRouter(prefix="/route", tags=["routing"])

_CAP_DB   = os.getenv("PACT_CAP_DB",   "capabilities.db")
_TRUST_DB = os.getenv("PACT_TRUST_DB", "trust.db")
_router   = AgentRouter(capability_db=_CAP_DB, trust_db=_TRUST_DB)


class RouteRequest(BaseModel):
    from_agent: str  = Field(..., min_length=1)
    skill:      str  = Field(..., min_length=1, description="Exact skill name")
    min_trust:  float = Field(0.0, ge=0.0, le=1.0)
    top_k:      int   = Field(5, ge=1, le=20)


class RouteAnyRequest(BaseModel):
    from_agent: str  = Field(..., min_length=1)
    query:      str  = Field(..., min_length=1, description="Keyword to fuzzy-match against capabilities")
    min_trust:  float = Field(0.0, ge=0.0, le=1.0)
    top_k:      int   = Field(5, ge=1, le=20)


@router.post("", summary="Route a task to the best trusted+capable agent (exact skill)")
def route(req: RouteRequest) -> Dict[str, Any]:
    """
    Query CapabilityRegistry for agents with *skill*, rank by trust score
    from *from_agent*'s perspective, filter below *min_trust*.
    """
    decision = _router.route(
        from_agent=req.from_agent,
        skill=req.skill,
        min_trust=req.min_trust,
        top_k=req.top_k,
    )
    get_bus().emit("route_decision",
                   from_agent=req.from_agent,
                   skill=req.skill,
                   best_agent=decision.best_agent or "",
                   strategy=decision.strategy_used,
                   candidates=len(decision.candidates))
    return decision.to_dict()


@router.post("/any", summary="Route by fuzzy keyword search across all capabilities")
def route_any(req: RouteAnyRequest) -> Dict[str, Any]:
    """
    Fuzzy-search capability descriptions for *query*, rank matches by trust.
    Useful when the caller doesn't know the exact skill name.
    """
    decision = _router.route_any(
        from_agent=req.from_agent,
        query=req.query,
        min_trust=req.min_trust,
        top_k=req.top_k,
    )
    get_bus().emit("route_any_decision",
                   from_agent=req.from_agent,
                   query=req.query,
                   best_agent=decision.best_agent or "",
                   strategy=decision.strategy_used,
                   candidates=len(decision.candidates))
    return decision.to_dict()
