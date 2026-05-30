"""
pact_ax/api/routes/episodic_memory.py
───────────────────────────────────────
REST endpoints for EpisodicMemory.

  POST   /memory/episodes/{agent_id}             — record an episode
  GET    /memory/episodes/{agent_id}             — recall (with filters)
  GET    /memory/episodes/{agent_id}/{partner_id} — episodes with specific partner
  GET    /memory/summary/{agent_id}              — aggregate stats
  DELETE /memory/episodes/{agent_id}             — clear all episodes for agent
  DELETE /memory/episodes/{agent_id}/{episode_id} — delete one episode
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.observability.event_bus import get_bus
from pact_ax.primitives.episodic_memory import EpisodicMemory, Outcome, Valence

router = APIRouter(prefix="/memory", tags=["memory"])

_MEM_DB = os.getenv("PACT_MEMORY_DB", "episodic.db")
_memory = EpisodicMemory(_MEM_DB)


class RecordEpisodeRequest(BaseModel):
    action:     str   = Field(..., min_length=1)
    partner_id: str   = Field("", description="Other agent involved (empty = solo)")
    outcome:    str   = Field(Outcome.NEUTRAL,
                               description="positive | negative | neutral | partial")
    importance: float = Field(0.5, ge=0.0, le=1.0)
    valence:    str   = Field(Valence.NEUTRAL,
                               description="positive | negative | neutral")
    session_id: str   = Field("")
    tags:       List[str] = Field(default_factory=list)
    context:    Dict[str, Any] = Field(default_factory=dict)
    timestamp:  Optional[str] = Field(None, description="ISO 8601; defaults to now")


@router.post("/episodes/{agent_id}", summary="Record an episodic interaction")
def record(agent_id: str, req: RecordEpisodeRequest) -> Dict[str, Any]:
    ep = _memory.record(
        agent_id=agent_id,
        partner_id=req.partner_id,
        action=req.action,
        outcome=req.outcome,
        importance=req.importance,
        valence=req.valence,
        session_id=req.session_id,
        tags=req.tags,
        context=req.context,
        timestamp=req.timestamp,
    )
    get_bus().emit("episode_recorded",
                   agent_id=agent_id,
                   partner_id=req.partner_id,
                   action=req.action,
                   outcome=req.outcome,
                   importance=req.importance)
    return ep.to_dict()


@router.get("/episodes/{agent_id}", summary="Recall episodes for an agent")
def recall(
    agent_id:       str,
    partner_id:     Optional[str]  = None,
    outcome:        Optional[str]  = None,
    min_importance: float          = 0.0,
    limit:          int            = 50,
    offset:         int            = 0,
) -> Dict[str, Any]:
    episodes = _memory.recall(
        agent_id=agent_id,
        partner_id=partner_id,
        outcome=outcome,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )
    return {
        "agent_id": agent_id,
        "episodes": [e.to_dict() for e in episodes],
        "count":    len(episodes),
    }


@router.get("/episodes/{agent_id}/{partner_id}", summary="Episodes between two agents")
def recall_partner(
    agent_id:   str,
    partner_id: str,
    limit:      int = 20,
) -> Dict[str, Any]:
    episodes = _memory.recall_partner(agent_id=agent_id, partner_id=partner_id, limit=limit)
    return {
        "agent_id":   agent_id,
        "partner_id": partner_id,
        "episodes":   [e.to_dict() for e in episodes],
        "count":      len(episodes),
    }


@router.get("/summary/{agent_id}", summary="Aggregate episodic stats for an agent")
def summary(agent_id: str) -> Dict[str, Any]:
    return _memory.summary(agent_id)


@router.delete("/episodes/{agent_id}", summary="Clear all episodes for an agent")
def clear(agent_id: str) -> Dict[str, Any]:
    count = _memory.clear(agent_id)
    get_bus().emit("episodes_cleared", agent_id=agent_id, count=count)
    return {"cleared": count, "agent_id": agent_id}


@router.delete("/episodes/{agent_id}/{episode_id}", summary="Delete a specific episode")
def delete_episode(agent_id: str, episode_id: str) -> Dict[str, Any]:
    removed = _memory.delete_episode(episode_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id!r} not found")
    return {"deleted": True, "episode_id": episode_id, "agent_id": agent_id}
