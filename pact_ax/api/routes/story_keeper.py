"""
pact_ax/api/routes/story_keeper.py
────────────────────────────────────
REST endpoints for StoryKeeper.

Each agent_id gets its own StoryKeeper instance held in a module-level
registry. Sessions are scoped within an agent — one agent can carry multiple
named sessions (story threads) by passing session_id on registration.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.primitives.story_keeper import StoryKeeper, StoryArc

router = APIRouter(prefix="/story", tags=["story-keeper"])

# ── In-memory registry ────────────────────────────────────────────────────────
_keepers: Dict[str, StoryKeeper] = {}


def _get_keeper(agent_id: str) -> StoryKeeper:
    if agent_id not in _keepers:
        _keepers[agent_id] = StoryKeeper(agent_id=agent_id)
    return _keepers[agent_id]


def _serialize_interaction(ix: Dict[str, Any]) -> Dict[str, Any]:
    """Make an interaction dict JSON-safe (datetime → isoformat, Enum → value)."""
    return {
        "timestamp":      ix["timestamp"].isoformat(),
        "user_input":     ix.get("user_input", ""),
        "agent_response": ix.get("agent_response", ""),
        "arc":            ix["arc"].value,
        "metadata":       ix.get("metadata", {}),
    }


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    config: Dict[str, Any] = {}


class ProcessTurnRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    user_id: Optional[str] = None


class ProcessInteractionRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    agent_response: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = {}


class LoadStateRequest(BaseModel):
    state: Dict[str, Any]


class RecallArcRequest(BaseModel):
    arc: str = Field(..., description="exploration | collaboration | integration")
    k: int = Field(3, ge=1, le=50)


class RecallContextRequest(BaseModel):
    query: Optional[str] = None
    prefer_current_arc: bool = True
    k: int = Field(5, ge=1, le=50)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register or reinitialise a StoryKeeper for an agent")
def register(req: RegisterRequest) -> Dict[str, Any]:
    """Create a fresh StoryKeeper for the given agent_id (idempotent)."""
    _keepers[req.agent_id] = StoryKeeper(
        agent_id=req.agent_id,
        session_id=req.session_id,
        config=req.config,
    )
    return {
        "registered": True,
        "agent_id":   req.agent_id,
        "session_id": req.session_id,
    }


@router.post("/{agent_id}/turn", summary="Process a single conversation turn")
def process_turn(agent_id: str, req: ProcessTurnRequest) -> Dict[str, Any]:
    """
    Advance the story by one user turn.
    Returns the narrative beat and current arc.
    """
    keeper = _get_keeper(agent_id)
    beat = keeper.process_turn(
        user_message=req.user_message,
        user_id=req.user_id,
    )
    return {
        "beat":        beat,
        "current_arc": keeper.current_arc.value,
        "interaction_count": len(keeper.interactions),
    }


@router.post("/{agent_id}/interaction", summary="Process a full user+agent interaction")
def process_interaction(agent_id: str, req: ProcessInteractionRequest) -> Dict[str, Any]:
    """
    Record a complete interaction (both sides) and return the enriched result.
    """
    keeper = _get_keeper(agent_id)
    ix = keeper.process_interaction(
        user_input=req.user_input,
        agent_response=req.agent_response,
        metadata=req.metadata,
    )
    return _serialize_interaction(ix)


@router.get("/{agent_id}/state", summary="Get current story state snapshot")
def get_state(agent_id: str) -> Dict[str, Any]:
    """Return the full story state: themes, arc, context, characters, last beat."""
    keeper = _get_keeper(agent_id)
    return keeper.get_story_state()


@router.post("/{agent_id}/state", summary="Load a previously saved story state")
def load_state(agent_id: str, req: LoadStateRequest) -> Dict[str, Any]:
    """Replace the agent's current story state with a saved snapshot."""
    keeper = _get_keeper(agent_id)
    keeper.load_story_state(req.state)
    return {"loaded": True, "agent_id": agent_id}


@router.get("/{agent_id}/summary", summary="Get story summary and arc history")
def get_summary(agent_id: str) -> Dict[str, Any]:
    """Return arc, interaction count, arc transitions, and arc history."""
    keeper = _get_keeper(agent_id)
    return keeper.get_story_summary()


@router.post("/{agent_id}/reset", summary="Reset story back to initial state")
def reset_story(agent_id: str) -> Dict[str, Any]:
    """Clear all interactions, themes, and arc history for this agent."""
    keeper = _get_keeper(agent_id)
    keeper.reset_story()
    return {"reset": True, "agent_id": agent_id}


@router.post("/{agent_id}/recall/arc", summary="Recall recent interactions from a specific arc")
def recall_arc(agent_id: str, req: RecallArcRequest) -> List[Dict[str, Any]]:
    """Return the k most recent interactions from the requested story arc."""
    try:
        arc = StoryArc(req.arc)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown arc {req.arc!r}. Valid: exploration, collaboration, integration",
        )
    keeper = _get_keeper(agent_id)
    return [_serialize_interaction(ix) for ix in keeper.recall_from_arc(arc, k=req.k)]


@router.post("/{agent_id}/recall/context", summary="Story-aware context retrieval")
def recall_context(agent_id: str, req: RecallContextRequest) -> List[Dict[str, Any]]:
    """
    Return up to k interactions most relevant to the current story context.
    Prefers interactions from the current arc unless prefer_current_arc=False.
    """
    keeper = _get_keeper(agent_id)
    results = keeper.recall_for_context(
        query=req.query,
        prefer_current_arc=req.prefer_current_arc,
        k=req.k,
    )
    return [_serialize_interaction(ix) for ix in results]
