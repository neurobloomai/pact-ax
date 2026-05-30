"""
pact_ax/api/routes/dead_letter.py
───────────────────────────────────
REST endpoints for DeadLetterQueue.

  POST   /dlq/enqueue          — add a failed packet
  GET    /dlq/pending          — list pending/retrying entries
  GET    /dlq/exhausted        — list exhausted entries
  GET    /dlq/stats            — counts by status
  GET    /dlq/{id}             — get a specific entry
  POST   /dlq/{id}/retry       — increment attempt, requeue
  POST   /dlq/{id}/resolve     — mark resolved
  DELETE /dlq/{id}             — remove entry
  GET    /dlq                  — full dump
"""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.observability.event_bus import get_bus
from pact_ax.primitives.dead_letter_queue import DeadLetterQueue

router = APIRouter(prefix="/dlq", tags=["dlq"])

_DLQ_DB = os.getenv("PACT_DLQ_DB", "dlq.db")
_dlq    = DeadLetterQueue(_DLQ_DB)


class EnqueueRequest(BaseModel):
    packet_id:    str = Field(..., min_length=1)
    from_agent:   str = Field(..., min_length=1)
    to_agent:     str = Field(..., min_length=1)
    payload:      Dict[str, Any] = Field(default_factory=dict)
    reason:       str = Field("")
    max_attempts: Optional[int] = Field(None, ge=1, le=20)


class RetryRequest(BaseModel):
    reason: str = Field("", description="Updated failure reason for this retry attempt")


@router.post("/enqueue", summary="Enqueue a failed packet into the DLQ")
def enqueue(req: EnqueueRequest) -> Dict[str, Any]:
    entry = _dlq.enqueue(
        packet_id=req.packet_id,
        from_agent=req.from_agent,
        to_agent=req.to_agent,
        payload=req.payload,
        reason=req.reason,
        max_attempts=req.max_attempts,
    )
    get_bus().emit("dlq_enqueued",
                   entry_id=entry.id,
                   from_agent=req.from_agent,
                   to_agent=req.to_agent,
                   packet_id=req.packet_id)
    return entry.to_dict()


@router.get("/pending", summary="List pending and retrying DLQ entries")
def list_pending() -> Dict[str, Any]:
    entries = _dlq.pending()
    return {"entries": [e.to_dict() for e in entries], "count": len(entries)}


@router.get("/exhausted", summary="List exhausted DLQ entries")
def list_exhausted() -> Dict[str, Any]:
    entries = _dlq.exhausted()
    return {"entries": [e.to_dict() for e in entries], "count": len(entries)}


@router.get("/stats", summary="DLQ counts by status")
def stats() -> Dict[str, Any]:
    return _dlq.stats()


@router.get("", summary="Full DLQ dump")
def list_all() -> Dict[str, Any]:
    entries = _dlq.all()
    return {"entries": [e.to_dict() for e in entries], "count": len(entries)}


@router.get("/{entry_id}", summary="Get a specific DLQ entry")
def get_entry(entry_id: str) -> Dict[str, Any]:
    entry = _dlq.get(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"DLQ entry {entry_id!r} not found")
    return entry.to_dict()


@router.post("/{entry_id}/retry", summary="Retry a DLQ entry (increments attempt counter)")
def retry_entry(entry_id: str, req: RetryRequest = RetryRequest()) -> Dict[str, Any]:
    try:
        entry = _dlq.retry(entry_id, reason=req.reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"DLQ entry {entry_id!r} not found")
    get_bus().emit("dlq_retry",
                   entry_id=entry_id,
                   attempt=entry.attempt,
                   status=entry.status)
    return entry.to_dict()


@router.post("/{entry_id}/resolve", summary="Mark a DLQ entry as resolved")
def resolve_entry(entry_id: str) -> Dict[str, Any]:
    try:
        entry = _dlq.resolve(entry_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"DLQ entry {entry_id!r} not found")
    get_bus().emit("dlq_resolved", entry_id=entry_id)
    return entry.to_dict()


@router.delete("/{entry_id}", summary="Delete a DLQ entry")
def delete_entry(entry_id: str) -> Dict[str, Any]:
    removed = _dlq.delete(entry_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"DLQ entry {entry_id!r} not found")
    get_bus().emit("dlq_deleted", entry_id=entry_id)
    return {"deleted": True, "entry_id": entry_id}
