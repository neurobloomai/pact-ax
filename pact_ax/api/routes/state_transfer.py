"""
pact_ax/api/routes/state_transfer.py
──────────────────────────────────────
REST endpoints for StateTransferManager.

Each agent has its own manager instance held in a module-level registry.
The in-flight packet (HandoffPacket) is serialised to/from dict so it can
travel across process boundaries via HTTP.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pact_ax.state.state_transfer_manager import (
    StateTransferManager,
    HandoffPacket,
    HandoffReason,
    TransferStatus,
)

router = APIRouter(prefix="/transfer", tags=["state-transfer"])

# ── In-memory registry ────────────────────────────────────────────────────────
_managers: Dict[str, StateTransferManager] = {}


def _get_manager(agent_id: str) -> StateTransferManager:
    if agent_id not in _managers:
        _managers[agent_id] = StateTransferManager(agent_id=agent_id)
    return _managers[agent_id]


# ── Request / Response models ─────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    from_agent_id: str
    to_agent_id: str
    state_data: Dict[str, Any]
    reason: str = "continuation"
    context: Dict[str, Any] = {}


class SendRequest(BaseModel):
    agent_id: str
    packet_id: str


class ReceiveRequest(BaseModel):
    agent_id: str
    packet: Dict[str, Any]   # serialised HandoffPacket (from to_dict())


class RollbackRequest(BaseModel):
    agent_id: str
    packet_id: str


class CheckpointRequest(BaseModel):
    agent_id: str
    label: str
    state_data: Dict[str, Any]


class RestoreRequest(BaseModel):
    agent_id: str
    checkpoint_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/prepare", summary="Prepare a handoff packet")
def prepare_handoff(req: PrepareRequest) -> Dict[str, Any]:
    """Stage a packet for delivery; returns packet_id."""
    try:
        reason = HandoffReason(req.reason)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mgr = _get_manager(req.from_agent_id)
    packet_id = mgr.prepare(
        to_agent_id=req.to_agent_id,
        state_data=req.state_data,
        reason=reason,
        context=req.context,
    )
    return {
        "packet_id": packet_id,
        "from_agent": req.from_agent_id,
        "to_agent":   req.to_agent_id,
        "status":     TransferStatus.PREPARING.value,
    }


@router.post("/send", summary="Mark a prepared packet as in-flight")
def send_packet(req: SendRequest) -> Dict[str, Any]:
    """Transition packet to IN_FLIGHT and return its serialised form."""
    mgr = _get_manager(req.agent_id)
    try:
        packet = mgr.send(req.packet_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Packet {req.packet_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return packet.to_dict()


@router.post("/receive", summary="Receive and integrate a handoff packet")
def receive_packet(req: ReceiveRequest) -> Dict[str, Any]:
    """Validate, integrate, and return the full integration result."""
    try:
        packet = HandoffPacket.from_dict(req.packet)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid packet format: {exc}")

    mgr = _get_manager(req.agent_id)
    result = mgr.receive(packet)

    return {
        "success":          result.success,
        "packet_id":        result.packet_id,
        "integrated_state": result.integrated_state,
        "warnings":         result.warnings,
        "error":            result.error,
    }


@router.post("/rollback", summary="Roll back an integrated packet")
def rollback(req: RollbackRequest) -> Dict[str, Any]:
    """Mark a packet as ROLLED_BACK."""
    mgr = _get_manager(req.agent_id)
    ok = mgr.rollback(req.packet_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot roll back packet {req.packet_id!r} — not found or wrong state",
        )
    return {"rolled_back": True, "packet_id": req.packet_id}


@router.post("/checkpoint", summary="Create a state checkpoint")
def create_checkpoint(req: CheckpointRequest) -> Dict[str, Any]:
    """Snapshot the current state for later rollback."""
    mgr = _get_manager(req.agent_id)
    ckpt_id = mgr.checkpoint(label=req.label, state_data=req.state_data)
    return {"checkpoint_id": ckpt_id, "label": req.label}


@router.post("/restore", summary="Restore from a checkpoint")
def restore_checkpoint(req: RestoreRequest) -> Dict[str, Any]:
    """Return the checkpoint snapshot and re-integrate narrative."""
    mgr = _get_manager(req.agent_id)
    try:
        snap = mgr.restore(req.checkpoint_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Checkpoint {req.checkpoint_id!r} not found")
    return snap


@router.get("/status/{agent_id}/{packet_id}", summary="Get packet status")
def packet_status(agent_id: str, packet_id: str) -> Dict[str, Any]:
    """Return the current transfer status of a packet."""
    mgr = _get_manager(agent_id)
    status = mgr.get_transfer_status(packet_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Packet {packet_id!r} not found")
    return {"packet_id": packet_id, "status": status.value}


@router.get("/checkpoints/{agent_id}", summary="List checkpoints")
def list_checkpoints(agent_id: str) -> List[Dict[str, Any]]:
    """Return all checkpoints for an agent."""
    mgr = _get_manager(agent_id)
    return mgr.list_checkpoints()


@router.get("/summary/{agent_id}", summary="Manager observability snapshot")
def summary(agent_id: str) -> Dict[str, Any]:
    """Return outbound/inbound counts and status breakdown."""
    mgr = _get_manager(agent_id)
    return mgr.summary()
