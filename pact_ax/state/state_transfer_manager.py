"""
pact_ax/state/state_transfer_manager.py
────────────────────────────────────────
PACT-AX State Transfer Manager
The core "wealth transfer protocol" for agent handoffs.

This is the orchestrator that was missing: it ties together

    EpistemicStateTransfer  — knowledge + confidence fidelity
    StateTransferManager    — narrative + story continuity     (primitives)
    Trust scoring           — weighted confidence on receive

into a single, validated, lifecycle-tracked handoff protocol.

Handoff lifecycle
─────────────────
  prepare()  →  PREPARING
  send()     →  IN_FLIGHT
  receive()  →  RECEIVED  →  integrate()  →  INTEGRATED
                          ↘  rollback()   →  ROLLED_BACK
  (any step can → FAILED)

Usage
─────
    from pact_ax.state import StateTransferManager, HandoffReason

    manager = StateTransferManager(agent_id="agent-A")

    # Sender side
    packet_id = manager.prepare(
        to_agent_id="agent-B",
        state_data={"task": "analyse Q3 report", "progress": 0.6},
        epistemic_states=[my_epistemic_state],
        reason=HandoffReason.CONTINUATION,
        context={"priority": "high"},
    )
    packet = manager.send(packet_id)

    # Receiver side (different manager instance, same or remote process)
    receiver = StateTransferManager(agent_id="agent-B")
    result   = receiver.receive(packet)

    if result.success:
        # state, epistemic context, and story are all integrated
        print(result.integrated_state)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class TransferStatus(str, Enum):
    PREPARING   = "preparing"
    IN_FLIGHT   = "in_flight"
    RECEIVED    = "received"
    INTEGRATED  = "integrated"
    FAILED      = "failed"
    ROLLED_BACK = "rolled_back"


class HandoffReason(str, Enum):
    CONTINUATION = "continuation"   # Work continues with a different agent
    ESCALATION   = "escalation"     # Needs higher-capability agent
    PAUSE        = "pause"          # Checkpoint before deferring work
    COMPLETION   = "completion"     # Final hand-back after task done
    LOAD_BALANCE = "load_balance"   # Capacity management


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

class HandoffPacket:
    """
    The complete, self-contained unit of wealth transfer.

    Contains everything the receiving agent needs to resume work with full
    fidelity — state, knowledge (with uncertainty), narrative, and provenance.
    """

    def __init__(
        self,
        packet_id: str,
        from_agent_id: str,
        to_agent_id: str,
        reason: HandoffReason,
        state_data: Dict[str, Any],
        epistemic_payload: List[Dict[str, Any]],
        narrative: Dict[str, Any],
        trust_score: float,
        context: Dict[str, Any],
        created_at: datetime,
    ) -> None:
        self.packet_id         = packet_id
        self.from_agent_id     = from_agent_id
        self.to_agent_id       = to_agent_id
        self.reason            = reason
        self.state_data        = state_data
        self.epistemic_payload = epistemic_payload   # list of serialised EpistemicStates
        self.narrative         = narrative           # story summary from StoryKeeper
        self.trust_score       = trust_score         # 0.0 – 1.0
        self.context           = context
        self.created_at        = created_at
        self.status            = TransferStatus.IN_FLIGHT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id":         self.packet_id,
            "from_agent_id":     self.from_agent_id,
            "to_agent_id":       self.to_agent_id,
            "reason":            self.reason.value,
            "state_data":        self.state_data,
            "epistemic_payload": self.epistemic_payload,
            "narrative":         self.narrative,
            "trust_score":       self.trust_score,
            "context":           self.context,
            "created_at":        self.created_at.isoformat(),
            "status":            self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffPacket":
        packet = cls(
            packet_id         = data["packet_id"],
            from_agent_id     = data["from_agent_id"],
            to_agent_id       = data["to_agent_id"],
            reason            = HandoffReason(data["reason"]),
            state_data        = data["state_data"],
            epistemic_payload = data["epistemic_payload"],
            narrative         = data["narrative"],
            trust_score       = data["trust_score"],
            context           = data.get("context", {}),
            created_at        = datetime.fromisoformat(data["created_at"]),
        )
        packet.status = TransferStatus(data.get("status", TransferStatus.IN_FLIGHT))
        return packet

    def __repr__(self) -> str:
        return (
            f"HandoffPacket({self.packet_id!r}, "
            f"{self.from_agent_id!r}→{self.to_agent_id!r}, "
            f"reason={self.reason.value!r}, status={self.status.value!r})"
        )


class IntegrationResult:
    """Outcome of receive() + integrate()."""

    def __init__(
        self,
        success: bool,
        packet_id: str,
        integrated_state: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success          = success
        self.packet_id        = packet_id
        self.integrated_state = integrated_state or {}
        self.warnings         = warnings or []
        self.error            = error

    def __repr__(self) -> str:
        tag = "OK" if self.success else f"FAIL({self.error})"
        return f"IntegrationResult({tag}, packet={self.packet_id!r})"


class ValidationResult:
    def __init__(
        self,
        valid: bool,
        issues: Optional[List[str]] = None,
    ) -> None:
        self.valid  = valid
        self.issues = issues or []

    def __repr__(self) -> str:
        return f"ValidationResult(valid={self.valid}, issues={self.issues})"


# ──────────────────────────────────────────────────────────────────────────────
# Core orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class StateTransferManager:
    """
    The wealth transfer protocol for PACT-AX agents.

    Orchestrates epistemic fidelity, narrative continuity, and trust into
    a single validated handoff lifecycle.

    Parameters
    ----------
    agent_id : str
        Identity of the agent this manager belongs to.
    story_keeper : optional
        A StoryKeeper instance.  If omitted, narrative context is stubbed.
    trust_floor : float
        Packets with trust_score below this threshold are rejected on receive.
        Default 0.3.
    packet_ttl_minutes : int
        How long completed/failed packets are retained before auto-cleanup.
        Default 120.
    """

    def __init__(
        self,
        agent_id: str,
        story_keeper=None,
        trust_floor: float = 0.3,
        packet_ttl_minutes: int = 120,
    ) -> None:
        self.agent_id            = agent_id
        self.story_keeper        = story_keeper
        self.trust_floor         = trust_floor
        self.packet_ttl_minutes  = packet_ttl_minutes

        self._outbound: Dict[str, HandoffPacket]  = {}   # packets we sent
        self._inbound:  Dict[str, HandoffPacket]  = {}   # packets we received
        self._checkpoints: Dict[str, Dict]        = {}

        # Lazy-import the primitives so this file stays importable even when
        # the optional primitives aren't installed yet.
        self._epistemic_transfer = self._load_epistemic_transfer()

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _load_epistemic_transfer():
        try:
            from pact_ax.state.epistemic_transfer import EpistemicStateTransfer
            return EpistemicStateTransfer()
        except ImportError:
            logger.warning(
                "EpistemicStateTransfer not importable; "
                "epistemic payloads will be passed through as raw dicts."
            )
            return None

    @staticmethod
    def _new_id(prefix: str = "pkt") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def _compute_trust_score(
        self,
        to_agent_id: str,
        state_data: Dict[str, Any],
        epistemic_payload: List[Dict],
        reason: HandoffReason,
    ) -> float:
        """
        Heuristic trust score (0.0 – 1.0).

        In production this should call a dedicated TrustScorer.  The heuristic
        below provides a sensible baseline without external dependencies.
        """
        score = 0.7  # baseline

        # Escalation warrants extra scrutiny
        if reason == HandoffReason.ESCALATION:
            score -= 0.05

        # Higher average confidence in epistemic states → higher trust
        if epistemic_payload:
            avg_conf = sum(
                ep.get("confidence_value", 0.5) for ep in epistemic_payload
            ) / len(epistemic_payload)
            score += (avg_conf - 0.5) * 0.2   # ±0.10 adjustment

        # Richer state context → slightly higher trust
        if len(state_data) >= 5:
            score += 0.05

        return round(min(max(score, 0.0), 1.0), 4)

    def _build_narrative(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pull story summary from StoryKeeper, or return a stub."""
        if self.story_keeper is None:
            return {
                "summary":           "No narrative context available (StoryKeeper not attached).",
                "current_arc":       "unknown",
                "arc_history":       [],
                "interaction_count": 0,
                "what_we_were_doing": "",
                "emotional_gravity":  0.5,
            }
        try:
            summary = self.story_keeper.get_story_summary()
            recent  = self.story_keeper.recall_for_context(k=5)
            what_we_were_doing = self._summarise_recent(recent)
            return {
                "summary":            summary,
                "recent_context":     recent,
                "current_arc":        getattr(self.story_keeper, "current_arc", "unknown"),
                "arc_history":        summary.get("arc_transitions", []),
                "interaction_count":  summary.get("total_interactions", 0),
                "what_we_were_doing": what_we_were_doing,
                "emotional_gravity":  self._emotional_gravity(summary),
            }
        except Exception as exc:
            logger.warning("StoryKeeper error during narrative build: %s", exc)
            return {"summary": "Narrative unavailable.", "error": str(exc),
                    "what_we_were_doing": "", "emotional_gravity": 0.5}

    @staticmethod
    def _summarise_recent(interactions: list) -> str:
        if not interactions:
            return "Starting fresh collaboration"
        excerpts = []
        for ix in interactions[-3:]:
            text = ix.get("user_input", "") if isinstance(ix, dict) else str(ix)
            excerpts.append(text[:60] + "…" if len(text) > 60 else text)
        return " → ".join(excerpts)

    @staticmethod
    def _emotional_gravity(summary: Dict[str, Any]) -> float:
        gravity = 0.5
        if summary.get("total_interactions", 0) > 10:
            gravity += 0.1
        return min(gravity, 1.0)

    def _serialize_epistemic(self, epistemic_states: list) -> List[Dict[str, Any]]:
        """
        Serialise a list of EpistemicState objects using EpistemicStateTransfer,
        falling back to raw dict representation when unavailable.
        """
        serialised = []
        for es in epistemic_states:
            if self._epistemic_transfer and hasattr(es, "value"):
                try:
                    pkg = self._epistemic_transfer.transfer(
                        state         = es,
                        from_agent_id = self.agent_id,
                        to_agent_id   = "__handoff__",
                    )
                    serialised.append(pkg)
                    continue
                except Exception as exc:
                    logger.warning("Epistemic serialisation failed: %s", exc)
            # Fallback: if it's already a dict, carry it through
            if isinstance(es, dict):
                serialised.append(es)
            else:
                serialised.append({"raw": str(es)})
        return serialised

    # ── public API ────────────────────────────────────────────────────────────

    def prepare(
        self,
        to_agent_id: str,
        state_data: Dict[str, Any],
        reason: HandoffReason = HandoffReason.CONTINUATION,
        epistemic_states: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Stage a handoff packet.

        Parameters
        ----------
        to_agent_id : str
            Target agent that will receive the packet.
        state_data : dict
            The task / world state to hand off.
        reason : HandoffReason
            Why the handoff is happening.
        epistemic_states : list of EpistemicState, optional
            Knowledge states to include with full uncertainty fidelity.
        context : dict, optional
            Arbitrary metadata (priority, tags, caller info, …).

        Returns
        -------
        str
            packet_id — pass to send() when ready.
        """
        context          = context or {}
        epistemic_states = epistemic_states or []

        ep_payload  = self._serialize_epistemic(epistemic_states)
        trust_score = self._compute_trust_score(to_agent_id, state_data, ep_payload, reason)
        narrative   = self._build_narrative(context)
        packet_id   = self._new_id("pkt")

        packet = HandoffPacket(
            packet_id         = packet_id,
            from_agent_id     = self.agent_id,
            to_agent_id       = to_agent_id,
            reason            = reason,
            state_data        = state_data,
            epistemic_payload = ep_payload,
            narrative         = narrative,
            trust_score       = trust_score,
            context           = context,
            created_at        = datetime.utcnow(),
        )
        packet.status         = TransferStatus.PREPARING
        self._outbound[packet_id] = packet

        logger.info(
            "Prepared handoff %s: %s→%s (reason=%s, trust=%.2f)",
            packet_id, self.agent_id, to_agent_id, reason.value, trust_score,
        )
        return packet_id

    def send(self, packet_id: str) -> HandoffPacket:
        """
        Mark a prepared packet as IN_FLIGHT and return it for delivery.

        Raises
        ------
        KeyError
            If the packet_id is unknown.
        ValueError
            If the packet is not in PREPARING state.
        """
        if packet_id not in self._outbound:
            raise KeyError(f"Unknown packet_id: {packet_id!r}")

        packet = self._outbound[packet_id]
        if packet.status != TransferStatus.PREPARING:
            raise ValueError(
                f"Cannot send packet in state {packet.status.value!r}. "
                "Only PREPARING packets can be sent."
            )

        packet.status = TransferStatus.IN_FLIGHT
        logger.info("Sent handoff packet %s", packet_id)
        return packet

    def validate(self, packet: HandoffPacket) -> ValidationResult:
        """
        Validate an inbound packet before integration.

        Checks:
         - Packet addressed to this agent
         - Trust score above floor
         - Required fields present
         - Packet not expired (TTL)
        """
        issues: List[str] = []

        if packet.to_agent_id != self.agent_id:
            issues.append(
                f"Packet addressed to {packet.to_agent_id!r}, "
                f"but this agent is {self.agent_id!r}."
            )

        if packet.trust_score < self.trust_floor:
            issues.append(
                f"Trust score {packet.trust_score:.2f} is below floor {self.trust_floor:.2f}."
            )

        if not packet.state_data:
            issues.append("state_data is empty — nothing to hand off.")

        age = datetime.utcnow() - packet.created_at
        if age > timedelta(minutes=self.packet_ttl_minutes):
            issues.append(
                f"Packet is {age.total_seconds()/60:.0f} min old "
                f"(TTL is {self.packet_ttl_minutes} min)."
            )

        return ValidationResult(valid=len(issues) == 0, issues=issues)

    def receive(self, packet: HandoffPacket) -> IntegrationResult:
        """
        Receive and integrate an inbound handoff packet.

        This is the main entry point on the receiver side.  It validates,
        integrates epistemic states, applies narrative context, and returns a
        rich IntegrationResult.

        Parameters
        ----------
        packet : HandoffPacket
            The packet to receive.  Can be a live object or reconstructed via
            ``HandoffPacket.from_dict()``.

        Returns
        -------
        IntegrationResult
        """
        validation = self.validate(packet)
        if not validation.valid:
            packet.status = TransferStatus.FAILED
            self._inbound[packet.packet_id] = packet
            logger.warning(
                "Rejected packet %s: %s", packet.packet_id, validation.issues
            )
            return IntegrationResult(
                success   = False,
                packet_id = packet.packet_id,
                error     = "; ".join(validation.issues),
            )

        packet.status = TransferStatus.RECEIVED
        warnings: List[str] = []

        # ── Reconstruct epistemic states ──────────────────────────────────
        reconstructed_epistemic = []
        if self._epistemic_transfer:
            for ep_pkg in packet.epistemic_payload:
                try:
                    es = self._epistemic_transfer.receive(ep_pkg, self.agent_id)
                    reconstructed_epistemic.append(es)
                except Exception as exc:
                    warnings.append(f"Epistemic reconstruction failed: {exc}")
        else:
            reconstructed_epistemic = packet.epistemic_payload  # raw dicts

        # ── Integrate narrative into StoryKeeper ─────────────────────────
        if self.story_keeper and packet.narrative.get("recent_context"):
            try:
                for interaction in packet.narrative["recent_context"]:
                    if isinstance(interaction, dict):
                        self.story_keeper.process_interaction(
                            user_input     = interaction.get("input", ""),
                            agent_response = interaction.get("response", ""),
                        )
            except Exception as exc:
                warnings.append(f"Narrative integration warning: {exc}")

        # ── Build integrated state ────────────────────────────────────────
        integrated_state = {
            "state_data":            packet.state_data,
            "epistemic_states":      reconstructed_epistemic,
            "narrative_context":     packet.narrative,
            "handoff_provenance": {
                "from_agent":    packet.from_agent_id,
                "reason":        packet.reason.value,
                "trust_score":   packet.trust_score,
                "transferred_at": packet.created_at.isoformat(),
                "received_at":   datetime.utcnow().isoformat(),
            },
            "context": packet.context,
        }

        packet.status = TransferStatus.INTEGRATED
        self._inbound[packet.packet_id] = packet

        logger.info(
            "Integrated packet %s from %s (trust=%.2f, warnings=%d)",
            packet.packet_id, packet.from_agent_id,
            packet.trust_score, len(warnings),
        )

        return IntegrationResult(
            success          = True,
            packet_id        = packet.packet_id,
            integrated_state = integrated_state,
            warnings         = warnings,
        )

    def rollback(self, packet_id: str) -> bool:
        """
        Roll back a received packet — mark it ROLLED_BACK and remove its
        contributions from the local state (best-effort).

        Returns True if the rollback was applied, False if not found.
        """
        packet = self._inbound.get(packet_id)
        if packet is None:
            logger.warning("rollback: packet %r not found in inbound.", packet_id)
            return False

        if packet.status not in (TransferStatus.RECEIVED, TransferStatus.INTEGRATED):
            logger.warning(
                "rollback: packet %r is in state %r — cannot roll back.",
                packet_id, packet.status.value,
            )
            return False

        packet.status = TransferStatus.ROLLED_BACK
        logger.info("Rolled back packet %s", packet_id)
        return True

    # ── Checkpoint management ─────────────────────────────────────────────────

    def checkpoint(
        self,
        label: str,
        state_data: Dict[str, Any],
        epistemic_states: Optional[list] = None,
    ) -> str:
        """
        Create a named checkpoint capturing the current complete state.

        Parameters
        ----------
        label : str
            Human-readable name for this checkpoint.
        state_data : dict
            Current task state to snapshot.
        epistemic_states : list, optional
            Current epistemic states to snapshot.

        Returns
        -------
        str
            checkpoint_id — pass to restore() to roll back.
        """
        ckpt_id   = self._new_id("ckpt")
        ep_snap   = self._serialize_epistemic(epistemic_states or [])
        narrative = self._build_narrative({})

        self._checkpoints[ckpt_id] = {
            "checkpoint_id":    ckpt_id,
            "label":            label,
            "agent_id":         self.agent_id,
            "state_data":       state_data,
            "epistemic_snap":   ep_snap,
            "narrative_snap":   narrative,
            "created_at":       datetime.utcnow().isoformat(),
        }

        logger.info("Checkpoint %s (%r) created for agent %s", ckpt_id, label, self.agent_id)
        return ckpt_id

    def restore(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore state from a previously created checkpoint.

        Returns the full checkpoint snapshot dict.

        Raises
        ------
        KeyError
            If the checkpoint_id is unknown.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found.")

        snap = self._checkpoints[checkpoint_id]

        # Re-integrate narrative if StoryKeeper is available
        if self.story_keeper and snap.get("narrative_snap", {}).get("recent_context"):
            try:
                for interaction in snap["narrative_snap"]["recent_context"]:
                    if isinstance(interaction, dict):
                        self.story_keeper.process_interaction(
                            user_input     = interaction.get("input", ""),
                            agent_response = interaction.get("response", ""),
                        )
            except Exception as exc:
                logger.warning("Narrative restore warning: %s", exc)

        logger.info(
            "Restored checkpoint %s (%r) for agent %s",
            checkpoint_id, snap["label"], self.agent_id,
        )
        return snap

    def list_checkpoints(self) -> List[Dict[str, str]]:
        """Return a summary list of all checkpoints for this manager."""
        return [
            {
                "checkpoint_id": ckpt["checkpoint_id"],
                "label":         ckpt["label"],
                "created_at":    ckpt["created_at"],
            }
            for ckpt in self._checkpoints.values()
        ]

    # ── Observability ─────────────────────────────────────────────────────────

    def get_active_transfers(self) -> List[HandoffPacket]:
        """Return all IN_FLIGHT outbound packets."""
        return [
            p for p in self._outbound.values()
            if p.status == TransferStatus.IN_FLIGHT
        ]

    def get_transfer_status(self, packet_id: str) -> Optional[TransferStatus]:
        """Return the status of any known packet (outbound or inbound)."""
        pkt = self._outbound.get(packet_id) or self._inbound.get(packet_id)
        return pkt.status if pkt else None

    def clear_completed(self) -> int:
        """
        Remove packets older than TTL that are INTEGRATED, FAILED, or ROLLED_BACK.
        Scans both outbound and inbound stores. Returns the number cleared.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=self.packet_ttl_minutes)
        terminal = {TransferStatus.INTEGRATED, TransferStatus.FAILED, TransferStatus.ROLLED_BACK}
        to_remove_out = [
            pid for pid, p in self._outbound.items()
            if p.status in terminal and p.created_at < cutoff
        ]
        to_remove_in = [
            pid for pid, p in self._inbound.items()
            if p.status in terminal and p.created_at < cutoff
        ]
        for pid in to_remove_out:
            del self._outbound[pid]
        for pid in to_remove_in:
            del self._inbound[pid]
        total = len(to_remove_out) + len(to_remove_in)
        if total:
            logger.info("Cleared %d completed packets.", total)
        return total

    def summary(self) -> Dict[str, Any]:
        """Return an observability snapshot of this manager's state."""
        status_counts: Dict[str, int] = {}
        for p in list(self._outbound.values()) + list(self._inbound.values()):
            status_counts[p.status.value] = status_counts.get(p.status.value, 0) + 1

        return {
            "agent_id":          self.agent_id,
            "outbound_count":    len(self._outbound),
            "inbound_count":     len(self._inbound),
            "checkpoint_count":  len(self._checkpoints),
            "active_transfers":  len(self.get_active_transfers()),
            "status_breakdown":  status_counts,
            "trust_floor":       self.trust_floor,
        }

    # ── Convenience / backward-compat wrappers ───────────────────────────────

    def prepare_handoff(
        self,
        target_agent: str,
        state_data: Dict[str, Any],
        handoff_reason: str = "continuation",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare and send a handoff in one call; return a narrative dict.

        Convenience wrapper around prepare() + send() for callers that want
        a simple dict-based interface rather than the full packet lifecycle.
        """
        reason_map = {r.value: r for r in HandoffReason}
        reason = reason_map.get(handoff_reason, HandoffReason.CONTINUATION)

        packet_id = self.prepare(
            to_agent_id=target_agent,
            state_data=state_data,
            reason=reason,
            context=context or {},
        )
        packet = self.send(packet_id)

        gravity = self._emotional_gravity_from_reason(handoff_reason, state_data)
        return {
            "state": state_data,
            "story_context": packet.narrative.get("summary"),
            "narrative": {
                "what_we_were_doing": packet.narrative.get("what_we_were_doing", ""),
                "emotional_gravity":  gravity,
                "why_it_matters":     self._why_it_matters(state_data, handoff_reason),
                "current_arc":        str(packet.narrative.get("current_arc", "unknown")),
                "handoff_reason":     handoff_reason,
                "continuity_preserved": self.story_keeper is not None,
            },
            "transfer_meta": {
                "from_agent":    self.agent_id,
                "to_agent":      target_agent,
                "timestamp":     packet.created_at.isoformat(),
                "handoff_reason": handoff_reason,
                "transfer_id":   packet.packet_id,
            },
            "additional_context": context or {},
            "_packet": packet,
        }

    def receive_handoff(
        self,
        transfer_packet: Any,
        integrate_story: bool = True,
    ) -> Dict[str, Any]:
        """
        Receive a handoff and return a confirmation dict.

        Accepts either a HandoffPacket directly or the dict produced by
        prepare_handoff() (which carries a ``_packet`` key).
        """
        if isinstance(transfer_packet, HandoffPacket):
            packet = transfer_packet
        elif isinstance(transfer_packet, dict) and "_packet" in transfer_packet:
            packet = transfer_packet["_packet"]
        elif isinstance(transfer_packet, dict):
            packet = HandoffPacket.from_dict(transfer_packet)
        else:
            raise TypeError(f"Unsupported transfer_packet type: {type(transfer_packet)}")

        result = self.receive(packet)
        narrative = (
            transfer_packet.get("narrative", {})
            if isinstance(transfer_packet, dict)
            else packet.narrative
        )

        return {
            "received":        result.success,
            "from_agent":      packet.from_agent_id,
            "state":           result.integrated_state.get("state_data", {}),
            "story_integrated": integrate_story and self.story_keeper is not None,
            "ready_to_continue": result.success,
            "understanding":   narrative,
            "timestamp":       datetime.utcnow().isoformat(),
            "error":           result.error,
        }

    def create_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Optional[Dict[str, Any]] = None,
        include_full_story: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a named checkpoint; return the checkpoint dict.

        Convenience wrapper around checkpoint() that matches the
        primitives-style API (name + optional state_data).
        """
        ckpt_id = self.checkpoint(
            label=checkpoint_name,
            state_data=state_data or {},
        )
        return self._checkpoints[ckpt_id]

    @staticmethod
    def _emotional_gravity_from_reason(reason: str, state_data: Dict[str, Any]) -> float:
        gravity = 0.5
        if reason == "escalation":
            gravity += 0.3
        elif reason == "completion":
            gravity += 0.2
        if state_data.get("progress", 0) > 0.7:
            gravity += 0.2
        return min(gravity, 1.0)

    @staticmethod
    def _why_it_matters(state_data: Dict[str, Any], reason: str) -> str:
        task = state_data.get("current_task", "current task")
        messages = {
            "continuation": f"Continuing work on: {task}",
            "pause":        f"Pausing work on: {task} (to be resumed later)",
            "escalation":   f"Escalating: {task} (needs higher-capability agent)",
            "completion":   f"Completing: {task} (final handoff)",
        }
        return messages.get(reason, f"Transferring: {task}")

    def __repr__(self) -> str:
        return (
            f"StateTransferManager(agent_id={self.agent_id!r}, "
            f"outbound={len(self._outbound)}, inbound={len(self._inbound)})"
        )
