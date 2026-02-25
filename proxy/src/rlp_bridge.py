"""
PACT-AX ↔ RLP-0 Bridge

Translates StoryKeeper session state into RLP-0's four relational primitives,
wiring behavioral drift detection into the Relational Ledger Protocol gate.

Layering:
    GitHub API
        ↑
    GitHub MCP Server (Docker)
        ↑
    PACT-AX Proxy  ←─── StoryKeeper (behavioral coherence, AX expression layer)
        ↑                     ↕  (this bridge)
    RLP-0                RelationalState (trust · intent · narrative · commitments)
                              ↓
                         Gate ──► RUPTURE_DETECTED signal ──► block request

Design principle (from RLP-0 DESIGN.md):
    "RLP-0 senses and signals, but does not decide how to respond.
     That is the expression layer's job."

PACT-AX is the expression layer. It supplies state; RLP-0 decides the gate.

Primitive mapping:
    StoryKeeper.trust_level        →  RLP-0 trust         (confidence signal)
    StoryKeeper.last_coherence_score → RLP-0 narrative    (coherence signal)
    StoryKeeper.trajectory         →  RLP-0 intent        (direction signal)
    pattern establishment rate     →  RLP-0 commitments   (accountability signal)
"""

from __future__ import annotations

import logging
from typing import Optional, List

try:
    from rlp_0 import RLP0, Signal, RelationalState, RUPTURE_DETECTED
except ImportError as exc:
    raise ImportError(
        "rlp-0 package not found.\n"
        "Install it:  pip install git+https://github.com/neurobloomai/rlp-0.git"
    ) from exc

from .story_keeper import SessionStory, TrustTrajectory

logger = logging.getLogger("pact-ax.rlp_bridge")

# ---------------------------------------------------------------------------
# Trajectory → RLP-0 intent mapping
# ---------------------------------------------------------------------------
# Intent = directional signal (where the relationship is heading)
# A building trajectory means high intent; violated means zero.
_TRAJECTORY_INTENT: dict = {
    TrustTrajectory.BUILDING:  1.0,
    TrustTrajectory.STABLE:    0.75,
    TrustTrajectory.ERODING:   0.35,
    TrustTrajectory.VIOLATED:  0.0,
}

# How many established patterns = "full commitment confidence"
_FULL_COMMITMENT_AT = 5


class RLPBridge:
    """
    Bridges a single PACT-AX session into RLP-0.

    One RLPBridge instance per MCP session (created by PACTAXProxy).

    Usage:
        bridge = RLPBridge(rupture_threshold=0.6)
        bridge.subscribe(my_handler)     # optional: react to RUPTURE signals

        # after every StoryKeeper.record_event():
        rupture_risk, is_gated = bridge.sync(session_story)
        if is_gated:
            # block the request — RLP-0 gate is closed
    """

    def __init__(self, rupture_threshold: float = 0.6):
        self.rlp = RLP0(rupture_threshold=rupture_threshold)
        self._rupture_signals: List[Signal] = []
        self.rlp.subscribe(self._on_signal)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _on_signal(self, signal: Signal) -> None:
        self._rupture_signals.append(signal)
        logger.warning(
            f"[RLP-0 SIGNAL] {signal.signal_type.name}  "
            f"rupture_risk={signal.rupture_risk:.2f}  context={signal.context}"
        )

    def subscribe(self, callback) -> None:
        """Pass-through: subscribe to RLP-0 signals directly."""
        self.rlp.subscribe(callback)

    # ------------------------------------------------------------------
    # Core sync
    # ------------------------------------------------------------------

    def sync(self, story: SessionStory) -> tuple:
        """
        Translate StoryKeeper session state into RLP-0 primitives and sync.

        Called after every StoryKeeper.record_event().

        Returns:
            (rupture_risk: float, is_gated: bool)
        """
        trust       = _clamp(story.trust_level)
        narrative   = _clamp(story.last_coherence_score)
        intent      = _TRAJECTORY_INTENT.get(story.trajectory, 0.5)
        commitments = _clamp(len(story.established_patterns) / _FULL_COMMITMENT_AT)

        self.rlp.update_state(
            trust=trust,
            intent=intent,
            narrative=narrative,
            commitments=commitments,
        )

        risk    = self.rlp.rupture_risk
        gated   = self.rlp.is_gated

        logger.debug(
            f"rlp.sync  trust={trust:.2f} intent={intent:.2f} "
            f"narrative={narrative:.2f} commitments={commitments:.2f} "
            f"→ rupture_risk={risk:.2f} gated={gated}"
        )

        return risk, gated

    # ------------------------------------------------------------------
    # Gate interaction
    # ------------------------------------------------------------------

    def gate_open(self) -> bool:
        """True if interaction is permitted (gate is open)."""
        return self.rlp.check_gate()

    def acknowledge_repair(self) -> bool:
        """
        Tell RLP-0 a repair was performed and release the gate.
        Call this when the expression layer has resolved the rupture
        (e.g. operator reviewed the alert and cleared the session).
        """
        released = self.rlp.acknowledge_repair()
        if released:
            logger.info("[RLP-0 GATE] Released after repair acknowledgment")
        return released

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def rupture_risk(self) -> float:
        return self.rlp.rupture_risk

    @property
    def is_gated(self) -> bool:
        return self.rlp.is_gated

    @property
    def last_rupture_signal(self) -> Optional[Signal]:
        return self._rupture_signals[-1] if self._rupture_signals else None

    @property
    def rupture_count(self) -> int:
        return len(self._rupture_signals)

    def status(self) -> dict:
        """Full RLP-0 status dict — suitable for JSON serialization."""
        s = self.rlp.status()
        s["rupture_signal_count"] = self.rupture_count
        if self.last_rupture_signal:
            sig = self.last_rupture_signal
            s["last_rupture"] = {
                "type":         sig.signal_type.name,
                "risk":         sig.rupture_risk,
                "timestamp":    sig.timestamp.isoformat(),
                "context":      sig.context,
            }
        return s

    def primitive_snapshot(self) -> dict:
        """The four current RLP-0 primitives — useful for demo output."""
        st = self.rlp.state
        return {
            "trust":        round(st.trust, 3),
            "intent":       round(st.intent, 3),
            "narrative":    round(st.narrative, 3),
            "commitments":  round(st.commitments, 3),
            "rupture_risk": round(st.rupture_risk, 3),
            "is_gated":     st.is_gated,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
