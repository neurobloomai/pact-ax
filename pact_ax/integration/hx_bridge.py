"""
pact_ax/integration/hx_bridge.py
──────────────────────────────────
Bridge between pact-ax StoryKeeper (agent-layer narrative) and
pact-hx MemoryManager (human-layer persistent memory).

What the bridge does
────────────────────
  on_interaction   — every StoryKeeper interaction feeds into pact-hx
                     as an EPISODIC memory
  on_arc_transition — arc shifts trigger pact-hx memory consolidation
                     (episodic → semantic → identity patterns)
  enrich_handoff_context — on handoff, pulls identity + semantic +
                     recent from pact-hx to augment the StateTransfer payload

Without this bridge, a handoff packet carries the arc narrative.
With it, the packet also carries who this person IS — their identity
traits, behavioural patterns, and learned semantics — built up over
weeks or months of interaction.

Usage
─────
    bridge = HXBridge.for_agent("marcus-001")
    keeper = StoryKeeper(agent_id="agent-A", hx_bridge=bridge)
    keeper.process_interaction(user_input, agent_response)
    # → pact-hx stores it as memory automatically

    context = keeper.get_enriched_context()
    # → {"story": {...}, "memory": {"identity": {...}, "semantic": {...}}}
"""

from typing import Any, Dict, Optional


_ARC_TO_VALENCE = {
    "exploration":   "neutral",
    "collaboration": "positive",
    "integration":   "mixed",
}


class HXBridge:
    """
    Connects StoryKeeper to pact-hx MemoryManager.

    Optional — StoryKeeper works perfectly without it.
    When attached, every interaction flows into pact-hx and
    identity context flows back into handoffs.
    """

    def __init__(self, memory_manager: Any):
        self._mm = memory_manager

    @classmethod
    def for_agent(cls, agent_id: str) -> "HXBridge":
        """Create a bridge with a fresh MemoryManager for agent_id."""
        from pact_hx.primitives.memory.manager import MemoryManager
        mm = MemoryManager(agent_id=agent_id, enable_collaboration=True)
        return cls(mm)

    # ── hooks called by StoryKeeper ───────────────────────────────────────────

    def on_interaction(self, agent_id: str, interaction: Dict[str, Any]) -> None:
        """Store a StoryKeeper interaction as an EPISODIC memory in pact-hx."""
        from pact_hx.primitives.memory.schemas import MemoryType, EmotionalValence

        arc_val = (
            interaction["arc"].value
            if hasattr(interaction["arc"], "value")
            else str(interaction["arc"])
        )
        valence_str = _ARC_TO_VALENCE.get(arc_val, "neutral").upper()
        try:
            valence = EmotionalValence(valence_str.lower())
        except ValueError:
            valence = EmotionalValence.NEUTRAL

        user_input     = interaction.get("user_input", "")
        agent_response = interaction.get("agent_response", "")
        content = (
            f"User: {user_input}\nAgent: {agent_response}"
            if agent_response
            else user_input
        )

        topics = [
            w for w in user_input.lower().split()
            if len(w) > 4 and w.isalpha()
        ][:8]

        self._mm.store_memory(
            content=content,
            memory_type=MemoryType.EPISODIC,
            entities=[],
            topics=topics,
            emotional_valence=valence,
        )

    def on_arc_transition(
        self,
        agent_id: str,
        from_arc: Any,
        to_arc: Any,
    ) -> None:
        """Arc transitions are consolidation moments — distil episodic → semantic."""
        self._mm.consolidate_memories()

    # ── context enrichment for handoffs ──────────────────────────────────────

    def enrich_handoff_context(self) -> Dict[str, Any]:
        """
        Pull identity + semantic + recent context from pact-hx.

        Returns a dict suitable for embedding in a StateTransfer payload:
            {
              "identity": { core_traits, preferences, behavioral_patterns },
              "semantic": { learned_patterns, knowledge_domains },
              "recent":   { recent_topics, emotional_trend }
            }
        """
        return {
            "identity": self._mm.share_memory_context("identity"),
            "semantic": self._mm.share_memory_context("semantic"),
            "recent":   self._mm.share_memory_context("recent"),
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Surface pact-hx memory stats for debugging / observability."""
        return self._mm.get_memory_summary()
