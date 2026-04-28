"""
TrustManager: Network-wide trust scoring for multi-agent collaboration.

Manages evolving trust relationships between agents, supporting:
- Per-context-type trust tracking with interaction history
- Trust decay over time (inactive relationships drift back to neutral)
- Network reputation (transitive trust inference)
- Collaboration outcome integration
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .context_share.schemas import (
    AgentTrustProfile,
    TrustEvolution,
    ContextType,
    TrustLevel,
    CollaborationOutcome,
    _TRUST_CEILING,
    _TRUST_FLOOR,
)


_DEFAULT_TRUST = 0.5
_DECAY_RATE_PER_DAY = 0.02        # Trust drifts 2% toward neutral per day of inactivity
_TRANSITIVE_WEIGHT = 0.3          # Weight of inferred reputation vs direct experience


class TrustManager:
    """
    Network-wide trust scoring for agent collaboration.

    Tracks direct trust relationships and infers transitive reputation
    across the agent network. All scores stay in [0.0, 1.0].
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # agent_id -> AgentTrustProfile
        self._profiles: Dict[str, AgentTrustProfile] = {}

    # ------------------------------------------------------------------
    # Core trust operations
    # ------------------------------------------------------------------

    def get_trust(
        self,
        target_id: str,
        context_type: Optional[ContextType] = None,
    ) -> float:
        """
        Return trust score for a target agent.

        If context_type is given, returns context-specific trust.
        Otherwise returns the overall trust score.
        """
        profile = self._profiles.get(target_id)
        if profile is None:
            return _DEFAULT_TRUST

        if context_type is not None:
            return profile.get_trust_for_context(context_type)

        return profile.overall_trust

    def update_trust(
        self,
        target_id: str,
        outcome: CollaborationOutcome,
        context_type: ContextType = ContextType.TASK_KNOWLEDGE,
        impact: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Record a collaboration outcome and return the updated trust score.

        Args:
            target_id: The agent whose trust score is being updated
            outcome: Result of the collaboration (POSITIVE / NEGATIVE / NEUTRAL / PARTIAL)
            context_type: The type of collaboration context
            impact: Multiplier for how strongly this outcome affects trust (0.0–1.0)
            context: Optional metadata about the interaction

        Returns:
            Updated context-specific trust score
        """
        profile = self._get_or_create_profile(target_id)
        profile.update_trust(
            context_type=context_type,
            outcome=outcome,
            impact=max(0.0, min(1.0, impact)),
            context=context or {"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        return profile.get_trust_for_context(context_type)

    def record_outcome(
        self,
        target_id: str,
        outcome: str,
        context_type: ContextType = ContextType.TASK_KNOWLEDGE,
        impact: float = 1.0,
    ) -> float:
        """
        Convenience wrapper that accepts a string outcome label.

        Accepts: "positive", "negative", "neutral", "partial"
        """
        try:
            outcome_enum = CollaborationOutcome(outcome.lower())
        except ValueError:
            outcome_enum = CollaborationOutcome.NEUTRAL

        return self.update_trust(
            target_id=target_id,
            outcome=outcome_enum,
            context_type=context_type,
            impact=impact,
        )

    def decay_trust(
        self,
        target_id: Optional[str] = None,
        days_inactive: Optional[float] = None,
    ) -> None:
        """
        Apply time-based trust decay toward neutral (0.5) for inactive relationships.

        If target_id is None, applies decay to all tracked agents.
        If days_inactive is None, infers elapsed time from last_interaction.
        """
        targets = [target_id] if target_id else list(self._profiles.keys())
        now = datetime.now(timezone.utc)

        for tid in targets:
            profile = self._profiles.get(tid)
            if profile is None:
                continue

            if days_inactive is not None:
                elapsed_days = days_inactive
            elif profile.last_interaction is not None:
                delta = now - profile.last_interaction
                elapsed_days = delta.total_seconds() / 86400
            else:
                continue

            decay = _DECAY_RATE_PER_DAY * elapsed_days
            for evolution in profile.trust_evolution.values():
                # Drift toward 0.5 (neutral)
                evolution.current_level += (_DEFAULT_TRUST - evolution.current_level) * decay
                evolution.current_level = max(_TRUST_FLOOR, min(_TRUST_CEILING, evolution.current_level))

            # Recalculate overall trust
            evolutions = list(profile.trust_evolution.values())
            if evolutions:
                profile.overall_trust = sum(e.current_level for e in evolutions) / len(evolutions)

    # ------------------------------------------------------------------
    # Network reputation
    # ------------------------------------------------------------------

    def get_network_trust(self, target_id: str) -> float:
        """
        Infer trust for an unknown agent using the network graph.

        If we have no direct history with target_id, computes a weighted
        average of trust scores from agents who know the target.
        Falls back to _DEFAULT_TRUST if the network has no data.
        """
        if target_id in self._profiles:
            return self._profiles[target_id].overall_trust

        # Look for indirect connections: agents A where we trust A and A trusts target
        referrals: List[float] = []
        for intermediary_id, our_profile in self._profiles.items():
            intermediary_trust_in_target = self._external_trust_score(
                intermediary_id, target_id
            )
            if intermediary_trust_in_target is None:
                continue

            our_trust_in_intermediary = our_profile.overall_trust
            # Weighted contribution: the more we trust the intermediary,
            # the more their opinion of target counts
            referrals.append(our_trust_in_intermediary * intermediary_trust_in_target)

        if not referrals:
            return _DEFAULT_TRUST

        inferred = sum(referrals) / len(referrals)
        # Blend: transitive inference weighted lower than direct experience
        return (_DEFAULT_TRUST * (1 - _TRANSITIVE_WEIGHT)) + (inferred * _TRANSITIVE_WEIGHT)

    def register_external_trust(
        self,
        source_agent: str,
        target_agent: str,
        score: float,
    ) -> None:
        """
        Register a known trust score from another agent (e.g. passed via TrustSignal context).
        Used to build network reputation without direct interaction.
        """
        profile = self._get_or_create_profile(source_agent)
        # Store as a synthetic observation on TRUST_SIGNAL context type
        profile.update_trust(
            context_type=ContextType.TRUST_SIGNAL,
            outcome=CollaborationOutcome.POSITIVE if score >= 0.5 else CollaborationOutcome.NEGATIVE,
            impact=abs(score - 0.5) * 2,  # Map [0,1] divergence from neutral to [0,1] impact
            context={"external_score": score, "target": target_agent},
        )

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def get_trust_insights(self) -> Dict[str, Any]:
        """Return a summary of all tracked trust relationships."""
        insights: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "tracked_agents": len(self._profiles),
            "relationships": {},
        }

        for agent_id, profile in self._profiles.items():
            evolutions = profile.trust_evolution
            total_interactions = sum(len(e.interactions) for e in evolutions.values())
            context_breakdown = {
                ct.value: round(evo.current_level, 3)
                for ct, evo in evolutions.items()
            }
            trending = self._dominant_trend(evolutions)

            insights["relationships"][agent_id] = {
                "overall_trust": round(profile.overall_trust, 3),
                "trust_level": TrustLevel.from_float(profile.overall_trust).name,
                "total_interactions": total_interactions,
                "context_breakdown": context_breakdown,
                "trend": trending,
                "last_interaction": (
                    profile.last_interaction.isoformat()
                    if profile.last_interaction
                    else None
                ),
            }

        return insights

    def get_trusted_agents(
        self,
        min_trust: float = 0.6,
        context_type: Optional[ContextType] = None,
    ) -> List[str]:
        """Return agents whose trust score meets or exceeds min_trust."""
        results = []
        for agent_id, profile in self._profiles.items():
            score = (
                profile.get_trust_for_context(context_type)
                if context_type
                else profile.overall_trust
            )
            if score >= min_trust:
                results.append(agent_id)
        return sorted(results, key=lambda a: self.get_trust(a, context_type), reverse=True)

    def reset_trust(self, target_id: str) -> None:
        """Remove all trust history for a specific agent."""
        self._profiles.pop(target_id, None)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, db_path: Union[str, Path] = "trust.db") -> None:
        """
        Persist all trust profiles to a SQLite file.

        Creates the file (and schema) if it does not yet exist.
        Subsequent calls update existing rows in-place.
        """
        from .trust_store import TrustStore
        store = TrustStore(db_path)
        store.save_all(self.agent_id, self._profiles)

    @classmethod
    def load(
        cls,
        db_path: Union[str, Path],
        agent_id: str,
    ) -> "TrustManager":
        """
        Restore a TrustManager from a previously saved SQLite file.

        Returns a fresh TrustManager (with default state) if the file
        does not exist or contains no rows for *agent_id*.
        """
        from .trust_store import TrustStore
        manager = cls(agent_id=agent_id)
        if not Path(str(db_path)).exists() and str(db_path) != ":memory:":
            return manager
        store = TrustStore(db_path)
        manager._profiles = store.load_profiles(agent_id)
        return manager

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_profile(self, target_id: str) -> AgentTrustProfile:
        if target_id not in self._profiles:
            self._profiles[target_id] = AgentTrustProfile(agent_id=target_id)
        return self._profiles[target_id]

    def _external_trust_score(
        self, intermediary_id: str, target_id: str
    ) -> Optional[float]:
        """
        Placeholder hook for querying another agent's trust score for a target.
        In a networked deployment, this would call an external registry.
        Returns None when no data is available.
        """
        return None

    def _dominant_trend(
        self, evolutions: Dict[ContextType, TrustEvolution]
    ) -> str:
        """Return the dominant trend ('up', 'down', 'stable') across all context types."""
        trends = [e.trend_direction for e in evolutions.values()]
        if not trends:
            return "stable"
        return max(set(trends), key=trends.count)
