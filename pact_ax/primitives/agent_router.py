"""
pact_ax/primitives/agent_router.py
────────────────────────────────────
Trust-weighted capability routing.

Given a task description and required skill, AgentRouter queries
CapabilityRegistry for capable agents, then ranks them by trust score
(from the requester's TrustManager).  Agents below min_trust are filtered.

Result
──────
    RouteDecision
        best_agent       — top-ranked agent_id (or None if no candidates)
        candidates       — ranked list of (agent_id, skill, trust_score)
        skill            — the skill that was matched
        strategy_used    — "trust_weighted" | "capability_only" | "none"

Usage
─────
    from pact_ax.primitives.agent_router import AgentRouter

    router = AgentRouter(capability_db="capabilities.db", trust_db="trust.db")
    decision = router.route(
        from_agent="orchestrator",
        skill="contract_review",
        min_trust=0.6,
        top_k=3,
    )
    if decision.best_agent:
        print(f"Route to {decision.best_agent} (trust={decision.candidates[0].trust_score:.2f})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .capability_registry import CapabilityRegistry
from .trust_score import TrustManager


_DEFAULT_TRUST = 0.5     # trust score assumed for unknown agents


@dataclass
class RouteCandidate:
    agent_id:    str
    skill:       str
    description: str
    tags:        List[str]
    trust_score: float
    version:     str = "1.0"

    def to_dict(self):
        return {
            "agent_id":    self.agent_id,
            "skill":       self.skill,
            "description": self.description,
            "tags":        self.tags,
            "trust_score": round(self.trust_score, 4),
            "version":     self.version,
        }


@dataclass
class RouteDecision:
    skill:          str
    from_agent:     str
    best_agent:     Optional[str]
    candidates:     List[RouteCandidate]
    strategy_used:  str
    min_trust:      float
    top_k:          int
    total_capable:  int      # total capable agents before trust filter

    @property
    def routed(self) -> bool:
        return self.best_agent is not None

    def to_dict(self):
        return {
            "skill":         self.skill,
            "from_agent":    self.from_agent,
            "best_agent":    self.best_agent,
            "routed":        self.routed,
            "strategy_used": self.strategy_used,
            "min_trust":     self.min_trust,
            "top_k":         self.top_k,
            "total_capable": self.total_capable,
            "candidates":    [c.to_dict() for c in self.candidates],
        }


class AgentRouter:
    """
    Routes a task to the best-trusted capable agent.

    Parameters
    ----------
    capability_db : path to CapabilityRegistry SQLite file
    trust_db      : path to TrustStore SQLite file (used to load TrustManagers)
    """

    def __init__(
        self,
        capability_db: Union[str, Path] = "capabilities.db",
        trust_db:      Union[str, Path] = "trust.db",
    ):
        self._cap_db   = str(capability_db)
        self._trust_db = str(trust_db)
        self._registry = CapabilityRegistry(self._cap_db)
        self._trust_managers: Dict[str, TrustManager] = {}

    def _get_trust(self, from_agent: str) -> TrustManager:
        if from_agent not in self._trust_managers:
            self._trust_managers[from_agent] = TrustManager.load(
                self._trust_db, agent_id=from_agent
            )
        return self._trust_managers[from_agent]

    def route(
        self,
        from_agent: str,
        skill:      str,
        min_trust:  float = 0.0,
        top_k:      int   = 5,
    ) -> RouteDecision:
        """
        Find the best agent for *skill* from *from_agent*'s perspective.

        Steps
        -----
        1. CapabilityRegistry.find_capable(skill)
        2. Load TrustManager for from_agent, score each candidate
        3. Filter by min_trust, sort descending by trust_score
        4. Return top_k candidates + best_agent

        Strategy falls back to "capability_only" if no trust history exists
        (all scores equal DEFAULT_TRUST=0.5) or if from_agent has no DB entry.
        """
        caps        = self._registry.find_capable(skill)
        total_cap   = len(caps)

        if not caps:
            return RouteDecision(
                skill=skill, from_agent=from_agent, best_agent=None,
                candidates=[], strategy_used="none",
                min_trust=min_trust, top_k=top_k, total_capable=0,
            )

        mgr = self._get_trust(from_agent)

        candidates = []
        for cap in caps:
            if cap.agent_id == from_agent:
                continue  # don't route to self
            trust = mgr.get_trust(cap.agent_id)
            if trust < min_trust:
                continue
            candidates.append(RouteCandidate(
                agent_id=cap.agent_id,
                skill=cap.skill,
                description=cap.description,
                tags=cap.tags,
                trust_score=trust,
                version=cap.version,
            ))

        # Sort by trust descending, then alphabetically for determinism
        candidates.sort(key=lambda c: (-c.trust_score, c.agent_id))
        top = candidates[:top_k]

        # Detect whether trust history actually differentiated candidates
        unique_scores = {round(c.trust_score, 3) for c in top}
        strategy = "capability_only" if len(unique_scores) <= 1 else "trust_weighted"

        return RouteDecision(
            skill=skill,
            from_agent=from_agent,
            best_agent=top[0].agent_id if top else None,
            candidates=top,
            strategy_used=strategy,
            min_trust=min_trust,
            top_k=top_k,
            total_capable=total_cap,
        )

    def route_any(
        self,
        from_agent: str,
        query:      str,
        min_trust:  float = 0.0,
        top_k:      int   = 5,
    ) -> RouteDecision:
        """
        Fuzzy route: search capability descriptions for *query*, then rank by trust.
        Returns the single best RouteDecision across all matching skills.
        """
        matches = self._registry.search(query)
        if not matches:
            return RouteDecision(
                skill=query, from_agent=from_agent, best_agent=None,
                candidates=[], strategy_used="none",
                min_trust=min_trust, top_k=top_k, total_capable=0,
            )

        mgr = self._get_trust(from_agent)

        candidates = []
        for cap in matches:
            if cap.agent_id == from_agent:
                continue
            trust = mgr.get_trust(cap.agent_id)
            if trust < min_trust:
                continue
            candidates.append(RouteCandidate(
                agent_id=cap.agent_id,
                skill=cap.skill,
                description=cap.description,
                tags=cap.tags,
                trust_score=trust,
                version=cap.version,
            ))

        candidates.sort(key=lambda c: (-c.trust_score, c.skill, c.agent_id))
        top = candidates[:top_k]
        unique_scores = {round(c.trust_score, 3) for c in top}
        strategy = "capability_only" if len(unique_scores) <= 1 else "trust_weighted"

        return RouteDecision(
            skill=query,
            from_agent=from_agent,
            best_agent=top[0].agent_id if top else None,
            candidates=top,
            strategy_used=strategy,
            min_trust=min_trust,
            top_k=top_k,
            total_capable=len(matches),
        )
