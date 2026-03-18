"""
pact_ax/coordination/consensus.py
──────────────────────────────────
PACT-AX Consensus Protocol

Implements the missing concrete mechanism behind
``PolicyConflictResolution.CONSENSUS_REQUIRED``.

When agents disagree, this module runs a structured consensus round:
  1. Each agent casts a ``Vote`` (decision + confidence + reasoning)
  2. ``ConsensusProtocol`` aggregates votes using one of four strategies
  3. A ``ConsensusResult`` is returned — accepted, deadlocked, or escalated

Strategies
──────────
  WEIGHTED_VOTE       confidence × trust-score weighted majority
  QUORUM              simple majority with a configurable quorum fraction
  UNANIMOUS           all agents must agree (use for safety-critical policies)
  CONFIDENCE_THRESHOLD  accept if the winner clears a minimum confidence bar

Integration points
──────────────────
  • Trust scores from ``trust_primitives.TrustNetwork`` weight the votes
  • ``EpistemicState`` from ``primitives.epistemic`` carries per-vote confidence
  • ``policy_alignment.PolicyConflictResolution.CONSENSUS_REQUIRED`` delegates here
  • ``coordination_bus`` publishes CONSENSUS_REACHED / CONSENSUS_FAILED events

Usage
─────
    from pact_ax.coordination.consensus import (
        ConsensusProtocol, ConsensusStrategy, Vote
    )

    protocol = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)

    votes = [
        Vote(agent_id="agent-A", decision="deploy-v2",
             confidence=0.85, reasoning="Metrics are stable"),
        Vote(agent_id="agent-B", decision="deploy-v2",
             confidence=0.72, reasoning="Risk acceptable"),
        Vote(agent_id="agent-C", decision="hold",
             confidence=0.65, reasoning="Need more data"),
    ]

    result = protocol.run(
        round_id    = "deploy-decision-42",
        votes       = votes,
        trust_scores = {"agent-A": 0.9, "agent-B": 0.8, "agent-C": 0.75},
    )

    if result.reached:
        print(result.winning_decision)   # "deploy-v2"
    else:
        print(result.outcome)            # DEADLOCK or ESCALATE_TO_HUMAN
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusStrategy(str, Enum):
    WEIGHTED_VOTE         = "weighted_vote"          # trust × confidence weighted majority
    QUORUM                = "quorum"                 # simple majority, configurable quorum
    UNANIMOUS             = "unanimous"              # all agents must agree
    CONFIDENCE_THRESHOLD  = "confidence_threshold"  # winner must clear a confidence bar


class ConsensusOutcome(str, Enum):
    ACCEPTED             = "accepted"           # consensus reached, decision adopted
    DEADLOCK             = "deadlock"           # no decision crosses threshold
    ESCALATE_TO_HUMAN    = "escalate_to_human" # disagreement too sharp, needs human
    INSUFFICIENT_VOTES   = "insufficient_votes" # not enough agents voted


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Vote:
    """A single agent's position in a consensus round."""

    agent_id:   str
    decision:   str              # what this agent proposes (e.g. "deploy-v2", "hold")
    confidence: float            # 0.0 – 1.0 epistemic confidence in the decision
    reasoning:  str = ""         # human-readable justification
    abstain:    bool = False     # agent may abstain (uncertainty too high)
    cast_at:    datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Vote confidence must be in [0, 1]; got {self.confidence}"
            )

    def effective_weight(self, trust_score: float = 1.0) -> float:
        """confidence × trust_score — the weight this vote carries."""
        if self.abstain:
            return 0.0
        return self.confidence * max(0.0, min(1.0, trust_score))


@dataclass
class ConsensusResult:
    """Complete outcome of a consensus round."""

    round_id:           str
    outcome:            ConsensusOutcome
    winning_decision:   Optional[str]         # None on DEADLOCK / ESCALATE
    total_weight:       float                 # sum of all effective weights
    winning_weight:     float                 # weight behind the winning decision
    vote_breakdown:     Dict[str, float]      # decision → total weight
    dissent_map:        Dict[str, List[str]]  # decision → [agent_ids who voted for it]
    abstentions:        List[str]             # agent_ids that abstained
    confidence_score:   float                 # 0–1 measure of consensus quality
    strategy_used:      ConsensusStrategy
    decided_at:         datetime
    metadata:           Dict[str, Any] = field(default_factory=dict)

    @property
    def reached(self) -> bool:
        return self.outcome == ConsensusOutcome.ACCEPTED

    @property
    def needs_human(self) -> bool:
        return self.outcome == ConsensusOutcome.ESCALATE_TO_HUMAN

    def summary(self) -> str:
        if self.reached:
            pct = (self.winning_weight / self.total_weight * 100
                   if self.total_weight > 0 else 0)
            return (
                f"Consensus REACHED: '{self.winning_decision}' "
                f"({pct:.1f}% weight, confidence={self.confidence_score:.2f})"
            )
        return f"Consensus {self.outcome.value.upper()} — no decision adopted."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id":          self.round_id,
            "outcome":           self.outcome.value,
            "winning_decision":  self.winning_decision,
            "total_weight":      round(self.total_weight, 4),
            "winning_weight":    round(self.winning_weight, 4),
            "vote_breakdown":    {k: round(v, 4) for k, v in self.vote_breakdown.items()},
            "dissent_map":       self.dissent_map,
            "abstentions":       self.abstentions,
            "confidence_score":  round(self.confidence_score, 4),
            "strategy_used":     self.strategy_used.value,
            "decided_at":        self.decided_at.isoformat(),
            "metadata":          self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core protocol
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusProtocol:
    """
    Runs a structured consensus round over a set of agent votes.

    Parameters
    ----------
    strategy : ConsensusStrategy
        Aggregation strategy to use.  Default: WEIGHTED_VOTE.
    quorum_fraction : float
        For QUORUM strategy — fraction of active voters the winner needs.
        Default 0.5 (simple majority).
    confidence_threshold : float
        For CONFIDENCE_THRESHOLD strategy — minimum average confidence
        the winner must have.  Default 0.7.
    escalation_threshold : float
        If the winning side's share of total weight is below this value,
        escalate to human rather than deadlocking silently.  Default 0.4.
    min_votes : int
        Minimum number of non-abstaining votes required to proceed.
        Default 2.
    """

    def __init__(
        self,
        strategy:               ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTE,
        quorum_fraction:        float = 0.5,
        confidence_threshold:   float = 0.7,
        escalation_threshold:   float = 0.4,
        min_votes:              int   = 2,
    ) -> None:
        self.strategy             = strategy
        self.quorum_fraction      = quorum_fraction
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        self.min_votes            = min_votes
        self._history:  List[ConsensusResult] = []

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        votes:        List[Vote],
        round_id:     Optional[str] = None,
        trust_scores: Optional[Dict[str, float]] = None,
        metadata:     Optional[Dict[str, Any]] = None,
    ) -> ConsensusResult:
        """
        Execute a consensus round.

        Parameters
        ----------
        votes : list[Vote]
            All votes cast for this round.
        round_id : str, optional
            Identifier for the round.  Auto-generated if omitted.
        trust_scores : dict, optional
            ``{agent_id: 0.0–1.0}`` trust weights.  Defaults to 1.0 for
            all agents (unweighted) when not provided.
        metadata : dict, optional
            Arbitrary metadata attached to the result.

        Returns
        -------
        ConsensusResult
        """
        round_id     = round_id or f"round-{uuid.uuid4().hex[:8]}"
        trust_scores = trust_scores or {}
        metadata     = metadata or {}

        logger.info(
            "ConsensusProtocol.run: round=%s, votes=%d, strategy=%s",
            round_id, len(votes), self.strategy.value,
        )

        active_votes   = [v for v in votes if not v.abstain]
        abstentions    = [v.agent_id for v in votes if v.abstain]

        # Guard: need enough real votes
        if len(active_votes) < self.min_votes:
            return self._insufficient(round_id, votes, abstentions, trust_scores, metadata)

        # Dispatch to strategy
        dispatch = {
            ConsensusStrategy.WEIGHTED_VOTE:        self._weighted_vote,
            ConsensusStrategy.QUORUM:               self._quorum,
            ConsensusStrategy.UNANIMOUS:            self._unanimous,
            ConsensusStrategy.CONFIDENCE_THRESHOLD: self._confidence_threshold,
        }
        result = dispatch[self.strategy](round_id, active_votes, abstentions, trust_scores, metadata)
        self._history.append(result)
        logger.info("Round %s → %s", round_id, result.outcome.value)
        return result

    # ── strategies ────────────────────────────────────────────────────────────

    def _weighted_vote(
        self,
        round_id:     str,
        votes:        List[Vote],
        abstentions:  List[str],
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        """Winner = highest (confidence × trust) weighted sum."""
        weights, dissent = self._tally(votes, trust_scores)
        total  = sum(weights.values())
        winner = max(weights, key=weights.get)
        w_frac = weights[winner] / total if total > 0 else 0.0

        # Need strict majority (>50%) or escalate
        if w_frac > 0.5:
            outcome = ConsensusOutcome.ACCEPTED
        elif w_frac >= self.escalation_threshold:
            outcome = ConsensusOutcome.DEADLOCK
        else:
            outcome = ConsensusOutcome.ESCALATE_TO_HUMAN

        return self._build_result(
            round_id, outcome,
            winner if outcome == ConsensusOutcome.ACCEPTED else None,
            weights, dissent, abstentions, total, trust_scores, metadata,
        )

    def _quorum(
        self,
        round_id:     str,
        votes:        List[Vote],
        abstentions:  List[str],
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        """Winner needs ≥ quorum_fraction of votes (by count, not weight)."""
        vote_counts: Dict[str, int] = defaultdict(int)
        dissent:     Dict[str, List[str]] = defaultdict(list)
        for v in votes:
            vote_counts[v.decision] += 1
            dissent[v.decision].append(v.agent_id)

        total_votes = len(votes)
        winner      = max(vote_counts, key=vote_counts.get)
        winner_frac = vote_counts[winner] / total_votes if total_votes > 0 else 0.0

        # For weight breakdown (needed by ConsensusResult)
        weights, _ = self._tally(votes, trust_scores)
        total_w     = sum(weights.values())

        if winner_frac >= self.quorum_fraction:
            outcome = ConsensusOutcome.ACCEPTED
        elif winner_frac >= self.escalation_threshold:
            outcome = ConsensusOutcome.DEADLOCK
        else:
            outcome = ConsensusOutcome.ESCALATE_TO_HUMAN

        return self._build_result(
            round_id, outcome,
            winner if outcome == ConsensusOutcome.ACCEPTED else None,
            weights, dissent, abstentions, total_w, trust_scores, metadata,
        )

    def _unanimous(
        self,
        round_id:     str,
        votes:        List[Vote],
        abstentions:  List[str],
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        """All active voters must agree."""
        decisions = {v.decision for v in votes}
        weights, dissent = self._tally(votes, trust_scores)
        total = sum(weights.values())

        if len(decisions) == 1:
            winner  = decisions.pop()
            outcome = ConsensusOutcome.ACCEPTED
        elif len(decisions) == 2:
            outcome = ConsensusOutcome.ESCALATE_TO_HUMAN
            winner  = None
        else:
            outcome = ConsensusOutcome.DEADLOCK
            winner  = None

        return self._build_result(
            round_id, outcome, winner,
            weights, dissent, abstentions, total, trust_scores, metadata,
        )

    def _confidence_threshold(
        self,
        round_id:     str,
        votes:        List[Vote],
        abstentions:  List[str],
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        """
        Winner = highest weighted decision AND its average voter
        confidence must exceed ``self.confidence_threshold``.
        """
        weights, dissent = self._tally(votes, trust_scores)
        total  = sum(weights.values())
        winner = max(weights, key=weights.get)

        # Average confidence of agents who voted for the winner
        winner_votes = [v for v in votes if v.decision == winner]
        avg_conf = (
            sum(v.confidence for v in winner_votes) / len(winner_votes)
            if winner_votes else 0.0
        )

        if avg_conf >= self.confidence_threshold:
            outcome = ConsensusOutcome.ACCEPTED
        elif avg_conf >= self.escalation_threshold:
            outcome = ConsensusOutcome.DEADLOCK
        else:
            outcome = ConsensusOutcome.ESCALATE_TO_HUMAN

        return self._build_result(
            round_id, outcome,
            winner if outcome == ConsensusOutcome.ACCEPTED else None,
            weights, dissent, abstentions, total, trust_scores, metadata,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tally(
        votes:        List[Vote],
        trust_scores: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, List[str]]]:
        """Return (decision → total_weight, decision → [agent_ids])."""
        weights: Dict[str, float]      = defaultdict(float)
        dissent: Dict[str, List[str]]  = defaultdict(list)
        for v in votes:
            ts = trust_scores.get(v.agent_id, 1.0)
            weights[v.decision] += v.effective_weight(ts)
            dissent[v.decision].append(v.agent_id)
        return dict(weights), dict(dissent)

    def _insufficient(
        self,
        round_id:     str,
        votes:        List[Vote],
        abstentions:  List[str],
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        weights, dissent = self._tally([v for v in votes if not v.abstain], trust_scores)
        return ConsensusResult(
            round_id         = round_id,
            outcome          = ConsensusOutcome.INSUFFICIENT_VOTES,
            winning_decision = None,
            total_weight     = sum(weights.values()),
            winning_weight   = 0.0,
            vote_breakdown   = weights,
            dissent_map      = dissent,
            abstentions      = abstentions,
            confidence_score = 0.0,
            strategy_used    = self.strategy,
            decided_at       = datetime.utcnow(),
            metadata         = metadata,
        )

    def _build_result(
        self,
        round_id:     str,
        outcome:      ConsensusOutcome,
        winner:       Optional[str],
        weights:      Dict[str, float],
        dissent:      Dict[str, List[str]],
        abstentions:  List[str],
        total:        float,
        trust_scores: Dict[str, float],
        metadata:     Dict[str, Any],
    ) -> ConsensusResult:
        w_weight = weights.get(winner, 0.0) if winner else 0.0
        # Confidence score = how decisive the consensus is (0 = tie, 1 = unanimous)
        conf_score = (w_weight / total) if total > 0 else 0.0

        return ConsensusResult(
            round_id         = round_id,
            outcome          = outcome,
            winning_decision = winner,
            total_weight     = round(total, 6),
            winning_weight   = round(w_weight, 6),
            vote_breakdown   = {k: round(v, 6) for k, v in weights.items()},
            dissent_map      = dissent,
            abstentions      = abstentions,
            confidence_score = round(conf_score, 4),
            strategy_used    = self.strategy,
            decided_at       = datetime.utcnow(),
            metadata         = metadata,
        )

    # ── history & observability ───────────────────────────────────────────────

    def history(self) -> List[ConsensusResult]:
        """All consensus results from this protocol instance."""
        return list(self._history)

    def acceptance_rate(self) -> float:
        """Fraction of rounds that reached ACCEPTED."""
        if not self._history:
            return 0.0
        accepted = sum(1 for r in self._history if r.reached)
        return round(accepted / len(self._history), 4)

    def escalation_rate(self) -> float:
        """Fraction of rounds that escalated to human."""
        if not self._history:
            return 0.0
        escalated = sum(1 for r in self._history if r.needs_human)
        return round(escalated / len(self._history), 4)

    def metrics(self) -> Dict[str, Any]:
        return {
            "strategy":         self.strategy.value,
            "total_rounds":     len(self._history),
            "acceptance_rate":  self.acceptance_rate(),
            "escalation_rate":  self.escalation_rate(),
            "avg_confidence":   round(
                sum(r.confidence_score for r in self._history) / len(self._history), 4
            ) if self._history else 0.0,
            "outcome_breakdown": {
                o.value: sum(1 for r in self._history if r.outcome == o)
                for o in ConsensusOutcome
            },
        }

    def __repr__(self) -> str:
        return (
            f"ConsensusProtocol(strategy={self.strategy.value!r}, "
            f"rounds={len(self._history)}, "
            f"acceptance_rate={self.acceptance_rate():.0%})"
        )
