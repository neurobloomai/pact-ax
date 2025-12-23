"""
PACT-AX Policy Alignment with Epistemic Humility
Policy decisions that respect knowledge boundaries and uncertainty
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..primitives.epistemic import (
    EpistemicState,
    ConfidenceLevel,
    KnowledgeBoundary,
    UnknownResponse
)


class PolicyConflictResolution(Enum):
    """How to resolve conflicts between policies"""
    DEFER_TO_MORE_CONFIDENT = "defer_confident"  # Let more confident agent decide
    DEFER_TO_SPECIALIST = "defer_specialist"     # Let domain expert decide
    ESCALATE_TO_HUMAN = "escalate_human"         # Human decides
    CONSENSUS_REQUIRED = "consensus"             # All must agree
    MOST_CONSERVATIVE = "conservative"           # Choose safest option


@dataclass
class PolicyDecision:
    """
    A policy decision with epistemic state.
    Can't make policy without expressing confidence.
    """
    decision: str
    confidence: ConfidenceLevel
    reasoning: str
    agent_id: str
    domain: str
    alternatives_considered: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_confident_enough(self, threshold: float = 0.7) -> bool:
        """Check if confidence meets threshold for action"""
        return self.confidence.value >= threshold
    
    def should_seek_consensus(self) -> bool:
        """Determine if decision needs validation from others"""
        return self.confidence.value < 0.8 or len(self.uncertainty_factors) > 2


@dataclass
class PolicyConstraint:
    """
    Constraints that policies must satisfy.
    Can include epistemic requirements.
    """
    name: str
    description: str
    min_confidence: float = 0.6  # Minimum confidence to proceed
    requires_specialist: bool = False  # Must have domain expert
    safety_critical: bool = False  # High stakes decision
    domains: Set[str] = field(default_factory=set)
    
    def is_satisfied(self, decision: PolicyDecision, agent_boundaries: List[KnowledgeBoundary]) -> bool:
        """Check if decision satisfies this constraint"""
        # Confidence check
        if decision.confidence.value < self.min_confidence:
            return False
        
        # Specialist requirement
        if self.requires_specialist:
            has_specialist = any(
                decision.domain in b.known_capabilities 
                for b in agent_boundaries
            )
            if not has_specialist:
                return False
        
        # Domain check
        if self.domains and decision.domain not in self.domains:
            return False
        
        return True


class PolicyAlignmentManager:
    """
    Manages policy decisions across agents with epistemic awareness.
    Ensures agents don't overreach their knowledge boundaries.
    """
    
    def __init__(self):
        self.constraints: Dict[str, PolicyConstraint] = {}
        self.decision_history: List[PolicyDecision] = []
        self.conflict_resolutions: List[Dict] = []
    
    def add_constraint(self, constraint: PolicyConstraint):
        """Add a policy constraint that must be satisfied"""
        self.constraints[constraint.name] = constraint
    
    def evaluate_decision(
        self,
        decision: PolicyDecision,
        agent_boundaries: List[KnowledgeBoundary]
    ) -> tuple[bool, List[str]]:
        """
        Evaluate if decision satisfies all constraints.
        Returns (is_valid, reasons_if_invalid)
        """
        violations = []
        
        for constraint_name, constraint in self.constraints.items():
            if not constraint.is_satisfied(decision, agent_boundaries):
                violations.append(f"Violates constraint: {constraint_name}")
        
        # Additional epistemic checks
        if decision.confidence == ConfidenceLevel.UNKNOWN:
            violations.append("Cannot make policy decision with UNKNOWN confidence")
        
        if decision.confidence == ConfidenceLevel.LOW and not decision.should_seek_consensus():
            violations.append("Low confidence decision should seek consensus")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def resolve_conflict(
        self,
        decisions: List[PolicyDecision],
        resolution_strategy: PolicyConflictResolution = PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT
    ) -> PolicyDecision:
        """
        Resolve conflicts between multiple policy decisions.
        Humility guides resolution strategy.
        """
        if not decisions:
            raise ValueError("No decisions to resolve")
        
        if len(decisions) == 1:
            return decisions[0]
        
        # Log conflict
        self.conflict_resolutions.append({
            'decisions': [d.decision for d in decisions],
            'agents': [d.agent_id for d in decisions],
            'strategy': resolution_strategy.value,
            'timestamp': datetime.now()
        })
        
        if resolution_strategy == PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT:
            # Choose decision from most confident agent
            return max(decisions, key=lambda d: d.confidence.value)
        
        elif resolution_strategy == PolicyConflictResolution.DEFER_TO_SPECIALIST:
            # Choose decision from agent with best domain fit
            # (assumes decisions have agent boundaries accessible)
            return max(decisions, key=lambda d: d.confidence.value)
        
        elif resolution_strategy == PolicyConflictResolution.MOST_CONSERVATIVE:
            # Choose the most cautious/safe decision
            # Lower confidence often means more conservative
            return min(decisions, key=lambda d: d.confidence.value)
        
        elif resolution_strategy == PolicyConflictResolution.ESCALATE_TO_HUMAN:
            # Return special decision indicating human needed
            return PolicyDecision(
                decision="ESCALATE_TO_HUMAN",
                confidence=ConfidenceLevel.UNKNOWN,
                reasoning=f"Conflicting decisions from {len(decisions)} agents require human judgment",
                agent_id="system",
                domain="meta"
            )
        
        elif resolution_strategy == PolicyConflictResolution.CONSENSUS_REQUIRED:
            # Check if all decisions agree
            unique_decisions = set(d.decision for d in decisions)
            if len(unique_decisions) == 1:
                # Consensus achieved, return with averaged confidence
                avg_confidence_value = sum(d.confidence.value for d in decisions) / len(decisions)
                consensus_confidence = self._map_to_confidence_level(avg_confidence_value)
                
                return PolicyDecision(
                    decision=decisions[0].decision,
                    confidence=consensus_confidence,
                    reasoning=f"Consensus from {len(decisions)} agents",
                    agent_id="consensus",
                    domain=decisions[0].domain
                )
            else:
                # No consensus, escalate
                return PolicyDecision(
                    decision="NO_CONSENSUS",
                    confidence=ConfidenceLevel.UNKNOWN,
                    reasoning=f"No consensus among {len(decisions)} agents: {unique_decisions}",
                    agent_id="system",
                    domain="meta"
                )
        
        raise ValueError(f"Unknown resolution strategy: {resolution_strategy}")
    
    def align_policies(
        self,
        decisions: List[PolicyDecision],
        agent_boundaries: Dict[str, List[KnowledgeBoundary]]
    ) -> PolicyDecision:
        """
        Align multiple policy decisions into coherent outcome.
        Respects epistemic boundaries throughout.
        """
        # Filter out invalid decisions
        valid_decisions = []
        for decision in decisions:
            boundaries = agent_boundaries.get(decision.agent_id, [])
            is_valid, violations = self.evaluate_decision(decision, boundaries)
            
            if is_valid:
                valid_decisions.append(decision)
            else:
                # Log why decision was rejected
                print(f"Rejected decision from {decision.agent_id}: {violations}")
        
        if not valid_decisions:
            return PolicyDecision(
                decision="NO_VALID_DECISIONS",
                confidence=ConfidenceLevel.UNKNOWN,
                reasoning="All proposed decisions violated constraints",
                agent_id="system",
                domain="meta"
            )
        
        # Resolve conflicts among valid decisions
        resolution_strategy = self._choose_resolution_strategy(valid_decisions)
        final_decision = self.resolve_conflict(valid_decisions, resolution_strategy)
        
        # Record in history
        self.decision_history.append(final_decision)
        
        return final_decision
    
    def _choose_resolution_strategy(self, decisions: List[PolicyDecision]) -> PolicyConflictResolution:
        """
        Choose appropriate conflict resolution strategy.
        Based on epistemic states and decision characteristics.
        """
        # If any decision is safety-critical, be conservative
        if any(len(d.uncertainty_factors) > 3 for d in decisions):
            return PolicyConflictResolution.MOST_CONSERVATIVE
        
        # If confidence varies widely, defer to most confident
        confidence_range = max(d.confidence.value for d in decisions) - min(d.confidence.value for d in decisions)
        if confidence_range > 0.3:
            return PolicyConflictResolution.DEFER_TO_MORE_CONFIDENT
        
        # If confidence is uniformly low, escalate
        max_confidence = max(d.confidence.value for d in decisions)
        if max_confidence < 0.5:
            return PolicyConflictResolution.ESCALATE_TO_HUMAN
        
        # Default to consensus
        return PolicyConflictResolution.CONSENSUS_REQUIRED
    
    def _map_to_confidence_level(self, value: float) -> ConfidenceLevel:
        """Map float confidence value to ConfidenceLevel enum"""
        if value >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif value >= 0.80:
            return ConfidenceLevel.CONFIDENT
        elif value >= 0.60:
            return ConfidenceLevel.MODERATE
        elif value >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    def get_policy_alignment_metrics(self) -> Dict:
        """Analytics on policy alignment quality"""
        if not self.decision_history:
            return {"message": "No decisions recorded"}
        
        return {
            'total_decisions': len(self.decision_history),
            'conflicts_resolved': len(self.conflict_resolutions),
            'avg_confidence': sum(d.confidence.value for d in self.decision_history) / len(self.decision_history),
            'escalations': sum(1 for d in self.decision_history if d.decision in ["ESCALATE_TO_HUMAN", "NO_CONSENSUS"]),
            'high_confidence_rate': sum(1 for d in self.decision_history if d.confidence.value >= 0.8) / len(self.decision_history)
        }


class PolicyLearning:
    """
    Learn from policy decisions over time.
    Humility means updating based on outcomes.
    """
    
    def __init__(self):
        self.outcomes: List[Dict] = []
    
    def record_outcome(
        self,
        decision: PolicyDecision,
        actual_outcome: str,
        was_correct: bool,
        feedback: Optional[str] = None
    ):
        """
        Record outcome of a policy decision.
        Used to calibrate future confidence.
        """
        self.outcomes.append({
            'decision': decision.decision,
            'predicted_confidence': decision.confidence.value,
            'actual_outcome': actual_outcome,
            'was_correct': was_correct,
            'agent_id': decision.agent_id,
            'domain': decision.domain,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
    
    def get_agent_calibration(self, agent_id: str) -> Dict:
        """
        Analyze how well-calibrated an agent's confidence is.
        Humility metric: does confidence match reality?
        """
        agent_outcomes = [o for o in self.outcomes if o['agent_id'] == agent_id]
        
        if not agent_outcomes:
            return {"message": f"No outcomes for agent {agent_id}"}
        
        # Calculate calibration
        correct_count = sum(1 for o in agent_outcomes if o['was_correct'])
        avg_predicted_confidence = sum(o['predicted_confidence'] for o in agent_outcomes) / len(agent_outcomes)
        actual_accuracy = correct_count / len(agent_outcomes)
        
        # Well-calibrated means predicted confidence ≈ actual accuracy
        calibration_error = abs(avg_predicted_confidence - actual_accuracy)
        
        return {
            'agent_id': agent_id,
            'total_decisions': len(agent_outcomes),
            'correct_decisions': correct_count,
            'accuracy': actual_accuracy,
            'avg_predicted_confidence': avg_predicted_confidence,
            'calibration_error': calibration_error,
            'is_well_calibrated': calibration_error < 0.15,  # Within 15%
            'tendency': 'overconfident' if avg_predicted_confidence > actual_accuracy else 'underconfident'
        }
    
    def suggest_confidence_adjustment(self, agent_id: str, domain: str) -> float:
        """
        Suggest confidence adjustment for agent in domain.
        Learning to be more humble (or less humble if too cautious).
        """
        calibration = self.get_agent_calibration(agent_id)
        
        if 'calibration_error' not in calibration:
            return 1.0  # No adjustment
        
        if calibration['tendency'] == 'overconfident':
            # Should be more humble
            return 0.85  # Reduce confidence by 15%
        elif calibration['tendency'] == 'underconfident':
            # Can be more confident
            return 1.1  # Increase confidence by 10%
        
        return 1.0  # Well-calibrated, no adjustment
