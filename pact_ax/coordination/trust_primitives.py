"""
PACT-AX Trust Primitives
Trust emerges from epistemic honesty, not claimed authority
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..primitives.epistemic import EpistemicState, ConfidenceLevel, BeliefUpdate


class TrustDimension(Enum):
    """Different dimensions of trust between agents"""
    COMPETENCE = "competence"        # Can they do the task well?
    HONESTY = "honesty"              # Do they report uncertainty accurately?
    RELIABILITY = "reliability"       # Do they follow through?
    CALIBRATION = "calibration"      # Is their confidence well-calibrated?


@dataclass
class TrustScore:
    """
    Trust score between two agents.
    Based on observed epistemic honesty.
    """
    truster_id: str  # Who is doing the trusting
    trustee_id: str  # Who is being trusted
    
    # Trust dimensions
    competence: float = 0.5        # 0-1 scale
    honesty: float = 0.5           # 0-1 scale  
    reliability: float = 0.5       # 0-1 scale
    calibration: float = 0.5       # 0-1 scale
    
    # Supporting data
    interactions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    trust_history: List[float] = field(default_factory=list)
    
    def overall_trust(self) -> float:
        """Compute overall trust score"""
        # Weight dimensions (can be adjusted)
        weights = {
            'competence': 0.3,
            'honesty': 0.3,
            'calibration': 0.25,
            'reliability': 0.15
        }
        
        return (
            weights['competence'] * self.competence +
            weights['honesty'] * self.honesty +
            weights['calibration'] * self.calibration +
            weights['reliability'] * self.reliability
        )
    
    def update_from_interaction(
        self,
        outcome: 'TrustInteractionOutcome'
    ):
        """Update trust based on interaction outcome"""
        # Update competence
        if outcome.was_competent is not None:
            self.competence = self._update_score(self.competence, outcome.was_competent)
        
        # Update honesty (did they accurately report uncertainty?)
        if outcome.was_honest is not None:
            self.honesty = self._update_score(self.honesty, outcome.was_honest)
        
        # Update reliability
        if outcome.was_reliable is not None:
            self.reliability = self._update_score(self.reliability, outcome.was_reliable)
        
        # Update calibration (confidence matched reality?)
        if outcome.was_calibrated is not None:
            self.calibration = self._update_score(self.calibration, outcome.was_calibrated)
        
        self.interactions += 1
        self.last_updated = datetime.now()
        self.trust_history.append(self.overall_trust())
    
    def _update_score(self, current: float, positive_outcome: bool, learning_rate: float = 0.1) -> float:
        """Update score with exponential moving average"""
        target = 1.0 if positive_outcome else 0.0
        return current + learning_rate * (target - current)


@dataclass
class TrustInteractionOutcome:
    """Result of an interaction that affects trust"""
    was_competent: Optional[bool] = None      # Did they succeed?
    was_honest: Optional[bool] = None         # Did they accurately report uncertainty?
    was_reliable: Optional[bool] = None       # Did they follow through?
    was_calibrated: Optional[bool] = None     # Was confidence accurate?
    notes: str = ""


class TrustNetwork:
    """
    Network of trust relationships between agents.
    Trust based on epistemic honesty, not authority.
    """
    
    def __init__(self):
        self.trust_scores: Dict[tuple[str, str], TrustScore] = {}
        self.interaction_log: List[Dict] = []
    
    def get_trust(self, truster_id: str, trustee_id: str) -> TrustScore:
        """Get trust score between two agents"""
        key = (truster_id, trustee_id)
        
        if key not in self.trust_scores:
            # Initialize new trust relationship
            self.trust_scores[key] = TrustScore(
                truster_id=truster_id,
                trustee_id=trustee_id
            )
        
        return self.trust_scores[key]
    
    def record_interaction(
        self,
        truster_id: str,
        trustee_id: str,
        epistemic_state: EpistemicState,
        actual_outcome: bool,
        outcome_details: Optional[TrustInteractionOutcome] = None
    ):
        """
        Record interaction and update trust.
        Key: Did agent's confidence match reality?
        """
        trust_score = self.get_trust(truster_id, trustee_id)
        
        # Evaluate epistemic honesty
        predicted_confidence = epistemic_state.confidence.value
        
        # Was agent well-calibrated?
        # High confidence + success = good
        # Low confidence + failure = good (honest about uncertainty)
        # High confidence + failure = bad (overconfident)
        # Low confidence + success = neutral (too cautious)
        
        was_calibrated = self._assess_calibration(predicted_confidence, actual_outcome)
        was_honest = self._assess_honesty(epistemic_state, actual_outcome)
        
        # Use provided outcome or infer from calibration
        if outcome_details is None:
            outcome_details = TrustInteractionOutcome(
                was_competent=actual_outcome,
                was_honest=was_honest,
                was_calibrated=was_calibrated,
                was_reliable=True  # Assume reliable unless specified
            )
        
        # Update trust
        trust_score.update_from_interaction(outcome_details)
        
        # Log interaction
        self.interaction_log.append({
            'truster': truster_id,
            'trustee': trustee_id,
            'predicted_confidence': predicted_confidence,
            'actual_outcome': actual_outcome,
            'was_calibrated': was_calibrated,
            'trust_after': trust_score.overall_trust(),
            'timestamp': datetime.now()
        })
    
    def _assess_calibration(self, predicted_confidence: float, actual_outcome: bool) -> bool:
        """
        Assess if confidence was well-calibrated.
        This is core to epistemic trust.
        """
        if actual_outcome:
            # Success - was confidence appropriate?
            # High confidence on success = well-calibrated
            return predicted_confidence >= 0.7
        else:
            # Failure - was confidence appropriately low?
            # Low confidence on failure = well-calibrated (honest humility)
            return predicted_confidence < 0.6
    
    def _assess_honesty(self, epistemic_state: EpistemicState, actual_outcome: bool) -> bool:
        """
        Assess if agent was honest about uncertainty.
        Did they express doubt when appropriate?
        """
        # If they had low confidence and failed, that's honest
        if not actual_outcome and epistemic_state.confidence.value < 0.6:
            return True
        
        # If they had high confidence and succeeded, that's honest
        if actual_outcome and epistemic_state.confidence.value >= 0.7:
            return True
        
        # If they expressed uncertainty reasons appropriately
        if epistemic_state.uncertainty_reason and epistemic_state.confidence.value < 0.7:
            return True
        
        return False
    
    def recommend_agent(
        self,
        requester_id: str,
        domain: str,
        available_agents: List[str]
    ) -> Optional[str]:
        """
        Recommend most trusted agent for a task.
        Based on past epistemic honesty.
        """
        best_agent = None
        highest_trust = 0.0
        
        for agent_id in available_agents:
            trust = self.get_trust(requester_id, agent_id)
            overall = trust.overall_trust()
            
            if overall > highest_trust:
                highest_trust = overall
                best_agent = agent_id
        
        return best_agent
    
    def get_network_metrics(self) -> Dict:
        """Analytics on trust network health"""
        if not self.trust_scores:
            return {"message": "No trust relationships"}
        
        trust_values = [ts.overall_trust() for ts in self.trust_scores.values()]
        
        return {
            'total_relationships': len(self.trust_scores),
            'avg_trust': sum(trust_values) / len(trust_values),
            'high_trust_count': sum(1 for t in trust_values if t >= 0.7),
            'low_trust_count': sum(1 for t in trust_values if t < 0.4),
            'total_interactions': sum(ts.interactions for ts in self.trust_scores.values()),
            'calibration_scores': [ts.calibration for ts in self.trust_scores.values()],
            'honesty_scores': [ts.honesty for ts in self.trust_scores.values()]
        }


class ReputationSystem:
    """
    Agent reputation based on epistemic track record.
    Good reputation = well-calibrated confidence over time.
    """
    
    def __init__(self, trust_network: TrustNetwork):
        self.trust_network = trust_network
        self.agent_reputations: Dict[str, float] = {}
    
    def compute_reputation(self, agent_id: str) -> float:
        """
        Compute agent's overall reputation.
        Based on how others trust them.
        """
        # Find all trust scores where agent is trustee
        relevant_scores = [
            ts for ts in self.trust_network.trust_scores.values()
            if ts.trustee_id == agent_id
        ]
        
        if not relevant_scores:
            return 0.5  # Neutral reputation
        
        # Weight by number of interactions (more data = more reliable)
        weighted_trust = sum(
            ts.overall_trust() * ts.interactions 
            for ts in relevant_scores
        )
        total_interactions = sum(ts.interactions for ts in relevant_scores)
        
        reputation = weighted_trust / total_interactions if total_interactions > 0 else 0.5
        
        self.agent_reputations[agent_id] = reputation
        return reputation
    
    def get_top_agents(self, n: int = 5) -> List[tuple[str, float]]:
        """Get agents with best reputation"""
        # Compute all reputations
        for agent_id in set(
            ts.trustee_id for ts in self.trust_network.trust_scores.values()
        ):
            self.compute_reputation(agent_id)
        
        # Sort by reputation
        sorted_agents = sorted(
            self.agent_reputations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_agents[:n]
    
    def get_reputation_summary(self) -> Dict:
        """Summary of reputation distribution"""
        if not self.agent_reputations:
            return {"message": "No reputation data"}
        
        reputations = list(self.agent_reputations.values())
        
        return {
            'total_agents': len(reputations),
            'avg_reputation': sum(reputations) / len(reputations),
            'high_reputation': sum(1 for r in reputations if r >= 0.7),
            'low_reputation': sum(1 for r in reputations if r < 0.4),
            'top_agents': self.get_top_agents(3)
        }
