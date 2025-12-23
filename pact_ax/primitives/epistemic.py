"""
PACT-AX Epistemic Primitives
Humility as foundational substrate for agent coordination
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Set, Dict, List
from enum import Enum
from datetime import datetime
import uuid


class ConfidenceLevel(Enum):
    """
    Calibrated confidence levels.
    These represent actual epistemic states, not aspirational claims.
    """
    CERTAIN = 0.95      # Verified, tested, proven
    CONFIDENT = 0.80    # Well-understood, reliable
    MODERATE = 0.60     # Partial knowledge, some uncertainty
    LOW = 0.40          # Limited knowledge, significant gaps
    UNKNOWN = 0.20      # Don't know, must defer
    
    def __str__(self):
        return self.name.lower()
    
    def is_sufficient_for(self, required_confidence: float) -> bool:
        """Check if this confidence meets requirement"""
        return self.value >= required_confidence


@dataclass
class KnowledgeBoundary:
    """
    Explicit representation of what an agent knows it can/cannot do.
    Humility encoded as first-class data structure.
    """
    domain: str
    proficiency: ConfidenceLevel
    known_limits: Set[str] = field(default_factory=set)
    known_capabilities: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def can_handle(self, topic: str) -> bool:
        """Check if topic is within capability boundary"""
        return topic in self.known_capabilities
    
    def should_decline(self, topic: str) -> bool:
        """Check if topic is explicitly outside boundary"""
        return topic in self.known_limits
    
    def update_boundary(self, learned_capability: str = None, learned_limit: str = None):
        """Update knowledge boundaries based on experience"""
        if learned_capability:
            self.known_capabilities.add(learned_capability)
        if learned_limit:
            self.known_limits.add(learned_limit)
        self.last_updated = datetime.now()


@dataclass
class DelegationMap:
    """
    Maps topics/domains to more capable agents.
    "I don't know, but I know who does" as primitive.
    """
    mappings: Dict[str, str] = field(default_factory=dict)  # topic -> agent_id
    rationale: Dict[str, str] = field(default_factory=dict)  # topic -> why delegate
    
    def add_delegation(self, topic: str, agent_id: str, reason: str):
        """Record where to delegate for this topic"""
        self.mappings[topic] = agent_id
        self.rationale[topic] = reason
    
    def get_delegate(self, topic: str) -> Optional[str]:
        """Find appropriate agent for topic"""
        return self.mappings.get(topic)
    
    def get_reason(self, topic: str) -> Optional[str]:
        """Why should this be delegated"""
        return self.rationale.get(topic)


@dataclass
class EpistemicState:
    """
    Core humility primitive.
    Knowledge bundled with confidence, uncertainty, and boundaries.
    Cannot express knowledge without expressing certainty.
    """
    value: Any  # The actual knowledge/answer
    confidence: ConfidenceLevel  # How certain about this
    uncertainty_reason: Optional[str] = None  # Why uncertain (if applicable)
    boundary: Optional[KnowledgeBoundary] = None  # Agent's capability limits
    delegation_map: Optional[DelegationMap] = None  # Where to defer
    source: str = "unknown"  # Provenance of knowledge
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def should_defer(self, query_topic: str, required_confidence: float = 0.6) -> bool:
        """
        Determine if this agent should defer to another.
        Humility check built into state assessment.
        """
        # Confidence too low
        if not self.confidence.is_sufficient_for(required_confidence):
            return True
        
        # Explicitly outside boundary
        if self.boundary and self.boundary.should_decline(query_topic):
            return True
        
        # Not within known capabilities
        if self.boundary and not self.boundary.can_handle(query_topic):
            return True
        
        return False
    
    def get_delegate(self, query_topic: str) -> Optional[str]:
        """Find appropriate agent to delegate to"""
        if self.delegation_map:
            return self.delegation_map.get_delegate(query_topic)
        return None
    
    def to_dict(self) -> Dict:
        """Serialize for transfer"""
        return {
            'id': self.id,
            'value': self.value,
            'confidence': self.confidence.name,
            'uncertainty_reason': self.uncertainty_reason,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'should_defer': self.should_defer("")  # General deferral state
        }


class UnknownResponse:
    """
    First-class 'I don't know' response.
    Not an error - a valid epistemic state.
    """
    def __init__(
        self, 
        reason: str,
        suggested_delegate: Optional[str] = None,
        delegation_reason: Optional[str] = None,
        can_learn: bool = True
    ):
        self.reason = reason
        self.suggested_delegate = suggested_delegate
        self.delegation_reason = delegation_reason
        self.can_learn = can_learn  # Can this agent learn to handle this?
        self.timestamp = datetime.now()
        self.id = str(uuid.uuid4())
    
    def to_epistemic_state(self) -> EpistemicState:
        """Convert to EpistemicState for consistency"""
        return EpistemicState(
            value=None,
            confidence=ConfidenceLevel.UNKNOWN,
            uncertainty_reason=self.reason,
            source="explicit_unknown"
        )
    
    def __repr__(self):
        delegate_info = f" → {self.suggested_delegate}" if self.suggested_delegate else ""
        return f"UnknownResponse(reason='{self.reason}'{delegate_info})"


@dataclass
class BeliefUpdate:
    """
    Represents change in epistemic state.
    Learning encoded as state transition.
    """
    previous_state: EpistemicState
    new_state: EpistemicState
    reason: str  # Why belief changed
    evidence: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def confidence_increased(self) -> bool:
        """Did we become more confident?"""
        return self.new_state.confidence.value > self.previous_state.confidence.value
    
    def confidence_decreased(self) -> bool:
        """Did we become less confident (more humble)?"""
        return self.new_state.confidence.value < self.previous_state.confidence.value
    
    def was_wrong(self) -> bool:
        """Did we have to change our answer (not just confidence)?"""
        return self.previous_state.value != self.new_state.value


# Helper functions for common epistemic operations

def merge_epistemic_states(
    states: List[EpistemicState],
    strategy: str = "confidence_weighted"
) -> EpistemicState:
    """
    Combine multiple epistemic states.
    Preserves uncertainty through aggregation.
    """
    if not states:
        return UnknownResponse("No states to merge").to_epistemic_state()
    
    if len(states) == 1:
        return states[0]
    
    if strategy == "confidence_weighted":
        # Weight by confidence, preserve uncertainty
        total_confidence = sum(s.confidence.value for s in states)
        weighted_value = None  # Implement domain-specific merging
        
        # Take most confident state's value for now
        best_state = max(states, key=lambda s: s.confidence.value)
        
        # But adjust confidence based on agreement
        confidence_values = [s.confidence.value for s in states]
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Map back to ConfidenceLevel
        for level in ConfidenceLevel:
            if level.value <= avg_confidence:
                merged_confidence = level
        
        return EpistemicState(
            value=best_state.value,
            confidence=merged_confidence,
            uncertainty_reason="Merged from multiple sources with varying confidence",
            source=f"merged_{len(states)}_states"
        )
    
    raise ValueError(f"Unknown merge strategy: {strategy}")


def calibrate_confidence(
    predicted_confidence: float,
    actual_outcome: bool
) -> float:
    """
    Adjust confidence based on whether prediction was correct.
    Learning to be appropriately humble.
    """
    if actual_outcome:
        # Was correct - can maintain or slightly increase confidence
        return min(predicted_confidence * 1.1, 1.0)
    else:
        # Was wrong - should decrease confidence significantly
        return max(predicted_confidence * 0.7, 0.1)
