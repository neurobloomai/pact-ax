"""
PACT-AX Humility-Aware Coordination
Coordination that assumes and requires epistemic honesty
"""

from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
from ..primitives.epistemic import (
    EpistemicState,
    ConfidenceLevel,
    UnknownResponse,
    KnowledgeBoundary,
    DelegationMap
)


class Agent(Protocol):
    """Protocol defining what an agent must implement"""
    id: str
    boundaries: List[KnowledgeBoundary]
    
    def assess_capability(self, query: 'Query') -> EpistemicState:
        """Assess if agent can handle query"""
        ...
    
    def handle(self, query: 'Query') -> EpistemicState:
        """Actually handle the query"""
        ...


@dataclass
class Query:
    """Represents a request to the system"""
    content: str
    domain: str
    required_confidence: float = 0.6
    context: Dict = None


class HumilityAwareCoordinator:
    """
    Routes queries to agents based on epistemic capability.
    Coordination emerges from humility substrate.
    """
    
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.routing_history: List[Dict] = []
    
    def route_query(self, query: Query) -> Agent:
        """
        Find most capable agent for query.
        Based on epistemic assessment, not just domain matching.
        """
        best_agent = None
        highest_capability = 0.0
        
        for agent_id, agent in self.agents.items():
            # Ask agent to assess its capability
            capability = agent.assess_capability(query)
            
            # Skip if agent knows it should defer
            if capability.should_defer(query.domain, query.required_confidence):
                continue
            
            # Track best match
            if capability.confidence.value > highest_capability:
                highest_capability = capability.confidence.value
                best_agent = agent
        
        # Log routing decision
        self.routing_history.append({
            'query': query.content,
            'routed_to': best_agent.id if best_agent else None,
            'confidence': highest_capability
        })
        
        return best_agent
    
    def should_escalate(self, responses: List[EpistemicState], threshold: float = 0.7) -> bool:
        """
        Determine if responses indicate need for escalation.
        System-level humility check.
        """
        if not responses:
            return True
        
        # Check if any response meets threshold
        max_confidence = max(r.confidence.value for r in responses)
        if max_confidence < threshold:
            return True
        
        # Check for significant disagreement
        if len(responses) > 1:
            confidence_range = max(r.confidence.value for r in responses) - min(r.confidence.value for r in responses)
            if confidence_range > 0.4:  # High disagreement
                return True
        
        return False
    
    def aggregate_responses(
        self,
        responses: List[EpistemicState],
        strategy: str = "confidence_weighted"
    ) -> EpistemicState:
        """
        Combine responses from multiple agents.
        Preserves uncertainty through aggregation.
        """
        from ..primitives.epistemic import merge_epistemic_states
        return merge_epistemic_states(responses, strategy)


class DelegationChain:
    """
    Tracks delegation path when agents defer to each other.
    Makes humility visible and trackable.
    """
    
    def __init__(self):
        self.chain: List[Dict] = []
    
    def add_delegation(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        query: Query
    ):
        """Record a delegation step"""
        self.chain.append({
            'from': from_agent,
            'to': to_agent,
            'reason': reason,
            'query': query.content,
            'domain': query.domain
        })
    
    def get_final_handler(self) -> Optional[str]:
        """Who ultimately handled the query"""
        return self.chain[-1]['to'] if self.chain else None
    
    def delegation_count(self) -> int:
        """How many times was query delegated"""
        return len(self.chain)
    
    def __repr__(self):
        if not self.chain:
            return "DelegationChain(empty)"
        
        path = " → ".join([self.chain[0]['from']] + [step['to'] for step in self.chain])
        return f"DelegationChain({path})"
