"""
PACT-AX Gossip Protocols with Epistemic Clarity
Information spreads WITH uncertainty preserved
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random

from ..primitives.epistemic import (
    EpistemicState,
    ConfidenceLevel,
    merge_epistemic_states
)


@dataclass
class GossipMessage:
    """
    Message in gossip protocol.
    Includes epistemic state, not just raw information.
    """
    content: EpistemicState
    source_agent: str
    hops: int = 0  # How many times has this been forwarded
    path: List[str] = field(default_factory=list)  # Who has seen this
    timestamp: datetime = field(default_factory=datetime.now)
    
    def degrade_confidence(self, degradation_factor: float = 0.95) -> 'GossipMessage':
        """
        Confidence degrades with forwarding (like whisper down the lane).
        Humility about indirect knowledge.
        """
        # Reduce confidence with each hop
        new_confidence_value = self.content.confidence.value * (degradation_factor ** self.hops)
        
        # Map back to confidence level
        new_confidence = self._map_to_confidence_level(new_confidence_value)
        
        # Create degraded state
        degraded_state = EpistemicState(
            value=self.content.value,
            confidence=new_confidence,
            uncertainty_reason=f"Information passed through {self.hops} agents (indirect knowledge)",
            source=f"gossip_from_{self.source_agent}",
            timestamp=self.content.timestamp
        )
        
        return GossipMessage(
            content=degraded_state,
            source_agent=self.source_agent,
            hops=self.hops,
            path=self.path.copy(),
            timestamp=self.timestamp
        )
    
    def _map_to_confidence_level(self, value: float) -> ConfidenceLevel:
        """Map float to ConfidenceLevel enum"""
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


class GossipClarityProtocol:
    """
    Gossip protocol that preserves epistemic clarity.
    Information spreads, but uncertainty preserved/amplified.
    """
    
    def __init__(
        self,
        agents: Dict[str, 'Agent'],
        max_hops: int = 5,
        confidence_degradation: float = 0.95
    ):
        self.agents = agents
        self.max_hops = max_hops
        self.confidence_degradation = confidence_degradation
        self.message_history: List[GossipMessage] = []
        self.agent_knowledge: Dict[str, List[EpistemicState]] = {
            agent_id: [] for agent_id in agents.keys()
        }
    
    def initiate_gossip(
        self,
        initial_state: EpistemicState,
        source_agent_id: str
    ):
        """
        Start gossip spread from source agent.
        """
        message = GossipMessage(
            content=initial_state,
            source_agent=source_agent_id,
            hops=0,
            path=[source_agent_id]
        )
        
        self._spread_message(message, source_agent_id)
    
    def _spread_message(
        self,
        message: GossipMessage,
        current_agent_id: str
    ):
        """
        Recursively spread message through network.
        """
        # Stop if max hops reached
        if message.hops >= self.max_hops:
            return
        
        # Stop if confidence too degraded
        if message.content.confidence.value < 0.3:
            return
        
        # Record message
        self.message_history.append(message)
        
        # Store knowledge at current agent
        self.agent_knowledge[current_agent_id].append(message.content)
        
        # Select neighbors to forward to (gossip style)
        neighbors = self._get_neighbors(current_agent_id)
        
        # Forward to random subset of neighbors
        forward_count = max(1, len(neighbors) // 2)  # Forward to ~half
        forward_to = random.sample(neighbors, min(forward_count, len(neighbors)))
        
        for neighbor_id in forward_to:
            # Skip if already in path (prevent cycles)
            if neighbor_id in message.path:
                continue
            
            # Degrade confidence and increment hops
            forwarded_message = message.degrade_confidence(self.confidence_degradation)
            forwarded_message.hops += 1
            forwarded_message.path.append(neighbor_id)
            
            # Recursively spread
            self._spread_message(forwarded_message, neighbor_id)
    
    def _get_neighbors(self, agent_id: str) -> List[str]:
        """
        Get neighboring agents for gossip spread.
        Could be based on network topology, domain overlap, etc.
        """
        # Simple: all other agents are neighbors
        return [aid for aid in self.agents.keys() if aid != agent_id]
    
    def query_knowledge(
        self,
        agent_id: str,
        query_topic: str
    ) -> Optional[EpistemicState]:
        """
        Query what an agent knows through gossip.
        May have multiple versions with different confidence.
        """
        agent_states = self.agent_knowledge.get(agent_id, [])
        
        if not agent_states:
            return None
        
        # If multiple states, merge them
        return merge_epistemic_states(agent_states)
    
    def get_network_consensus(self, topic: str) -> EpistemicState:
        """
        Aggregate what entire network knows about topic.
        Network-level epistemic state.
        """
        all_states = []
        for agent_states in self.agent_knowledge.values():
            all_states.extend(agent_states)
        
        if not all_states:
            from ..primitives.epistemic import UnknownResponse
            return UnknownResponse("No knowledge in network").to_epistemic_state()
        
        return merge_epistemic_states(all_states)
    
    def get_gossip_metrics(self) -> Dict:
        """
        Analytics on gossip quality and epistemic preservation.
        """
        if not self.message_history:
            return {"message": "No gossip activity"}
        
        # Track confidence degradation
        initial_confidences = [m.content.confidence.value for m in self.message_history if m.hops == 0]
        final_confidences = [m.content.confidence.value for m in self.message_history if m.hops == self.max_hops - 1]
        
        avg_initial = sum(initial_confidences) / len(initial_confidences) if initial_confidences else 0
        avg_final = sum(final_confidences) / len(final_confidences) if final_confidences else 0
        
        return {
            'total_messages': len(self.message_history),
            'unique_sources': len(set(m.source_agent for m in self.message_history)),
            'avg_hops': sum(m.hops for m in self.message_history) / len(self.message_history),
            'avg_initial_confidence': avg_initial,
            'avg_final_confidence': avg_final,
            'confidence_degradation': avg_initial - avg_final if avg_initial else 0,
            'network_coverage': len([k for k in self.agent_knowledge.values() if k]) / len(self.agents)
        }


class ClarityAmplification:
    """
    Instead of spreading rumors, amplify clarity.
    Agents share what they DON'T know, not just what they do.
    """
    
    def __init__(self):
        self.shared_unknowns: Dict[str, Set[str]] = {}  # agent_id -> set of known unknowns
        self.delegation_network: Dict[str, Dict[str, str]] = {}  # who knows who to ask
    
    def share_unknown(self, agent_id: str, unknown_domain: str):
        """
        Agent shares what it doesn't know.
        Humility as positive signal.
        """
        if agent_id not in self.shared_unknowns:
            self.shared_unknowns[agent_id] = set()
        
        self.shared_unknowns[agent_id].add(unknown_domain)
    
    def share_delegation(self, agent_id: str, domain: str, expert_id: str):
        """
        Agent shares "I don't know X, but Agent Y does."
        Building collective knowledge map.
        """
        if agent_id not in self.delegation_network:
            self.delegation_network[agent_id] = {}
        
        self.delegation_network[agent_id][domain] = expert_id
    
    def find_expert(self, domain: str) -> Optional[str]:
        """
        Find agent that knows about domain.
        Based on shared delegation knowledge.
        """
        for agent_delegations in self.delegation_network.values():
            if domain in agent_delegations:
                return agent_delegations[domain]
        
        return None
    
    def get_knowledge_map(self) -> Dict:
        """
        Visualize collective knowledge and unknowns.
        """
        return {
            'total_shared_unknowns': sum(len(unknowns) for unknowns in self.shared_unknowns.values()),
            'agents_sharing_unknowns': len(self.shared_unknowns),
            'delegation_connections': sum(len(dels) for dels in self.delegation_network.values()),
            'coverage': {
                domain: expert 
                for agent_dels in self.delegation_network.values() 
                for domain, expert in agent_dels.items()
            }
        }
