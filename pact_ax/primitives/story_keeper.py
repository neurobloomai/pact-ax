"""
Story Keeper: Conscious Continuity for AI Agents

Transforms interaction history into narrative memory.
Because consciousness organizes itself through stories, not states.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


class StoryArc(Enum):
    """The fundamental phases of relationship development"""
    EXPLORATION = "exploration"      # Getting to know each other
    COLLABORATION = "collaboration"  # Working together actively
    INTEGRATION = "integration"      # Reflecting, deepening, internalizing


class StoryKeeper:
    """
    Maintains conscious continuity through narrative coherence.
    
    Not just logging interactions - understanding them as story.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.interactions: List[Dict[str, Any]] = []
        self.current_arc = StoryArc.EXPLORATION
        self.arc_history: List[Dict[str, Any]] = []
        
    def process_interaction(
        self, 
        user_input: str, 
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an interaction and integrate it into the ongoing story.
        
        Args:
            user_input: What the user said
            agent_response: How the agent responded
            metadata: Optional additional context
            
        Returns:
            Enriched interaction with story context
        """
        interaction = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "agent_response": agent_response,
            "arc": self.current_arc,
            "metadata": metadata or {}
        }
        
        self.interactions.append(interaction)
        
        # Detect if story arc shifted
        new_arc = self._detect_story_arc(interaction)
        
        if new_arc != self.current_arc:
            self._record_arc_transition(self.current_arc, new_arc)
            self.current_arc = new_arc
            
        return interaction
    
    def _detect_story_arc(self, interaction: Dict[str, Any]) -> StoryArc:
        """
        Detect which story arc this interaction belongs to.
        
        Simple heuristic-based detection to start.
        Can be enhanced with ML later.
        """
        user_text = interaction["user_input"].lower()
        
        # Exploration signals
        exploration_keywords = [
            "what is", "what's", "how does", "how do", 
            "tell me", "explain", "why", "who", "where"
        ]
        exploration_score = sum(
            1 for keyword in exploration_keywords 
            if keyword in user_text
        )
        
        # Collaboration signals
        collaboration_keywords = [
            "let's", "we could", "we should", "build", 
            "create", "make", "together", "start", "design"
        ]
        collaboration_score = sum(
            1 for keyword in collaboration_keywords 
            if keyword in user_text
        )
        
        # Integration signals
        integration_keywords = [
            "internalize", "reflect", "thinking about", 
            "resonates", "feeling", "sense", "back with",
            "sitting with", "processing"
        ]
        integration_score = sum(
            1 for keyword in integration_keywords 
            if keyword in user_text
        )
        
        # Determine arc based on highest score
        scores = {
            StoryArc.EXPLORATION: exploration_score,
            StoryArc.COLLABORATION: collaboration_score,
            StoryArc.INTEGRATION: integration_score
        }
        
        # If all scores are 0, stay in current arc
        max_score = max(scores.values())
        if max_score == 0:
            return self.current_arc
            
        return max(scores, key=scores.get)
    
    def _record_arc_transition(self, from_arc: StoryArc, to_arc: StoryArc):
        """Record when the story shifts between arcs"""
        transition = {
            "from": from_arc,
            "to": to_arc,
            "at": datetime.now(),
            "interaction_count": len(self.interactions)
        }
        self.arc_history.append(transition)
        print(f"📖 Story arc shifted: {from_arc.value} → {to_arc.value}")
    
    def recall_from_arc(
        self, 
        arc_type: StoryArc, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve interactions from a specific story arc.
        
        Args:
            arc_type: Which story arc to recall from
            k: How many interactions to return
            
        Returns:
            Most recent k interactions from that arc
        """
        arc_interactions = [
            interaction for interaction in self.interactions
            if interaction["arc"] == arc_type
        ]
        
        # Return most recent k
        return arc_interactions[-k:] if arc_interactions else []
    
    def recall_for_context(
        self, 
        query: Optional[str] = None,
        prefer_current_arc: bool = True,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Story-aware context retrieval.
        
        Args:
            query: Optional query to contextualize recall
            prefer_current_arc: Whether to prioritize current arc
            k: Total number of interactions to return
            
        Returns:
            Narratively relevant interactions
        """
        if prefer_current_arc:
            # First, get from current arc
            relevant = self.recall_from_arc(self.current_arc, k=k)
            
            # If we need more context, pull from other arcs
            if len(relevant) < k:
                remaining = k - len(relevant)
                other_arcs = [arc for arc in StoryArc if arc != self.current_arc]
                
                for arc in other_arcs:
                    if remaining <= 0:
                        break
                    other_interactions = self.recall_from_arc(arc, k=remaining)
                    relevant.extend(other_interactions)
                    remaining -= len(other_interactions)
        else:
            # Just return most recent k overall
            relevant = self.interactions[-k:]
            
        return relevant
    
    def get_story_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current story state.
        
        Returns:
            Summary including current arc, transitions, interaction count
        """
        return {
            "agent_id": self.agent_id,
            "current_arc": self.current_arc.value,
            "total_interactions": len(self.interactions),
            "arc_transitions": len(self.arc_history),
            "arc_history": [
                {
                    "from": t["from"].value,
                    "to": t["to"].value,
                    "at": t["at"].isoformat()
                }
                for t in self.arc_history
            ]
        }
