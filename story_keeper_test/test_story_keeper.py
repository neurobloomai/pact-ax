# test_story_keeper.py
# (paste the entire StoryKeeper class code here)

# Then add at bottom:
"""
Standalone Story Keeper Test
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


class StoryArc(Enum):
    """The fundamental phases of relationship development"""
    EXPLORATION = "exploration"
    COLLABORATION = "collaboration"
    INTEGRATION = "integration"


class StoryKeeper:
    """
    Maintains conscious continuity through narrative coherence.
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
        """Process an interaction and integrate it into the ongoing story."""
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
        """Detect which story arc this interaction belongs to."""
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
        """Retrieve interactions from a specific story arc."""
        arc_interactions = [
            interaction for interaction in self.interactions
            if interaction["arc"] == arc_type
        ]
        return arc_interactions[-k:] if arc_interactions else []
    
    def recall_for_context(
        self, 
        query: Optional[str] = None,
        prefer_current_arc: bool = True,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Story-aware context retrieval."""
        if prefer_current_arc:
            relevant = self.recall_from_arc(self.current_arc, k=k)
            
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
            relevant = self.interactions[-k:]
            
        return relevant
    
    def get_story_summary(self) -> Dict[str, Any]:
        """Get a summary of the current story state."""
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


# ============================================
# TEST CODE STARTS HERE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Story Keeper Test: Conscious Continuity in Action")
    print("=" * 60)
    print()
    
    # Create Story Keeper
    sk = StoryKeeper("test-agent")
    
    # Test conversation with arc transitions
    conversations = [
        # EXPLORATION phase
        ("What is Story Keeper?", 
         "Story Keeper maintains conscious continuity by transforming interactions into narrative memory."),
        
        ("How does it work?", 
         "It detects story arcs and recalls context based on narrative coherence."),
        
        # Transition to COLLABORATION
        ("Let's build something together",
         "Yes! I'd love to co-create with you. What should we build?"),
        
        ("We could start with improving arc detection",
         "Great idea. We could add more sophisticated signals."),
        
        # Transition to INTEGRATION
        ("Let me think about this for a bit",
         "Of course. Take the time you need to internalize."),
        
        ("I'm back with some thoughts",
         "Welcome b:ack! What emerged during your reflection?"),
    ]
    
    # Process interactions
    print("CONVERSATION:")
    print("-" * 60)
    for user_msg, agent_msg in conversations:
        print(f"\nUser: {user_msg}")
        print(f"Agent: {agent_msg}")
        sk.process_interaction(user_msg, agent_msg)
    
    # Show summary
    print("\n" + "=" * 60)
    print("STORY SUMMARY")
    print("=" * 60)
    summary = sk.get_story_summary()
    print(f"Current Arc: {summary['current_arc']}")
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"Arc Transitions: {summary['arc_transitions']}")
    
    if summary['arc_history']:
        print("\nArc Transition History:")
        for transition in summary['arc_history']:
            print(f"  {transition['from']} → {transition['to']}")
    
    # Test arc-aware recall
    print("\n" + "=" * 60)
    print("ARC-AWARE RECALL")
    print("=" * 60)
    
    print("\n📖 From EXPLORATION arc:")
    for moment in sk.recall_from_arc(StoryArc.EXPLORATION, k=2):
        print(f"   User: {moment['user_input']}")
    
    print("\n📖 From COLLABORATION arc:")
    for moment in sk.recall_from_arc(StoryArc.COLLABORATION, k=2):
        print(f"   User: {moment['user_input']}")
    
    print("\n📖 From INTEGRATION arc:")
    for moment in sk.recall_from_arc(StoryArc.INTEGRATION, k=2):
        print(f"   User: {moment['user_input']}")
    
    print("\n" + "=" * 60)
    print("✨ Story Keeper is alive!")
    print("=" * 60)
