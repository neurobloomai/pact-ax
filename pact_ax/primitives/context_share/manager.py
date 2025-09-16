"""
PACT-AX: Agent Collaboration Layer
ContextShareManager - First Primitive

Enables organic context sharing with trust awareness and capability sensing.
Built on principles of dynamic evolution and natural harmony.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import json


class ContextType(Enum):
    """Types of context that can be shared between agents"""
    TASK_KNOWLEDGE = "task_knowledge"
    EMOTIONAL_STATE = "emotional_state"
    CAPABILITY_STATUS = "capability_status"
    LEARNING_INSIGHT = "learning_insight"
    HANDOFF_REQUEST = "handoff_request"
    TRUST_SIGNAL = "trust_signal"


class TrustLevel(Enum):
    """Dynamic trust levels that evolve over time"""
    UNKNOWN = 0.0
    EMERGING = 0.3
    BUILDING = 0.6
    STRONG = 0.8
    DEEP = 1.0


@dataclass
class ContextPacket:
    """A unit of context shared between agents"""
    from_agent: str
    to_agent: str
    context_type: ContextType
    payload: Dict[str, Any]
    timestamp: float
    trust_required: TrustLevel
    priority: int = 5  # 1-10, 10 being highest
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class AgentTrustProfile:
    """Dynamic trust relationship between agents"""
    agent_id: str
    trust_levels: Dict[ContextType, float]
    interaction_history: List[Dict]
    last_updated: float
    evolution_rate: float = 0.1  # How quickly trust adapts
    
    def update_trust(self, context_type: ContextType, outcome: str, impact: float):
        """Evolve trust based on interaction outcomes"""
        current_trust = self.trust_levels.get(context_type, 0.5)
        
        if outcome == "positive":
            new_trust = min(1.0, current_trust + (impact * self.evolution_rate))
        elif outcome == "negative":
            new_trust = max(0.0, current_trust - (impact * self.evolution_rate))
        else:  # neutral
            new_trust = current_trust
            
        self.trust_levels[context_type] = new_trust
        self.last_updated = time.time()
        
        # Record interaction for learning
        self.interaction_history.append({
            "context_type": context_type.value,
            "outcome": outcome,
            "impact": impact,
            "trust_before": current_trust,
            "trust_after": new_trust,
            "timestamp": time.time()
        })


class ContextShareManager:
    """
    Core primitive for agent collaboration in PACT-AX
    Enables organic context sharing with evolving trust and capability awareness
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.trust_profiles: Dict[str, AgentTrustProfile] = {}
        self.context_buffer: List[ContextPacket] = []
        self.capability_sensors: Dict[str, float] = {}  # task -> confidence
        self.collaboration_patterns: Dict[str, Any] = {}  # learned patterns
        
    def create_context_packet(
        self,
        target_agent: str,
        context_type: ContextType,
        payload: Dict[str, Any],
        priority: int = 5,
        ttl_seconds: Optional[int] = None
    ) -> ContextPacket:
        """Create a context packet with trust and capability awareness"""
        
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds
            
        # Determine required trust level based on context sensitivity
        trust_required = self._assess_required_trust(context_type, payload)
        
        packet = ContextPacket(
            from_agent=self.agent_id,
            to_agent=target_agent,
            context_type=context_type,
            payload=payload,
            timestamp=time.time(),
            trust_required=trust_required,
            priority=priority,
            expires_at=expires_at
        )
        
        return packet
    
    def assess_trust(
        self,
        target_agent: str,
        context_type: ContextType,
        current_situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess trust level for sharing specific context with target agent"""
        
        if target_agent not in self.trust_profiles:
            # Initialize new agent with neutral trust
            self.trust_profiles[target_agent] = AgentTrustProfile(
                agent_id=target_agent,
                trust_levels={ct: 0.5 for ct in ContextType},
                interaction_history=[],
                last_updated=time.time()
            )
        
        profile = self.trust_profiles[target_agent]
        base_trust = profile.trust_levels.get(context_type, 0.5)
        
        # Adjust trust based on current situation
        situation_adjustment = self._calculate_situation_adjustment(
            current_situation, profile.interaction_history
        )
        
        adjusted_trust = max(0.0, min(1.0, base_trust + situation_adjustment))
        
        return {
            "agent_id": target_agent,
            "context_type": context_type.value,
            "base_trust": base_trust,
            "situation_adjustment": situation_adjustment,
            "final_trust": adjusted_trust,
            "recommendation": "share" if adjusted_trust > 0.6 else "caution"
        }
    
    def sense_capability_limit(
        self,
        current_task: str,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Sense approaching capability limits for proactive handoff"""
        
        current_confidence = self.capability_sensors.get(current_task, 1.0)
        approaching_limit = current_confidence < confidence_threshold
        
        # Calculate how close we are to the limit
        limit_proximity = max(0.0, 1.0 - (current_confidence / confidence_threshold))
        
        return {
            "task": current_task,
            "current_confidence": current_confidence,
            "threshold": confidence_threshold,
            "approaching_limit": approaching_limit,
            "limit_proximity": limit_proximity,
            "recommendation": self._get_capability_recommendation(
                current_confidence, confidence_threshold
            )
        }
    
    def prepare_handoff(
        self,
        target_agent: str,
        current_task: str,
        preserve_emotional_context: bool = True,
        transfer_ownership: bool = True
    ) -> ContextPacket:
        """Prepare smooth handoff before hitting capability limits"""
        
        handoff_payload = {
            "current_task": current_task,
            "handoff_reason": "proactive_capability_limit",
            "task_progress": self._gather_task_progress(),
            "transfer_ownership": transfer_ownership,
            "recommendations": self._generate_handoff_recommendations(current_task)
        }
        
        if preserve_emotional_context:
            handoff_payload["emotional_context"] = self._gather_emotional_context()
        
        return self.create_context_packet(
            target_agent=target_agent,
            context_type=ContextType.HANDOFF_REQUEST,
            payload=handoff_payload,
            priority=8  # High priority for handoffs
        )
    
    def update_capability_confidence(self, task: str, confidence: float):
        """Update confidence level for a specific task"""
        self.capability_sensors[task] = max(0.0, min(1.0, confidence))
    
    def record_collaboration_outcome(
        self,
        target_agent: str,
        context_type: ContextType,
        outcome: str,
        impact: float = 1.0
    ):
        """Record collaboration outcome to evolve trust and patterns"""
        
        if target_agent in self.trust_profiles:
            self.trust_profiles[target_agent].update_trust(
                context_type, outcome, impact
            )
        
        # Learn collaboration patterns
        pattern_key = f"{target_agent}_{context_type.value}"
        if pattern_key not in self.collaboration_patterns:
            self.collaboration_patterns[pattern_key] = {
                "successes": 0,
                "failures": 0,
                "patterns": []
            }
        
        pattern = self.collaboration_patterns[pattern_key]
        if outcome == "positive":
            pattern["successes"] += 1
        elif outcome == "negative":
            pattern["failures"] += 1
    
    def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights about collaboration patterns and trust evolution"""
        
        insights = {
            "trust_summary": {},
            "capability_status": dict(self.capability_sensors),
            "collaboration_patterns": self.collaboration_patterns,
            "evolution_trends": []
        }
        
        # Summarize trust across all agents
        for agent_id, profile in self.trust_profiles.items():
            insights["trust_summary"][agent_id] = {
                "average_trust": sum(profile.trust_levels.values()) / len(profile.trust_levels),
                "strongest_context": max(profile.trust_levels.items(), key=lambda x: x[1]),
                "interaction_count": len(profile.interaction_history),
                "last_interaction": profile.last_updated
            }
        
        return insights
    
    # Private helper methods
    
    def _assess_required_trust(self, context_type: ContextType, payload: Dict[str, Any]) -> TrustLevel:
        """Determine required trust level based on context sensitivity"""
        
        sensitive_keys = ["password", "secret", "private", "confidential"]
        is_sensitive = any(key in str(payload).lower() for key in sensitive_keys)
        
        if context_type == ContextType.EMOTIONAL_STATE:
            return TrustLevel.STRONG
        elif context_type == ContextType.HANDOFF_REQUEST:
            return TrustLevel.BUILDING
        elif is_sensitive:
            return TrustLevel.DEEP
        else:
            return TrustLevel.EMERGING
    
    def _calculate_situation_adjustment(
        self, 
        situation: Dict[str, Any], 
        history: List[Dict]
    ) -> float:
        """Calculate trust adjustment based on current situation"""
        
        adjustment = 0.0
        
        # High-stakes situations require more trust
        if situation.get("stakes") == "high":
            adjustment -= 0.1
        elif situation.get("stakes") == "low":
            adjustment += 0.05
        
        # Recent positive interactions boost trust
        recent_interactions = [h for h in history if time.time() - h["timestamp"] < 3600]
        if recent_interactions:
            recent_outcomes = [h["outcome"] for h in recent_interactions]
            positive_ratio = recent_outcomes.count("positive") / len(recent_outcomes)
            adjustment += (positive_ratio - 0.5) * 0.2
        
        return adjustment
    
    def _get_capability_recommendation(self, confidence: float, threshold: float) -> str:
        """Get recommendation based on capability confidence"""
        
        if confidence > threshold * 1.2:
            return "continue"
        elif confidence > threshold:
            return "monitor"
        elif confidence > threshold * 0.8:
            return "prepare_handoff"
        else:
            return "immediate_handoff"
    
    def _gather_task_progress(self) -> Dict[str, Any]:
        """Gather current task progress for handoff"""
        # Placeholder - would integrate with actual task tracking
        return {
            "completion_percentage": 0.75,
            "current_step": "analysis_phase",
            "next_steps": ["validation", "implementation"],
            "blockers": []
        }
    
    def _gather_emotional_context(self) -> Dict[str, Any]:
        """Gather emotional context for handoff"""
        # Placeholder - would integrate with PACT-HX layer
        return {
            "user_sentiment": "neutral",
            "interaction_tone": "professional",
            "stress_indicators": [],
            "preferences": {}
        }
    
    def _generate_handoff_recommendations(self, task: str) -> List[str]:
        """Generate recommendations for the receiving agent"""
        return [
            "Maintain current communication style",
            "User prefers detailed explanations",
            "Task complexity is moderate"
        ]


# Example usage demonstrating organic collaboration
if __name__ == "__main__":
    # Initialize agent collaboration
    manager = ContextShareManager("agent-001")
    
    # Share context with trust awareness
    context_packet = manager.create_context_packet(
        target_agent="agent-002",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "current_task": "customer_support",
            "priority": "high",
            "user_context": "technical_issue"
        }
    )
    
    # Assess trust for collaboration
    trust_assessment = manager.assess_trust(
        target_agent="agent-002",
        context_type=ContextType.TASK_KNOWLEDGE,
        current_situation={"complexity": "high", "stakes": "medium"}
    )
    
    print("Trust Assessment:", json.dumps(trust_assessment, indent=2))
    
    # Monitor capability and prepare for handoff
    capability_status = manager.sense_capability_limit(
        current_task="customer_support",
        confidence_threshold=0.7
    )
    
    print("Capability Status:", json.dumps(capability_status, indent=2))
    
    # Record collaboration outcome to evolve trust
    manager.record_collaboration_outcome(
        target_agent="agent-002",
        context_type=ContextType.TASK_KNOWLEDGE,
        outcome="positive",
        impact=1.0
    )
    
    # Get insights about collaboration evolution
    insights = manager.get_collaboration_insights()
    print("Collaboration Insights:", json.dumps(insights, indent=2))
