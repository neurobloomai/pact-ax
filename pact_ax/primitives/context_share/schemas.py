"""
PACT-AX: Agent Collaboration Layer
Context Share Schemas

Defines the data structures for organic agent collaboration,
with support for dynamic evolution and trust-aware context sharing.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import uuid
import json


class ContextType(Enum):
    """Types of context that can be shared between agents"""
    TASK_KNOWLEDGE = "task_knowledge"
    EMOTIONAL_STATE = "emotional_state"
    CAPABILITY_STATUS = "capability_status"
    LEARNING_INSIGHT = "learning_insight"
    HANDOFF_REQUEST = "handoff_request"
    TRUST_SIGNAL = "trust_signal"
    SYSTEM_STATE = "system_state"
    USER_PREFERENCE = "user_preference"
    COLLABORATION_PATTERN = "collaboration_pattern"


class TrustLevel(Enum):
    """Dynamic trust levels that evolve through collaboration"""
    UNKNOWN = 0.0
    EMERGING = 0.3
    BUILDING = 0.6
    STRONG = 0.8
    DEEP = 1.0
    
    @classmethod
    def from_float(cls, value: float) -> 'TrustLevel':
        """Convert float to closest trust level"""
        closest = min(cls, key=lambda x: abs(x.value - value))
        return closest


class Priority(Enum):
    """Message priority levels"""
    BACKGROUND = 1
    LOW = 3
    NORMAL = 5
    HIGH = 7
    URGENT = 9
    CRITICAL = 10


class CollaborationOutcome(Enum):
    """Outcomes of collaboration attempts"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class CapabilityStatus(Enum):
    """Agent capability status indicators"""
    EXCELLENT = auto()
    GOOD = auto()
    MODERATE = auto()
    LIMITED = auto()
    DEGRADED = auto()
    FAILING = auto()


@dataclass
class AgentIdentity:
    """Immutable agent identity with capabilities"""
    agent_id: str
    agent_type: str
    version: str
    capabilities: List[str]
    specializations: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validate agent identity"""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        if not self.agent_type:
            raise ValueError("Agent type must be specified")


@dataclass
class ContextMetadata:
    """Metadata for context packets"""
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    encryption_level: str = "none"
    checksum: Optional[str] = None
    lineage: List[str] = field(default_factory=list)  # Track context evolution
    
    def add_to_lineage(self, agent_id: str, operation: str):
        """Add to context lineage for traceability"""
        self.lineage.append(f"{agent_id}:{operation}:{datetime.now(timezone.utc).isoformat()}")


@dataclass
class ContextPacket:
    """Core unit of context shared between agents"""
    from_agent: AgentIdentity
    to_agent: str  # Can be specific agent or pattern
    context_type: ContextType
    payload: Dict[str, Any]
    metadata: ContextMetadata
    trust_required: TrustLevel
    priority: Priority = Priority.NORMAL
    expires_at: Optional[datetime] = None
    conditions: Optional[Dict[str, Any]] = None  # Conditions for acceptance
    
    def __post_init__(self):
        """Validate context packet"""
        if not self.payload:
            raise ValueError("Context payload cannot be empty")
        
        # Add creation info to lineage
        self.metadata.add_to_lineage(
            self.from_agent.agent_id, 
            f"created_{self.context_type.value}"
        )
    
    def is_expired(self) -> bool:
        """Check if context packet has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid_for_agent(self, agent_identity: AgentIdentity) -> bool:
        """Check if packet is valid for receiving agent"""
        if self.is_expired():
            return False
        
        # Check if agent has required capabilities for this context type
        if self.conditions:
            required_caps = self.conditions.get("required_capabilities", [])
            if not all(cap in agent_identity.capabilities for cap in required_caps):
                return False
        
        return True
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sensitive data marked"""
        return {
            "packet_id": self.metadata.packet_id,
            "from_agent": self.from_agent.agent_id,
            "to_agent": self.to_agent,
            "context_type": self.context_type.value,
            "payload": "[ENCRYPTED]" if self.metadata.encryption_level != "none" else self.payload,
            "trust_required": self.trust_required.value,
            "priority": self.priority.value,
            "created_at": self.metadata.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "lineage_length": len(self.metadata.lineage)
        }


@dataclass
class TrustEvolution:
    """Tracks how trust evolves over time"""
    context_type: ContextType
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    current_level: float = 0.5  # Start neutral
    trend_direction: str = "stable"  # up, down, stable
    volatility: float = 0.1  # How much trust fluctuates
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_interaction(self, outcome: CollaborationOutcome, impact: float, context: Dict[str, Any]):
        """Record a new interaction and update trust"""
        interaction = {
            "outcome": outcome.value,
            "impact": impact,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trust_before": self.current_level
        }
        
        # Update trust level
        if outcome == CollaborationOutcome.POSITIVE:
            self.current_level = min(1.0, self.current_level + (impact * 0.1))
        elif outcome == CollaborationOutcome.NEGATIVE:
            self.current_level = max(0.0, self.current_level - (impact * 0.2))
        # Neutral and partial outcomes don't change trust significantly
        
        interaction["trust_after"] = self.current_level
        self.interactions.append(interaction)
        self.last_updated = datetime.now(timezone.utc)
        
        # Update trend and volatility
        self._update_trend()
    
    def _update_trend(self):
        """Calculate trust trend from recent interactions"""
        if len(self.interactions) < 3:
            self.trend_direction = "stable"
            return
        
        recent = self.interactions[-5:]  # Last 5 interactions
        trust_changes = [i["trust_after"] - i["trust_before"] for i in recent]
        avg_change = sum(trust_changes) / len(trust_changes)
        
        if avg_change > 0.05:
            self.trend_direction = "up"
        elif avg_change < -0.05:
            self.trend_direction = "down"
        else:
            self.trend_direction = "stable"
        
        # Update volatility based on variance
        variance = sum((change - avg_change) ** 2 for change in trust_changes) / len(trust_changes)
        self.volatility = min(1.0, variance * 10)  # Scale to 0-1


@dataclass
class AgentTrustProfile:
    """Complete trust profile between two agents"""
    agent_id: str
    trust_evolution: Dict[ContextType, TrustEvolution] = field(default_factory=dict)
    overall_trust: float = 0.5
    collaboration_patterns: Dict[str, Any] = field(default_factory=dict)
    last_interaction: Optional[datetime] = None
    interaction_frequency: float = 0.0  # interactions per day
    
    def get_trust_for_context(self, context_type: ContextType) -> float:
        """Get current trust level for specific context type"""
        if context_type not in self.trust_evolution:
            self.trust_evolution[context_type] = TrustEvolution(context_type=context_type)
        
        return self.trust_evolution[context_type].current_level
    
    def update_trust(self, context_type: ContextType, outcome: CollaborationOutcome, 
                    impact: float, context: Dict[str, Any]):
        """Update trust for specific context type"""
        if context_type not in self.trust_evolution:
            self.trust_evolution[context_type] = TrustEvolution(context_type=context_type)
        
        self.trust_evolution[context_type].add_interaction(outcome, impact, context)
        self.last_interaction = datetime.now(timezone.utc)
        
        # Recalculate overall trust as weighted average
        total_weight = 0
        total_trust = 0
        for ct, evolution in self.trust_evolution.items():
            weight = len(evolution.interactions)  # More interactions = higher weight
            total_weight += weight
            total_trust += evolution.current_level * weight
        
        if total_weight > 0:
            self.overall_trust = total_trust / total_weight


@dataclass
class CapabilitySensor:
    """Monitors agent capability for specific tasks"""
    task_type: str
    current_confidence: float = 1.0
    confidence_history: List[Dict[str, Any]] = field(default_factory=list)
    degradation_rate: float = 0.0  # How fast capability degrades
    recovery_rate: float = 0.1   # How fast capability recovers
    threshold_warning: float = 0.7
    threshold_critical: float = 0.5
    
    def update_confidence(self, new_confidence: float, context: Dict[str, Any]):
        """Update confidence level with context"""
        old_confidence = self.current_confidence
        self.current_confidence = max(0.0, min(1.0, new_confidence))
        
        self.confidence_history.append({
            "confidence_before": old_confidence,
            "confidence_after": self.current_confidence,
            "change": self.current_confidence - old_confidence,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update degradation/recovery rates
        if len(self.confidence_history) > 1:
            recent_changes = [h["change"] for h in self.confidence_history[-5:]]
            avg_change = sum(recent_changes) / len(recent_changes)
            
            if avg_change < 0:
                self.degradation_rate = abs(avg_change)
            else:
                self.recovery_rate = avg_change
    
    def get_status(self) -> CapabilityStatus:
        """Get current capability status"""
        if self.current_confidence >= 0.9:
            return CapabilityStatus.EXCELLENT
        elif self.current_confidence >= 0.8:
            return CapabilityStatus.GOOD
        elif self.current_confidence >= 0.7:
            return CapabilityStatus.MODERATE
        elif self.current_confidence >= 0.5:
            return CapabilityStatus.LIMITED
        elif self.current_confidence >= 0.3:
            return CapabilityStatus.DEGRADED
        else:
            return CapabilityStatus.FAILING
    
    def should_warn(self) -> bool:
        """Check if capability warning should be issued"""
        return self.current_confidence <= self.threshold_warning
    
    def should_handoff(self) -> bool:
        """Check if immediate handoff is recommended"""
        return self.current_confidence <= self.threshold_critical


@dataclass
class HandoffRequest:
    """Schema for agent handoff requests"""
    from_agent: AgentIdentity
    to_agent: str
    task_context: Dict[str, Any]
    handoff_reason: str
    urgency: Priority = Priority.NORMAL
    preserve_context: List[ContextType] = field(default_factory=list)
    transfer_ownership: bool = True
    rollback_conditions: Optional[Dict[str, Any]] = None
    success_criteria: Optional[Dict[str, Any]] = None
    
    def to_context_packet(self) -> ContextPacket:
        """Convert handoff request to context packet"""
        payload = {
            "task_context": self.task_context,
            "handoff_reason": self.handoff_reason,
            "preserve_context": [ct.value for ct in self.preserve_context],
            "transfer_ownership": self.transfer_ownership,
            "rollback_conditions": self.rollback_conditions,
            "success_criteria": self.success_criteria
        }
        
        return ContextPacket(
            from_agent=self.from_agent,
            to_agent=self.to_agent,
            context_type=ContextType.HANDOFF_REQUEST,
            payload=payload,
            metadata=ContextMetadata(),
            trust_required=TrustLevel.BUILDING,
            priority=self.urgency
        )


@dataclass
class CollaborationPattern:
    """Learned patterns of successful collaboration"""
    pattern_id: str
    context_types: List[ContextType]
    agent_types: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    
    def update_success(self, successful: bool):
        """Update pattern success rate"""
        old_total = self.usage_count * self.success_rate
        self.usage_count += 1
        
        if successful:
            old_total += 1
        
        self.success_rate = old_total / self.usage_count
        self.last_used = datetime.now(timezone.utc)


# Validation functions
def validate_context_packet(packet: ContextPacket) -> List[str]:
    """Validate context packet and return list of errors"""
    errors = []
    
    if not packet.from_agent.agent_id:
        errors.append("Missing from_agent ID")
    
    if not packet.to_agent:
        errors.append("Missing to_agent")
    
    if not packet.payload:
        errors.append("Empty payload")
    
    if packet.is_expired():
        errors.append("Packet has expired")
    
    # Context-specific validation
    if packet.context_type == ContextType.HANDOFF_REQUEST:
        required_fields = ["task_context", "handoff_reason"]
        missing_fields = [f for f in required_fields if f not in packet.payload]
        if missing_fields:
            errors.append(f"Handoff request missing fields: {missing_fields}")
    
    return errors


def serialize_context_packet(packet: ContextPacket) -> str:
    """Serialize context packet to JSON string"""
    data = {
        "from_agent": {
            "agent_id": packet.from_agent.agent_id,
            "agent_type": packet.from_agent.agent_type,
            "version": packet.from_agent.version,
            "capabilities": packet.from_agent.capabilities
        },
        "to_agent": packet.to_agent,
        "context_type": packet.context_type.value,
        "payload": packet.payload,
        "metadata": {
            "packet_id": packet.metadata.packet_id,
            "created_at": packet.metadata.created_at.isoformat(),
            "version": packet.metadata.version,
            "encryption_level": packet.metadata.encryption_level,
            "lineage": packet.metadata.lineage
        },
        "trust_required": packet.trust_required.value,
        "priority": packet.priority.value,
        "expires_at": packet.expires_at.isoformat() if packet.expires_at else None,
        "conditions": packet.conditions
    }
    
    return json.dumps(data, indent=2)


def deserialize_context_packet(json_str: str) -> ContextPacket:
    """Deserialize JSON string to context packet"""
    data = json.loads(json_str)
    
    from_agent = AgentIdentity(
        agent_id=data["from_agent"]["agent_id"],
        agent_type=data["from_agent"]["agent_type"],
        version=data["from_agent"]["version"],
        capabilities=data["from_agent"]["capabilities"],
        specializations=data["from_agent"].get("specializations", [])
    )
    
    metadata = ContextMetadata(
        packet_id=data["metadata"]["packet_id"],
        created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
        version=data["metadata"]["version"],
        encryption_level=data["metadata"]["encryption_level"],
        lineage=data["metadata"]["lineage"]
    )
    
    expires_at = None
    if data.get("expires_at"):
        expires_at = datetime.fromisoformat(data["expires_at"])
    
    return ContextPacket(
        from_agent=from_agent,
        to_agent=data["to_agent"],
        context_type=ContextType(data["context_type"]),
        payload=data["payload"],
        metadata=metadata,
        trust_required=TrustLevel(data["trust_required"]),
        priority=Priority(data["priority"]),
        expires_at=expires_at,
        conditions=data.get("conditions")
    )


# Example usage and testing
if __name__ == "__main__":
    # Create agent identity
    agent = AgentIdentity(
        agent_id="agent-001",
        agent_type="support_specialist",
        version="1.0.0",
        capabilities=["natural_language", "customer_support", "technical_analysis"],
        specializations=["troubleshooting", "escalation_handling"]
    )
    
    # Create context packet
    packet = ContextPacket(
        from_agent=agent,
        to_agent="agent-002",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "current_task": "customer_support",
            "priority": "high",
            "user_context": {"issue_type": "technical", "urgency": "high"}
        },
        metadata=ContextMetadata(),
        trust_required=TrustLevel.BUILDING,
        priority=Priority.HIGH
    )
    
    # Validate and serialize
    errors = validate_context_packet(packet)
    if not errors:
        serialized = serialize_context_packet(packet)
        print("Serialized packet:")
        print(serialized)
        
        # Test deserialization
        deserialized = deserialize_context_packet(serialized)
        print(f"\nDeserialized successfully: {deserialized.from_agent.agent_id}")
    else:
        print("Validation errors:", errors)
    
    # Create and test trust profile
    trust_profile = AgentTrustProfile(agent_id="agent-002")
    trust_profile.update_trust(
        context_type=ContextType.TASK_KNOWLEDGE,
        outcome=CollaborationOutcome.POSITIVE,
        impact=1.0,
        context={"task_completion": "success", "user_satisfaction": "high"}
    )
    
    print(f"\nTrust level for task knowledge: {trust_profile.get_trust_for_context(ContextType.TASK_KNOWLEDGE)}")
    
    # Test capability sensor
    sensor = CapabilitySensor(task_type="customer_support")
    sensor.update_confidence(0.6, {"recent_failures": 2, "complexity": "high"})
    print(f"Capability status: {sensor.get_status()}")
    print(f"Should handoff: {sensor.should_handoff()}")
