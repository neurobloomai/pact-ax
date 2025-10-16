"""
PACT-AX: Agent Collaboration Layer
Context Security Manager

Central security primitive that enables trust-aware protection across all PACT-AX primitives.
Balances security with organic collaboration through adaptive, context-sensitive protection.
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import uuid
import json
import threading
from collections import defaultdict, deque

# Import from sibling primitives
from ..context_share.schemas import (
    ContextPacket, AgentIdentity, ContextType, TrustLevel, 
    CollaborationOutcome, ContextMetadata
)
from ..context_share.encryption import TrustAwareEncryption, EncryptionLevel


class SecurityEventType(Enum):
    """Types of security events for audit and learning"""
    CONTEXT_SECURED = "context_secured"
    CONTEXT_VERIFIED = "context_verified"
    DECRYPTION_SUCCESS = "decryption_success"
    DECRYPTION_FAILURE = "decryption_failure"
    TRUST_THRESHOLD_BREACH = "trust_threshold_breach"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    POLICY_VIOLATION = "policy_violation"
    KEY_ROTATION = "key_rotation"
    SECURITY_UPGRADE = "security_upgrade"
    SECURITY_DOWNGRADE = "security_downgrade"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    MINIMAL = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


class SecurityPolicy(Enum):
    """Security policy types"""
    TRUST_BASED = "trust_based"          # Default PACT-AX approach
    ALWAYS_ENCRYPT = "always_encrypt"     # Paranoid mode
    MINIMAL_SECURITY = "minimal_security" # High-trust environment
    REGULATORY_COMPLIANCE = "regulatory"  # Meet specific regulations
    ADAPTIVE_LEARNING = "adaptive"        # Learn from patterns


@dataclass
class SecurityEvent:
    """Security event for audit trail and learning"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.CONTEXT_SECURED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_from: Optional[str] = None
    agent_to: Optional[str] = None
    context_type: Optional[ContextType] = None
    trust_level: Optional[float] = None
    encryption_level: Optional[EncryptionLevel] = None
    threat_level: ThreatLevel = ThreatLevel.MINIMAL
    details: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    impact_score: float = 0.0
    
    def to_learning_data(self) -> Dict[str, Any]:
        """Convert to format suitable for learning algorithms"""
        return {
            "event_type": self.event_type.value,
            "trust_level": self.trust_level,
            "encryption_level": self.encryption_level.value if self.encryption_level else None,
            "threat_level": self.threat_level.value,
            "outcome": self.outcome,
            "impact_score": self.impact_score,
            "context_type": self.context_type.value if self.context_type else None,
            "timestamp_hour": self.timestamp.hour,
            "timestamp_day": self.timestamp.weekday()
        }


@dataclass
class ThreatAssessment:
    """Assessment of current threat landscape"""
    overall_threat_level: ThreatLevel = ThreatLevel.LOW
    context_specific_threats: Dict[ContextType, ThreatLevel] = field(default_factory=dict)
    agent_specific_threats: Dict[str, ThreatLevel] = field(default_factory=dict)
    recent_incidents: List[SecurityEvent] = field(default_factory=list)
    assessment_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = 0.8  # How confident we are in this assessment
    
    def get_threat_for_context(self, context_type: ContextType, agent_id: Optional[str] = None) -> ThreatLevel:
        """Get threat level for specific context and optionally agent"""
        context_threat = self.context_specific_threats.get(context_type, self.overall_threat_level)
        
        if agent_id:
            agent_threat = self.agent_specific_threats.get(agent_id, self.overall_threat_level)
            # Take the higher threat level
            return ThreatLevel(max(context_threat.value, agent_threat.value))
        
        return context_threat


@dataclass
class SecurityMetrics:
    """Metrics for security system performance"""
    total_contexts_secured: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    trust_based_decisions: int = 0
    policy_violations: int = 0
    threat_detections: int = 0
    false_positives: int = 0
    security_downgrades: int = 0
    security_upgrades: int = 0
    average_response_time: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate overall security success rate"""
        total = self.successful_verifications + self.failed_verifications
        if total == 0:
            return 1.0
        return self.successful_verifications / total
    
    def get_threat_detection_accuracy(self) -> float:
        """Calculate threat detection accuracy"""
        total_detections = self.threat_detections + self.false_positives
        if total_detections == 0:
            return 1.0
        return self.threat_detections / total_detections


class ContextSecurityManager:
    """
    Central security manager for all PACT-AX context operations.
    Provides trust-aware security that adapts to collaboration patterns while maintaining protection.
    """
    
    def __init__(self, agent_identity: AgentIdentity, security_policy: SecurityPolicy = SecurityPolicy.TRUST_BASED),story_keeper: Optional[StoryKeeper] = None):
        self.agent_identity = agent_identity
        self.security_policy = security_policy
        self.story_keeper = story_keeper
        
        # Core components
        self.encryption_engine = TrustAwareEncryption()
        self.threat_assessment = ThreatAssessment()
        self.security_metrics = SecurityMetrics()
        
        # Event and audit management
        self.security_events: deque = deque(maxlen=10000)  # Ring buffer for events
        self.event_handlers: Dict[SecurityEventType, List[Callable]] = defaultdict(list)
        
        # Learning and adaptation
        self.learned_patterns: Dict[str, Dict[str, Any]] = {}
        self.trust_deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # Track trust changes
        self.security_adaptations: Dict[str, Dict[str, Any]] = {}
        
        # Threading for async operations
        self._lock = threading.RLock()
        self._background_tasks_enabled = True
        
        # Initialize default threat baselines
        self._initialize_threat_baselines()
    
    def secure_context_for_sharing(self, 
                                 packet: ContextPacket, 
                                 target_agent: str,
                                 trust_level: float,
                                 override_policy: Optional[SecurityPolicy] = None) -> ContextPacket:
        """
        Secure context packet for sharing with another agent.
        Applies trust-aware security while maintaining collaboration fluidity.
        """
        
        with self._lock:
            # Assess current threat landscape
            threat_level = self.threat_assessment.get_threat_for_context(
                packet.context_type, target_agent
            )
            
            # Determine security approach
            policy = override_policy or self.security_policy
            security_decision = self._make_security_decision(
                packet, target_agent, trust_level, threat_level, policy
            )
            
            try:
                # Apply security measures
                secured_packet = self._apply_security_measures(
                    packet, security_decision
                )
                
                # Log successful security application
                self._record_security_event(
                    event_type=SecurityEventType.CONTEXT_SECURED,
                    agent_from=packet.from_agent.agent_id,
                    agent_to=target_agent,
                    context_type=packet.context_type,
                    trust_level=trust_level,
                    encryption_level=security_decision.get("encryption_level"),
                    details=security_decision,
                    outcome="success"
                )
                
                self.security_metrics.total_contexts_secured += 1
                
                # Learn from this decision
                self._learn_from_security_decision(packet, target_agent, trust_level, security_decision)
                
                return secured_packet
                
            except Exception as e:
                # Log security failure
                self._record_security_event(
                    event_type=SecurityEventType.CONTEXT_SECURED,
                    agent_from=packet.from_agent.agent_id,
                    agent_to=target_agent,
                    context_type=packet.context_type,
                    trust_level=trust_level,
                    details={"error": str(e)},
                    outcome="failure",
                    impact_score=0.8
                )
                raise
    
    def verify_and_decrypt_context(self, 
                                 packet: ContextPacket,
                                 expected_from: Optional[str] = None) -> ContextPacket:
        """
        Verify and decrypt context packet for the receiving agent.
        Includes integrity checks and trust verification.
        """
        
        with self._lock:
            verification_start = datetime.now(timezone.utc)
            
            try:
                # Pre-verification security checks
                self._perform_pre_verification_checks(packet, expected_from)
                
                # Decrypt context
                decrypted_packet = self.encryption_engine.decrypt_context_packet(
                    packet, self.agent_identity
                )
                
                # Post-verification integrity checks
                self._perform_post_verification_checks(decrypted_packet)
                
                # Record successful verification
                verification_time = (datetime.now(timezone.utc) - verification_start).total_seconds()
                self._record_security_event(
                    event_type=SecurityEventType.DECRYPTION_SUCCESS,
                    agent_from=packet.from_agent.agent_id,
                    agent_to=self.agent_identity.agent_id,
                    context_type=packet.context_type,
                    encryption_level=EncryptionLevel(packet.metadata.encryption_level),
                    details={"verification_time": verification_time},
                    outcome="success"
                )
                
                self.security_metrics.successful_verifications += 1
                self._update_average_response_time(verification_time)
                
                return decrypted_packet
                
            except Exception as e:
                # Record verification failure
                verification_time = (datetime.now(timezone.utc) - verification_start).total_seconds()
                self._record_security_event(
                    event_type=SecurityEventType.DECRYPTION_FAILURE,
                    agent_from=packet.from_agent.agent_id,
                    agent_to=self.agent_identity.agent_id,
                    context_type=packet.context_type,
                    details={"error": str(e), "verification_time": verification_time},
                    outcome="failure",
                    impact_score=0.6
                )
                
                self.security_metrics.failed_verifications += 1
                self._update_threat_assessment_from_failure(packet, str(e))
                
                raise
    
    def update_trust_relationship(self, 
                                agent_id: str, 
                                context_type: ContextType,
                                new_trust_level: float,
                                collaboration_outcome: CollaborationOutcome):
        """
        Update trust relationship and adapt security measures accordingly.
        """
        
        with self._lock:
            # Track trust changes for learning
            trust_key = (self.agent_identity.agent_id, agent_id)
            self.trust_deltas[trust_key].append(new_trust_level)
            
            # Keep only recent trust changes
            if len(self.trust_deltas[trust_key]) > 100:
                self.trust_deltas[trust_key] = self.trust_deltas[trust_key][-50:]
            
            # Check for significant trust changes
            if len(self.trust_deltas[trust_key]) > 1:
                trust_change = new_trust_level - self.trust_deltas[trust_key][-2]
                
                if abs(trust_change) > 0.2:  # Significant change threshold
                    self._record_security_event(
                        event_type=SecurityEventType.TRUST_THRESHOLD_BREACH,
                        agent_from=self.agent_identity.agent_id,
                        agent_to=agent_id,
                        context_type=context_type,
                        trust_level=new_trust_level,
                        details={
                            "trust_change": trust_change,
                            "collaboration_outcome": collaboration_outcome.value
                        },
                        impact_score=abs(trust_change)
                    )
            
            # Adapt security policies based on trust evolution
            self._adapt_security_for_trust_change(agent_id, context_type, new_trust_level)
    
    def assess_threat_landscape(self) -> ThreatAssessment:
        """
        Perform comprehensive threat assessment based on recent events and patterns.
        """
        
        with self._lock:
            # Analyze recent security events
            recent_events = [e for e in self.security_events 
                           if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600]
            
            # Calculate threat indicators
            failure_rate = len([e for e in recent_events 
                              if e.outcome == "failure"]) / max(len(recent_events), 1)
            
            # Assess overall threat level
            if failure_rate > 0.1:
                overall_threat = ThreatLevel.HIGH
            elif failure_rate > 0.05:
                overall_threat = ThreatLevel.MODERATE
            else:
                overall_threat = ThreatLevel.LOW
            
            # Context-specific threat analysis
            context_threats = {}
            for context_type in ContextType:
                context_events = [e for e in recent_events 
                                if e.context_type == context_type]
                if context_events:
                    context_failure_rate = len([e for e in context_events 
                                              if e.outcome == "failure"]) / len(context_events)
                    if context_failure_rate > 0.1:
                        context_threats[context_type] = ThreatLevel.HIGH
                    elif context_failure_rate > 0.05:
                        context_threats[context_type] = ThreatLevel.MODERATE
                    else:
                        context_threats[context_type] = ThreatLevel.LOW
            
            # Update threat assessment
            self.threat_assessment = ThreatAssessment(
                overall_threat_level=overall_threat,
                context_specific_threats=context_threats,
                recent_incidents=[e for e in recent_events if e.impact_score > 0.5],
                confidence_score=min(1.0, len(recent_events) / 50.0)  # More events = higher confidence
            )
            
            return self.threat_assessment
    
    def get_security_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights about security system performance and patterns.
        """
        
        with self._lock:
            # Update threat assessment
            current_threats = self.assess_threat_landscape()
            
            # Analyze learned patterns
            pattern_insights = self._analyze_learned_patterns()
            
            # Calculate trust evolution trends
            trust_trends = self._analyze_trust_trends()
            
            return {
                "security_metrics": {
                    "success_rate": self.security_metrics.get_success_rate(),
                    "threat_detection_accuracy": self.security_metrics.get_threat_detection_accuracy(),
                    "total_contexts_secured": self.security_metrics.total_contexts_secured,
                    "average_response_time": self.security_metrics.average_response_time,
                    "policy_violations": self.security_metrics.policy_violations
                },
                "threat_assessment": {
                    "overall_threat_level": current_threats.overall_threat_level.value,
                    "confidence_score": current_threats.confidence_score,
                    "recent_incidents": len(current_threats.recent_incidents)
                },
                "learned_patterns": pattern_insights,
                "trust_trends": trust_trends,
                "security_adaptations": len(self.security_adaptations),
                "recent_events": len([e for e in self.security_events 
                                    if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600])
            }
    
    def register_event_handler(self, event_type: SecurityEventType, handler: Callable[[SecurityEvent], None]):
        """Register handler for specific security events"""
        self.event_handlers[event_type].append(handler)
    
    def set_security_policy(self, policy: SecurityPolicy):
        """Update security policy with transition handling"""
        
        with self._lock:
            old_policy = self.security_policy
            self.security_policy = policy
            
            self._record_security_event(
                event_type=SecurityEventType.SECURITY_UPGRADE if policy.value > old_policy.value 
                          else SecurityEventType.SECURITY_DOWNGRADE,
                details={
                    "old_policy": old_policy.value,
                    "new_policy": policy.value
                },
                impact_score=0.3
            )
    
    # Private helper methods
    
    def _make_security_decision(self, 
                              packet: ContextPacket,
                              target_agent: str, 
                              trust_level: float,
                              threat_level: ThreatLevel,
                              policy: SecurityPolicy) -> Dict[str, Any]:
        """Make intelligent security decision based on all factors"""
        
        # Base decision on policy
        if policy == SecurityPolicy.ALWAYS_ENCRYPT:
            base_encryption = EncryptionLevel.ASYMMETRIC
        elif policy == SecurityPolicy.MINIMAL_SECURITY:
            base_encryption = EncryptionLevel.OBFUSCATED
        else:  # TRUST_BASED, REGULATORY_COMPLIANCE, ADAPTIVE_LEARNING
            base_encryption = self.encryption_engine._get_trust_based_encryption(trust_level)
        
        # Adjust for threat level
        if threat_level == ThreatLevel.CRITICAL:
            base_encryption = self.encryption_engine._increase_encryption_level(base_encryption, 2)
        elif threat_level == ThreatLevel.HIGH:
            base_encryption = self.encryption_engine._increase_encryption_level(base_encryption, 1)
        
        # Check learned patterns
        pattern_key = f"{target_agent}_{packet.context_type.value}"
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            if pattern.get("success_rate", 0) > 0.8:
                # High success rate - can potentially downgrade security
                base_encryption = max(base_encryption, EncryptionLevel.SYMMETRIC)
        
        return {
            "encryption_level": base_encryption,
            "policy_applied": policy.value,
            "trust_level": trust_level,
            "threat_level": threat_level.value,
            "decision_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _apply_security_measures(self, packet: ContextPacket, security_decision: Dict[str, Any]) -> ContextPacket:
        """Apply determined security measures to context packet"""
        
        encryption_level = security_decision["encryption_level"]
        trust_level = security_decision["trust_level"]
        
        # Use the encryption engine to secure the packet
        secured_packet = self.encryption_engine.encrypt_context_packet(packet, trust_level)
        
        # Add security decision metadata
        secured_packet.metadata.encryption_level = encryption_level.value
        secured_packet.metadata.add_to_lineage(
            self.agent_identity.agent_id,
            f"security_applied_{encryption_level.value}"
        )
        
        return secured_packet
    
    def _perform_pre_verification_checks(self, packet: ContextPacket, expected_from: Optional[str]):
        """Perform security checks before decryption"""
        
        # Check packet integrity
        if packet.is_expired():
            raise ValueError("Context packet has expired")
        
        # Verify sender if specified
        if expected_from and packet.from_agent.agent_id != expected_from:
            raise ValueError(f"Unexpected sender: {packet.from_agent.agent_id}")
        
        # Check if we have permission to decrypt
        if not self._can_decrypt_context(packet):
            raise PermissionError(f"No permission to decrypt context type {packet.context_type.value}")
    
    def _perform_post_verification_checks(self, decrypted_packet: ContextPacket):
        """Perform integrity checks after decryption"""
        
        # Verify payload integrity
        if not decrypted_packet.payload:
            raise ValueError("Decrypted payload is empty")
        
        # Check for suspicious patterns
        if self._detect_suspicious_patterns(decrypted_packet):
            self._record_security_event(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                context_type=decrypted_packet.context_type,
                details={"payload_keys": list(decrypted_packet.payload.keys())},
                threat_level=ThreatLevel.MODERATE,
                impact_score=0.4
            )
    
    def _can_decrypt_context(self, packet: ContextPacket) -> bool:
        """Check if current agent can decrypt this context"""
        
        # Check if we're the intended recipient
        if packet.to_agent == self.agent_identity.agent_id:
            return True
        
        # Check capabilities
        required_capability = f"decrypt_{packet.context_type.value}"
        return required_capability in self.agent_identity.capabilities
    
    def _detect_suspicious_patterns(self, packet: ContextPacket) -> bool:
        """Detect potentially suspicious patterns in context"""
        
        # Check for known suspicious indicators
        payload_str = json.dumps(packet.payload).lower()
        suspicious_keywords = ["inject", "exploit", "malicious", "override", "bypass"]
        
        return any(keyword in payload_str for keyword in suspicious_keywords)
    
    def _record_security_event(self, 
                             event_type: SecurityEventType,
                             agent_from: Optional[str] = None,
                             agent_to: Optional[str] = None,
                             context_type: Optional[ContextType] = None,
                             trust_level: Optional[float] = None,
                             encryption_level: Optional[EncryptionLevel] = None,
                             threat_level: ThreatLevel = ThreatLevel.MINIMAL,
                             details: Dict[str, Any] = None,
                             outcome: Optional[str] = None,
                             impact_score: float = 0.0):
        """Record security event for audit and learning"""

     # At the end of _record_security_event method:

     # Track security events as story if available
     if self.story_keeper and event.impact_score > 0.3:  # Only significant events
        self.story_keeper.process_interaction(
        user_input=f"[SECURITY] {event.event_type.value}: {event.agent_from or 'system'} → {event.agent_to or 'system'}",
        agent_response=f"Outcome: {event.outcome}, Threat: {event.threat_level.name}",
        metadata={
            "emotional_gravity": event.impact_score,
            "is_security_event": True,
            "threat_level": event.threat_level.value,
            "event_type": event.event_type.value
        }
    )
                                 
        
        event = SecurityEvent(
            event_type=event_type,
            agent_from=agent_from,
            agent_to=agent_to,
            context_type=context_type,
            trust_level=trust_level,
            encryption_level=encryption_level,
            threat_level=threat_level,
            details=details or {},
            outcome=outcome,
            impact_score=impact_score
        )
        
        self.security_events.append(event)
        
        # Trigger registered event handlers
        for handler in self.event_handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                # Don't let handler failures break security operations
                print(f"Event handler failed: {e}")
    
    def _learn_from_security_decision(self, 
                                    packet: ContextPacket, 
                                    target_agent: str,
                                    trust_level: float, 
                                    security_decision: Dict[str, Any]):
        """Learn from security decisions to improve future choices"""
        
        pattern_key = f"{target_agent}_{packet.context_type.value}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "total_decisions": 0,
                "successful_decisions": 0,
                "average_trust": 0.0,
                "common_encryption": None,
                "last_updated": datetime.now(timezone.utc)
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["total_decisions"] += 1
        pattern["successful_decisions"] += 1  # Assume success until proven otherwise
        pattern["average_trust"] = (pattern["average_trust"] * (pattern["total_decisions"] - 1) + trust_level) / pattern["total_decisions"]
        pattern["last_updated"] = datetime.now(timezone.utc)
    
    def _adapt_security_for_trust_change(self, agent_id: str, context_type: ContextType, new_trust_level: float):
        """Adapt security measures based on trust evolution"""
        
        adaptation_key = f"{agent_id}_{context_type.value}"
        
        if adaptation_key not in self.security_adaptations:
            self.security_adaptations[adaptation_key] = {
                "trust_history": [],
                "security_adjustments": [],
                "last_adaptation": None
            }
        
        adaptation = self.security_adaptations[adaptation_key]
        adaptation["trust_history"].append({
            "trust_level": new_trust_level,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Keep only recent history
        if len(adaptation["trust_history"]) > 50:
            adaptation["trust_history"] = adaptation["trust_history"][-30:]
    
    def _update_threat_assessment_from_failure(self, packet: ContextPacket, error: str):
        """Update threat assessment based on security failure"""
        
        # Increase threat level for this agent
        agent_id = packet.from_agent.agent_id
        current_threat = self.threat_assessment.agent_specific_threats.get(agent_id, ThreatLevel.LOW)
        
        # Escalate threat level
        if "permission" in error.lower():
            new_threat = ThreatLevel.MODERATE
        elif "expired" in error.lower():
            new_threat = ThreatLevel.LOW
        else:
            new_threat = ThreatLevel.HIGH
        
        self.threat_assessment.agent_specific_threats[agent_id] = ThreatLevel(max(current_threat.value, new_threat.value))
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time metric"""
        
        if self.security_metrics.average_response_time == 0:
            self.security_metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.security_metrics.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.security_metrics.average_response_time
            )
    
    def _analyze_learned_patterns(self) -> Dict[str, Any]:
        """Analyze learned patterns for insights"""
        
        if not self.learned_patterns:
            return {"total_patterns": 0}
        
        total_patterns = len(self.learned_patterns)
        high_success_patterns = len([p for p in self.learned_patterns.values() 
                                   if p.get("successful_decisions", 0) / max(p.get("total_decisions", 1), 1) > 0.8])
        
        return {
            "total_patterns": total_patterns,
            "high_success_patterns": high_success_patterns,
            "success_rate": high_success_patterns / total_patterns if total_patterns > 0 else 0
        }
    
    def _analyze_trust_trends(self) -> Dict[str, Any]:
        """Analyze trust evolution trends"""
        
        trends = {}
        for trust_key, trust_history in self.trust_deltas.items():
            if len(trust_history) < 2:
                continue
                
            recent_trend = trust_history[-5:]  # Last 5 measurements
            if len(recent_trend) >= 2:
                trend_direction = "stable"
                avg_change = sum(recent_trend[i] - recent_trend[i-1] for i in range(1, len(recent_trend))) / (len(recent_trend) - 1)
                
                if avg_change > 0.05:
                    trend_direction = "increasing"
                elif avg_change < -0.05:
                    trend_direction = "decreasing"
                
                agent_pair = f"{trust_key[0]}-{trust_key[1]}"
                trends[agent_pair] = {
                    "direction": trend_direction,
                    "current_level": trust_history[-1],
                    "change_rate": avg_change
                }
        
        return trends
    
    def _initialize_threat_baselines(self):
        """Initialize baseline threat levels"""
        
        # Set default threat levels for different context types
        baseline_threats = {
            ContextType.EMOTIONAL_STATE: ThreatLevel.MODERATE,
            ContextType.TASK_KNOWLEDGE: ThreatLevel.LOW,
            ContextType.CAPABILITY_STATUS: ThreatLevel.LOW,
            ContextType.TRUST_SIGNAL: ThreatLevel.MODERATE,
            ContextType.HANDOFF_REQUEST: ThreatLevel.MODERATE,
            ContextType.SYSTEM_STATE: ThreatLevel.HIGH,
            ContextType.USER_PREFERENCE: ThreatLevel.HIGH
        }
        
        self.threat_assessment.context_specific_threats = baseline_threats


# Example usage and testing
if __name__ == "__main__":
    from ..context_share.schemas import Priority
    
    # Create security manager for an agent
    agent = AgentIdentity(
        agent_id="security-test-agent",
        agent_type="test_agent",
        version="1.0.0",
        capabilities=["natural_language", "decrypt_task_knowledge", "decrypt_emotional_state"]
    )
    
    security_manager = ContextSecurityManager(agent, SecurityPolicy.TRUST_BASED)
    
    # Create a test context packet
    test_packet = ContextPacket(
        from_agent=agent,
        to_agent="target-agent",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "task": "customer_support",
            "priority": "high",
            "sensitive_data": "customer_info"
        },
        metadata=ContextMetadata(),
        trust_required=TrustLevel.BUILDING,
        priority=Priority.HIGH
    )
    
    # Test securing context
    try:
        secured_packet = security_manager.secure_context_for_sharing(
            test_packet,
            target_agent="target-agent",
            trust_level=0.7
        )
        
        print("Context secured successfully!")
        print(f"Original encryption: {test_packet.metadata.encryption_level}")
        print(f"Secured encryption: {secured_packet.metadata.encryption_level}")
        print(f"Lineage length: {len(secured_packet.metadata.lineage)}")
        
        # Test verification and decryption
        try:
            decrypted_packet = security_manager.verify_and_decrypt_context(secured_packet)
            print("Context verified and decrypted successfully!")
            print(f"Decrypted payload keys: {list(decrypted_packet.payload.keys())}")
        except Exception as decrypt_error:
            print(f"Decryption failed: {decrypt_error}")
        
        # Test trust relationship update
        security_manager.update_trust_relationship(
            agent_id="target-agent",
            context_type=ContextType.TASK_KNOWLEDGE,
            new_trust_level=0.8,
            collaboration_outcome=CollaborationOutcome.POSITIVE
        )
        
        # Get security insights
        insights = security_manager.get_security_insights()
        print("\nSecurity Insights:")
        print(json.dumps(insights, indent=2, default=str))
        
        # Test threat assessment
        threat_assessment = security_manager.assess_threat_landscape()
        print(f"\nThreat Assessment:")
        print(f"Overall threat level: {threat_assessment.overall_threat_level.value}")
        print(f"Confidence score: {threat_assessment.confidence_score}")
        
        # Test event handler registration
        def security_event_handler(event: SecurityEvent):
            print(f"Security event: {event.event_type.value} - {event.outcome}")
        
        security_manager.register_event_handler(
            SecurityEventType.CONTEXT_SECURED,
            security_event_handler
        )
        
        # Test another security operation to trigger handler
        test_packet2 = ContextPacket(
            from_agent=agent,
            to_agent="another-agent",
            context_type=ContextType.EMOTIONAL_STATE,
            payload={"mood": "focused", "stress_level": "low"},
            metadata=ContextMetadata(),
            trust_required=TrustLevel.STRONG,
            priority=Priority.NORMAL
        )
        
        secured_packet2 = security_manager.secure_context_for_sharing(
            test_packet2,
            target_agent="another-agent", 
            trust_level=0.9
        )
        
        print(f"\nSecond packet secured with trust level 0.9")
        print(f"Encryption level: {secured_packet2.metadata.encryption_level}")
        
    except Exception as e:
        print(f"Security operation failed: {e}")
        import traceback
        traceback.print_exc()
