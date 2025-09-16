"""
PACT-AX: Agent Collaboration Layer
Context Share Encryption

Provides trust-aware encryption for organic agent collaboration.
Balances security with the fluidity needed for natural collaboration patterns.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import secrets
import base64
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.backends import default_backend

from .schemas import ContextPacket, TrustLevel, AgentIdentity, ContextType


class EncryptionLevel(Enum):
    """Encryption levels based on context sensitivity and trust"""
    NONE = "none"               # Plain text, full trust
    OBFUSCATED = "obfuscated"   # Simple obfuscation for casual privacy
    SYMMETRIC = "symmetric"      # Shared key encryption
    ASYMMETRIC = "asymmetric"   # Public key encryption
    MULTI_LAYER = "multi_layer" # Multiple encryption layers
    QUANTUM_SAFE = "quantum_safe" # Post-quantum cryptography


class KeyRotationPolicy(Enum):
    """How often encryption keys should be rotated"""
    NEVER = auto()
    DAILY = auto()
    HOURLY = auto()
    PER_CONTEXT = auto()  # New key for each context type
    ADAPTIVE = auto()     # Based on trust evolution


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata"""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    associated_agents: List[str] = field(default_factory=list)
    trust_threshold: float = 0.5
    
    def is_valid(self) -> bool:
        """Check if key is still valid"""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        if self.max_usage and self.usage_count >= self.max_usage:
            return False
        
        return True
    
    def can_use_for_trust_level(self, trust_level: float) -> bool:
        """Check if key is appropriate for given trust level"""
        return trust_level >= self.trust_threshold
    
    def increment_usage(self):
        """Track key usage"""
        self.usage_count += 1


@dataclass
class EncryptionContext:
    """Context for encryption decisions"""
    source_agent: AgentIdentity
    target_agent: str
    context_type: ContextType
    trust_level: float
    sensitivity_score: float
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    environment_risk: str = "low"  # low, medium, high
    regulatory_requirements: List[str] = field(default_factory=list)


class TrustAwareEncryption:
    """
    Encryption system that adapts based on trust relationships and context sensitivity.
    Higher trust = lighter encryption, Lower trust = stronger encryption
    """
    
    def __init__(self):
        self.agent_keys: Dict[str, Dict[str, EncryptionKey]] = {}
        self.shared_keys: Dict[Tuple[str, str], EncryptionKey] = {}
        self.rotation_policies: Dict[str, KeyRotationPolicy] = {}
        self.trust_thresholds: Dict[EncryptionLevel, float] = {
            EncryptionLevel.NONE: 0.9,
            EncryptionLevel.OBFUSCATED: 0.7,
            EncryptionLevel.SYMMETRIC: 0.5,
            EncryptionLevel.ASYMMETRIC: 0.3,
            EncryptionLevel.MULTI_LAYER: 0.1,
            EncryptionLevel.QUANTUM_SAFE: 0.0
        }
    
    def determine_encryption_level(self, encryption_context: EncryptionContext) -> EncryptionLevel:
        """
        Intelligently determine appropriate encryption level based on:
        - Trust relationship between agents
        - Context sensitivity 
        - Environmental factors
        - Regulatory requirements
        """
        
        # Start with trust-based recommendation
        base_level = self._get_trust_based_encryption(encryption_context.trust_level)
        
        # Adjust for context sensitivity
        if encryption_context.sensitivity_score > 0.8:
            base_level = self._increase_encryption_level(base_level, 2)
        elif encryption_context.sensitivity_score > 0.6:
            base_level = self._increase_encryption_level(base_level, 1)
        
        # Adjust for environment risk
        if encryption_context.environment_risk == "high":
            base_level = self._increase_encryption_level(base_level, 2)
        elif encryption_context.environment_risk == "medium":
            base_level = self._increase_encryption_level(base_level, 1)
        
        # Handle regulatory requirements
        if "HIPAA" in encryption_context.regulatory_requirements:
            base_level = max(base_level, EncryptionLevel.ASYMMETRIC)
        if "GDPR" in encryption_context.regulatory_requirements:
            base_level = max(base_level, EncryptionLevel.SYMMETRIC)
        if "quantum_safe" in encryption_context.regulatory_requirements:
            base_level = EncryptionLevel.QUANTUM_SAFE
        
        return base_level
    
    def encrypt_context_packet(self, packet: ContextPacket, target_trust_level: float) -> ContextPacket:
        """Encrypt context packet based on trust and sensitivity"""
        
        encryption_context = EncryptionContext(
            source_agent=packet.from_agent,
            target_agent=packet.to_agent,
            context_type=packet.context_type,
            trust_level=target_trust_level,
            sensitivity_score=self._calculate_sensitivity_score(packet.payload, packet.context_type)
        )
        
        encryption_level = self.determine_encryption_level(encryption_context)
        
        if encryption_level == EncryptionLevel.NONE:
            # No encryption needed - high trust relationship
            packet.metadata.encryption_level = "none"
            return packet
        
        # Encrypt the payload
        encrypted_payload, encryption_metadata = self._encrypt_payload(
            packet.payload, 
            encryption_level,
            encryption_context
        )
        
        # Update packet
        packet.payload = encrypted_payload
        packet.metadata.encryption_level = encryption_level.value
        packet.metadata.checksum = self._calculate_checksum(encrypted_payload)
        
        # Add encryption info to lineage
        packet.metadata.add_to_lineage(
            packet.from_agent.agent_id,
            f"encrypted_{encryption_level.value}"
        )
        
        return packet
    
    def decrypt_context_packet(self, packet: ContextPacket, agent_identity: AgentIdentity) -> ContextPacket:
        """Decrypt context packet for receiving agent"""
        
        if packet.metadata.encryption_level == "none":
            return packet
        
        encryption_level = EncryptionLevel(packet.metadata.encryption_level)
        
        # Verify agent has permission to decrypt
        if not self._can_agent_decrypt(agent_identity, packet):
            raise PermissionError(f"Agent {agent_identity.agent_id} cannot decrypt this context")
        
        # Decrypt payload
        decrypted_payload = self._decrypt_payload(
            packet.payload,
            encryption_level,
            packet.from_agent,
            agent_identity
        )
        
        # Verify integrity
        if packet.metadata.checksum:
            expected_checksum = self._calculate_checksum(packet.payload)
            if expected_checksum != packet.metadata.checksum:
                raise ValueError("Context packet integrity check failed")
        
        # Update packet
        packet.payload = decrypted_payload
        packet.metadata.add_to_lineage(
            agent_identity.agent_id,
            f"decrypted_{encryption_level.value}"
        )
        
        return packet
    
    def generate_shared_key(self, agent1: str, agent2: str, context_type: ContextType) -> EncryptionKey:
        """Generate shared encryption key for two agents"""
        
        key_id = f"shared_{agent1}_{agent2}_{context_type.value}_{secrets.token_hex(8)}"
        key_data = secrets.token_bytes(32)  # 256-bit key
        
        shared_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm="AES-256-GCM",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            max_usage=1000,
            associated_agents=[agent1, agent2],
            trust_threshold=0.5
        )
        
        # Store bidirectionally
        self.shared_keys[(agent1, agent2)] = shared_key
        self.shared_keys[(agent2, agent1)] = shared_key
        
        return shared_key
    
    def rotate_keys_if_needed(self, agent_pair: Tuple[str, str], force: bool = False):
        """Rotate encryption keys based on policy"""
        
        if agent_pair not in self.shared_keys:
            return
        
        key = self.shared_keys[agent_pair]
        should_rotate = force or not key.is_valid()
        
        # Check rotation policy
        policy = self.rotation_policies.get(f"{agent_pair[0]}_{agent_pair[1]}", KeyRotationPolicy.DAILY)
        
        if policy == KeyRotationPolicy.HOURLY:
            if datetime.now(timezone.utc) - key.created_at > timedelta(hours=1):
                should_rotate = True
        elif policy == KeyRotationPolicy.DAILY:
            if datetime.now(timezone.utc) - key.created_at > timedelta(days=1):
                should_rotate = True
        
        if should_rotate:
            # Generate new key
            new_key = self.generate_shared_key(
                agent_pair[0], 
                agent_pair[1], 
                ContextType.TASK_KNOWLEDGE  # Default context
            )
            
            # Update key references
            self.shared_keys[agent_pair] = new_key
            self.shared_keys[(agent_pair[1], agent_pair[0])] = new_key
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get statistics about encryption usage and trust patterns"""
        
        stats = {
            "total_shared_keys": len(self.shared_keys),
            "encryption_levels_used": {},
            "trust_based_decisions": {
                "high_trust_plain": 0,
                "medium_trust_symmetric": 0,
                "low_trust_asymmetric": 0
            },
            "key_rotation_events": 0,
            "failed_decryptions": 0
        }
        
        # Analyze encryption level usage
        for key in self.shared_keys.values():
            if key.algorithm not in stats["encryption_levels_used"]:
                stats["encryption_levels_used"][key.algorithm] = 0
            stats["encryption_levels_used"][key.algorithm] += key.usage_count
        
        return stats
    
    # Private helper methods
    
    def _get_trust_based_encryption(self, trust_level: float) -> EncryptionLevel:
        """Get encryption level based on trust"""
        for level, threshold in sorted(self.trust_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if trust_level >= threshold:
                return level
        return EncryptionLevel.QUANTUM_SAFE  # Lowest trust
    
    def _increase_encryption_level(self, current: EncryptionLevel, steps: int) -> EncryptionLevel:
        """Increase encryption level by specified steps"""
        levels = list(EncryptionLevel)
        current_index = levels.index(current)
        new_index = min(len(levels) - 1, current_index + steps)
        return levels[new_index]
    
    def _calculate_sensitivity_score(self, payload: Dict[str, Any], context_type: ContextType) -> float:
        """Calculate sensitivity score for payload"""
        
        base_sensitivity = {
            ContextType.EMOTIONAL_STATE: 0.8,
            ContextType.TASK_KNOWLEDGE: 0.4,
            ContextType.CAPABILITY_STATUS: 0.3,
            ContextType.TRUST_SIGNAL: 0.6,
            ContextType.HANDOFF_REQUEST: 0.5,
            ContextType.SYSTEM_STATE: 0.7,
            ContextType.USER_PREFERENCE: 0.9,
            ContextType.LEARNING_INSIGHT: 0.2
        }.get(context_type, 0.5)
        
        # Check for sensitive data patterns
        payload_str = json.dumps(payload).lower()
        sensitive_keywords = [
            "password", "secret", "private", "confidential", "personal",
            "ssn", "credit_card", "phone", "email", "address", "medical"
        ]
        
        sensitivity_boost = sum(0.1 for keyword in sensitive_keywords 
                              if keyword in payload_str)
        
        return min(1.0, base_sensitivity + sensitivity_boost)
    
    def _encrypt_payload(self, payload: Dict[str, Any], level: EncryptionLevel, 
                        context: EncryptionContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Encrypt payload based on encryption level"""
        
        payload_bytes = json.dumps(payload).encode('utf-8')
        
        if level == EncryptionLevel.OBFUSCATED:
            # Simple base64 obfuscation
            encrypted_data = base64.b64encode(payload_bytes).decode('utf-8')
            return {"encrypted_data": encrypted_data}, {"method": "base64"}
        
        elif level == EncryptionLevel.SYMMETRIC:
            # AES encryption with shared key
            key = self._get_or_create_shared_key(
                context.source_agent.agent_id, 
                context.target_agent
            )
            
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad data
            padder = sym_padding.PKCS7(128).padder()
            padded_data = padder.update(payload_bytes) + padder.finalize()
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            key.increment_usage()
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "key_id": key.key_id
            }, {"method": "AES-CBC", "key_usage": key.usage_count}
        
        elif level == EncryptionLevel.ASYMMETRIC:
            # RSA encryption (for small payloads) or hybrid approach
            return self._hybrid_encrypt(payload_bytes, context)
        
        else:  # MULTI_LAYER or QUANTUM_SAFE
            # Multiple layers of encryption
            encrypted, metadata = self._encrypt_payload(payload, EncryptionLevel.SYMMETRIC, context)
            return self._encrypt_payload(encrypted, EncryptionLevel.ASYMMETRIC, context)
    
    def _decrypt_payload(self, encrypted_payload: Dict[str, Any], level: EncryptionLevel,
                        from_agent: AgentIdentity, to_agent: AgentIdentity) -> Dict[str, Any]:
        """Decrypt payload based on encryption level"""
        
        if level == EncryptionLevel.OBFUSCATED:
            encrypted_data = encrypted_payload["encrypted_data"]
            decrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            return json.loads(decrypted_bytes.decode('utf-8'))
        
        elif level == EncryptionLevel.SYMMETRIC:
            key_id = encrypted_payload["key_id"]
            key = self._find_key(key_id)
            if not key or not key.is_valid():
                raise ValueError(f"Invalid or expired key: {key_id}")
            
            encrypted_data = base64.b64decode(encrypted_payload["encrypted_data"])
            iv = base64.b64decode(encrypted_payload["iv"])
            
            cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Unpad data
            unpadder = sym_padding.PKCS7(128).unpadder()
            payload_bytes = unpadder.update(padded_data) + unpadder.finalize()
            
            return json.loads(payload_bytes.decode('utf-8'))
        
        # Add more decryption methods as needed
        else:
            raise NotImplementedError(f"Decryption for {level} not yet implemented")
    
    def _hybrid_encrypt(self, data: bytes, context: EncryptionContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Hybrid encryption using RSA + AES"""
        
        # Generate symmetric key for data
        aes_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        # Encrypt data with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        padder = sym_padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt AES key with RSA (would need public key of target agent)
        # This is simplified - in practice would use actual RSA key management
        encrypted_key = base64.b64encode(aes_key).decode('utf-8')  # Placeholder
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
            "encrypted_key": encrypted_key,
            "iv": base64.b64encode(iv).decode('utf-8'),
            "method": "hybrid"
        }, {"method": "RSA-AES-hybrid"}
    
    def _get_or_create_shared_key(self, agent1: str, agent2: str) -> EncryptionKey:
        """Get existing shared key or create new one"""
        
        key_pair = (agent1, agent2)
        if key_pair in self.shared_keys and self.shared_keys[key_pair].is_valid():
            return self.shared_keys[key_pair]
        
        return self.generate_shared_key(agent1, agent2, ContextType.TASK_KNOWLEDGE)
    
    def _find_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Find encryption key by ID"""
        
        for key in self.shared_keys.values():
            if key.key_id == key_id:
                return key
        
        return None
    
    def _can_agent_decrypt(self, agent: AgentIdentity, packet: ContextPacket) -> bool:
        """Check if agent has permission to decrypt packet"""
        
        # Agent is the intended recipient
        if packet.to_agent == agent.agent_id:
            return True
        
        # Agent has decryption capability for this context type
        required_capability = f"decrypt_{packet.context_type.value}"
        if required_capability in agent.capabilities:
            return True
        
        return False
    
    def _calculate_checksum(self, data: Union[Dict[str, Any], bytes]) -> str:
        """Calculate checksum for integrity verification"""
        
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = data
        
        return hashlib.sha256(data_bytes).hexdigest()


class ContextSecurityManager:
    """
    High-level manager for context security that integrates with trust relationships
    """
    
    def __init__(self):
        self.encryption_engine = TrustAwareEncryption()
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
    
    def secure_context_for_sharing(self, packet: ContextPacket, trust_level: float) -> ContextPacket:
        """Secure context packet based on trust relationship"""
        
        # Log security decision
        self._log_security_event(
            event_type="encryption_decision",
            packet_id=packet.metadata.packet_id,
            from_agent=packet.from_agent.agent_id,
            to_agent=packet.to_agent,
            trust_level=trust_level,
            context_type=packet.context_type.value
        )
        
        return self.encryption_engine.encrypt_context_packet(packet, trust_level)
    
    def verify_and_decrypt_context(self, packet: ContextPacket, 
                                 receiving_agent: AgentIdentity) -> ContextPacket:
        """Verify and decrypt context for receiving agent"""
        
        try:
            decrypted_packet = self.encryption_engine.decrypt_context_packet(packet, receiving_agent)
            
            self._log_security_event(
                event_type="successful_decryption",
                packet_id=packet.metadata.packet_id,
                agent=receiving_agent.agent_id,
                encryption_level=packet.metadata.encryption_level
            )
            
            return decrypted_packet
            
        except Exception as e:
            self._log_security_event(
                event_type="decryption_failure",
                packet_id=packet.metadata.packet_id,
                agent=receiving_agent.agent_id,
                error=str(e)
            )
            raise
    
    def _log_security_event(self, event_type: str, **kwargs):
        """Log security-related events for audit"""
        
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **kwargs
        }
        
        self.audit_log.append(event)
        
        # Keep only recent events to prevent memory bloat
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-8000:]  # Keep last 8000 events


# Example usage
if __name__ == "__main__":
    from .schemas import AgentIdentity, ContextPacket, ContextType, TrustLevel, Priority, ContextMetadata
    
    # Create security manager
    security_manager = ContextSecurityManager()
    
    # Create test agents
    agent1 = AgentIdentity(
        agent_id="agent-001",
        agent_type="support_specialist",
        version="1.0.0",
        capabilities=["natural_language", "customer_support", "decrypt_task_knowledge"]
    )
    
    agent2 = AgentIdentity(
        agent_id="agent-002", 
        agent_type="technical_specialist",
        version="1.0.0",
        capabilities=["technical_analysis", "troubleshooting"]
    )
    
    # Create context packet
    packet = ContextPacket(
        from_agent=agent1,
        to_agent="agent-002",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "customer_issue": "login_problem",
            "severity": "high",
            "user_details": {"account_type": "premium"}
        },
        metadata=ContextMetadata(),
        trust_required=TrustLevel.BUILDING,
        priority=Priority.HIGH
    )
    
    # Secure the packet (medium trust level)
    trust_level = 0.6
    secured_packet = security_manager.secure_context_for_sharing(packet, trust_level)
    
    print("Original payload keys:", list(packet.payload.keys()))
    print("Secured packet encryption level:", secured_packet.metadata.encryption_level)
    print("Secured payload keys:", list(secured_packet.payload.keys()))
    
    # Decrypt for receiving agent
    try:
        decrypted_packet = security_manager.verify_and_decrypt_context(secured_packet, agent2)
        print("Decryption successful!")
        print("Decrypted payload keys:", list(decrypted_packet.payload.keys()))
    except Exception as e:
        print(f"Decryption failed: {e}")
    
    # Show encryption stats
    stats = security_manager.encryption_engine.get_encryption_stats()
    print("Encryption stats:", json.dumps(stats, indent=2))
