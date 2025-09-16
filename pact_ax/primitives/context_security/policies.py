"""
PACT-AX: Agent Collaboration Layer
Context Security Policies

Dynamic policy engine that adapts security measures based on collaboration patterns,
trust evolution, and environmental factors while maintaining organic collaboration flow.
"""

from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import json
import re
from abc import ABC, abstractmethod

# Import from sibling modules
from ..context_share.schemas import (
    ContextType, TrustLevel, AgentIdentity, ContextPacket,
    CollaborationOutcome
)
from ..context_share.encryption import EncryptionLevel


class PolicyType(Enum):
    """Types of security policies"""
    TRUST_BASED = "trust_based"
    COMPLIANCE = "compliance"
    ENVIRONMENT = "environment"
    CONTEXT_SENSITIVE = "context_sensitive"
    TEMPORAL = "temporal"
    ADAPTIVE_LEARNING = "adaptive_learning"
    RISK_BASED = "risk_based"
    AGENT_SPECIFIC = "agent_specific"


class PolicyScope(Enum):
    """Scope of policy application"""
    GLOBAL = "global"              # Applies to all contexts
    CONTEXT_TYPE = "context_type"  # Applies to specific context types
    AGENT_PAIR = "agent_pair"      # Applies to specific agent relationships
    ENVIRONMENT = "environment"    # Applies to specific environments
    TIME_WINDOW = "time_window"    # Applies during specific time periods


class PolicyPriority(Enum):
    """Policy priority levels for conflict resolution"""
    REGULATORY = 10  # Highest - legal/compliance requirements
    SECURITY = 8     # High - security-critical policies
    OPERATIONAL = 6  # Medium - operational requirements
    ADAPTIVE = 4     # Medium-Low - learned/adaptive policies
    DEFAULT = 2      # Low - default fallback policies
    LEGACY = 1       # Lowest - deprecated policies


class PolicyAction(Enum):
    """Actions that policies can mandate"""
    REQUIRE_ENCRYPTION = "require_encryption"
    MINIMUM_TRUST = "minimum_trust"
    FORBID_CONTEXT = "forbid_context"
    AUDIT_REQUIRED = "audit_required"
    KEY_ROTATION = "key_rotation"
    MULTI_FACTOR = "multi_factor"
    TIME_LIMIT = "time_limit"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class PolicyCondition:
    """Condition that must be met for policy to apply"""
    field: str              # e.g., "trust_level", "context_type", "agent_id"
    operator: str           # e.g., "equals", "greater_than", "contains"
    value: Any             # The value to compare against
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if condition is met in given context"""
        
        if self.field not in context:
            return False
        
        context_value = context[self.field]
        
        if self.operator == "equals":
            return context_value == self.value
        elif self.operator == "not_equals":
            return context_value != self.value
        elif self.operator == "greater_than":
            return context_value > self.value
        elif self.operator == "less_than":
            return context_value < self.value
        elif self.operator == "greater_equal":
            return context_value >= self.value
        elif self.operator == "less_equal":
            return context_value <= self.value
        elif self.operator == "contains":
            return self.value in context_value
        elif self.operator == "not_contains":
            return self.value not in context_value
        elif self.operator == "matches_regex":
            return re.match(str(self.value), str(context_value)) is not None
        elif self.operator == "in_list":
            return context_value in self.value
        elif self.operator == "not_in_list":
            return context_value not in self.value
        else:
            return False


@dataclass
class PolicyRule:
    """Individual policy rule with conditions and actions"""
    rule_id: str
    name: str
    description: str
    conditions: List[PolicyCondition]
    actions: Dict[PolicyAction, Any]
    priority: PolicyPriority = PolicyPriority.DEFAULT
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    success_rate: float = 1.0
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches the given context"""
        
        if not self.enabled:
            return False
        
        # All conditions must be true
        return all(condition.evaluate(context) for condition in self.conditions)
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule actions to context and return policy decisions"""
        
        if not self.matches(context):
            return {}
        
        self.usage_count += 1
        self.last_modified = datetime.now(timezone.utc)
        
        decisions = {}
        for action, value in self.actions.items():
            decisions[action.value] = value
        
        return decisions
    
    def update_success_rate(self, successful: bool):
        """Update rule success rate based on outcomes"""
        
        # Exponential moving average
        alpha = 0.1
        if successful:
            self.success_rate = alpha * 1.0 + (1 - alpha) * self.success_rate
        else:
            self.success_rate = alpha * 0.0 + (1 - alpha) * self.success_rate


@dataclass
class SecurityPolicy:
    """Complete security policy with multiple rules"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    rules: List[PolicyRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    enabled: bool = True
    
    def add_rule(self, rule: PolicyRule):
        """Add rule to policy"""
        self.rules.append(rule)
        self.last_updated = datetime.now(timezone.utc)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule from policy"""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        if len(self.rules) < initial_count:
            self.last_updated = datetime.now(timezone.utc)
            return True
        return False
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy against context and return combined decisions"""
        
        if not self.enabled:
            return {}
        
        # Apply all matching rules and combine decisions
        combined_decisions = {}
        matching_rules = []
        
        for rule in sorted(self.rules, key=lambda r: r.priority.value, reverse=True):
            if rule.matches(context):
                matching_rules.append(rule)
                rule_decisions = rule.apply(context)
                
                # Merge decisions, higher priority rules override lower priority
                for decision_key, decision_value in rule_decisions.items():
                    if decision_key not in combined_decisions:
                        combined_decisions[decision_key] = decision_value
                    else:
                        # Handle conflicts based on policy type
                        combined_decisions[decision_key] = self._resolve_decision_conflict(
                            decision_key, combined_decisions[decision_key], decision_value
                        )
        
        return {
            "decisions": combined_decisions,
            "matching_rules": [r.rule_id for r in matching_rules],
            "policy_id": self.policy_id
        }
    
    def _resolve_decision_conflict(self, decision_key: str, existing_value: Any, new_value: Any) -> Any:
        """Resolve conflicts between policy decisions"""
        
        # For encryption levels, take the higher security level
        if decision_key == "require_encryption":
            if isinstance(existing_value, str) and isinstance(new_value, str):
                existing_level = EncryptionLevel(existing_value)
                new_level = EncryptionLevel(new_value)
                levels = list(EncryptionLevel)
                return levels[max(levels.index(existing_level), levels.index(new_level))].value
        
        # For trust levels, take the higher requirement
        elif decision_key == "minimum_trust":
            return max(existing_value, new_value)
        
        # For time limits, take the shorter limit (more restrictive)
        elif decision_key == "time_limit":
            return min(existing_value, new_value)
        
        # For boolean decisions, OR them together
        elif isinstance(existing_value, bool) and isinstance(new_value, bool):
            return existing_value or new_value
        
        # Default: keep existing value (first match wins)
        return existing_value


class PolicyEngine:
    """
    Central policy engine that manages and evaluates security policies
    for PACT-AX context operations with adaptive learning capabilities.
    """
    
    def __init__(self, agent_identity: AgentIdentity):
        self.agent_identity = agent_identity
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_templates: Dict[str, Dict[str, Any]] = {}
        self.adaptive_policies: Dict[str, SecurityPolicy] = {}
        self.policy_conflicts: List[Dict[str, Any]] = []
        
        # Learning and adaptation
        self.decision_history: List[Dict[str, Any]] = []
        self.pattern_insights: Dict[str, Any] = {}
        
        # Initialize with default policies
        self._initialize_default_policies()
        self._initialize_policy_templates()
    
    def create_policy(self, policy_config: Dict[str, Any]) -> SecurityPolicy:
        """Create new security policy from configuration"""
        
        policy = SecurityPolicy(
            policy_id=policy_config["policy_id"],
            name=policy_config["name"],
            description=policy_config["description"],
            policy_type=PolicyType(policy_config["policy_type"]),
            scope=PolicyScope(policy_config["scope"]),
            metadata=policy_config.get("metadata", {}),
            version=policy_config.get("version", "1.0")
        )
        
        # Add rules
        for rule_config in policy_config.get("rules", []):
            rule = self._create_rule_from_config(rule_config)
            policy.add_rule(rule)
        
        self.policies[policy.policy_id] = policy
        return policy
    
    def evaluate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all applicable policies for given context and return security decisions
        """
        
        evaluation_start = datetime.now(timezone.utc)
        
        # Find applicable policies
        applicable_policies = self._find_applicable_policies(context)
        
        # Evaluate each policy
        policy_results = {}
        combined_decisions = {}
        all_matching_rules = []
        
        for policy in applicable_policies:
            result = policy.evaluate(context)
            policy_results[policy.policy_id] = result
            
            if result.get("decisions"):
                all_matching_rules.extend(result.get("matching_rules", []))
                
                # Combine decisions across policies
                for decision_key, decision_value in result["decisions"].items():
                    if decision_key not in combined_decisions:
                        combined_decisions[decision_key] = decision_value
                    else:
                        # Resolve conflicts between policies
                        combined_decisions[decision_key] = self._resolve_policy_conflicts(
                            decision_key, combined_decisions[decision_key], decision_value,
                            context
                        )
        
        evaluation_time = (datetime.now(timezone.utc) - evaluation_start).total_seconds()
        
        # Record decision for learning
        decision_record = {
            "context": context,
            "decisions": combined_decisions,
            "policies_evaluated": len(applicable_policies),
            "rules_matched": len(all_matching_rules),
            "evaluation_time": evaluation_time,
            "timestamp": datetime.now(timezone.utc)
        }
        self.decision_history.append(decision_record)
        
        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-800:]
        
        return {
            "decisions": combined_decisions,
            "policy_results": policy_results,
            "evaluation_metadata": {
                "policies_evaluated": len(applicable_policies),
                "rules_matched": len(all_matching_rules),
                "evaluation_time": evaluation_time
            }
        }
    
    def create_adaptive_policy(self, pattern_data: Dict[str, Any]) -> SecurityPolicy:
        """
        Create adaptive policy based on learned patterns
        """
        
        pattern_id = pattern_data["pattern_id"]
        success_rate = pattern_data.get("success_rate", 0.8)
        
        # Only create adaptive policy if pattern has high success rate
        if success_rate < 0.7:
            return None
        
        adaptive_policy = SecurityPolicy(
            policy_id=f"adaptive_{pattern_id}",
            name=f"Adaptive Policy for {pattern_id}",
            description=f"Auto-generated policy based on learned collaboration pattern",
            policy_type=PolicyType.ADAPTIVE_LEARNING,
            scope=PolicyScope.AGENT_PAIR,
            metadata={
                "learned_from": pattern_data,
                "auto_generated": True,
                "success_rate": success_rate
            }
        )
        
        # Create rule based on pattern
        conditions = []
        actions = {}
        
        if "context_type" in pattern_data:
            conditions.append(PolicyCondition(
                field="context_type",
                operator="equals",
                value=pattern_data["context_type"]
            ))
        
        if "agent_pair" in pattern_data:
            conditions.append(PolicyCondition(
                field="agent_pair",
                operator="equals", 
                value=pattern_data["agent_pair"]
            ))
        
        # Determine actions based on successful patterns
        if pattern_data.get("optimal_encryption"):
            actions[PolicyAction.REQUIRE_ENCRYPTION] = pattern_data["optimal_encryption"]
        
        if pattern_data.get("minimum_trust"):
            actions[PolicyAction.MINIMUM_TRUST] = pattern_data["minimum_trust"]
        
        adaptive_rule = PolicyRule(
            rule_id=f"adaptive_rule_{pattern_id}",
            name=f"Adaptive Rule for {pattern_id}",
            description="Auto-generated rule from collaboration patterns",
            conditions=conditions,
            actions=actions,
            priority=PolicyPriority.ADAPTIVE,
            success_rate=success_rate
        )
        
        adaptive_policy.add_rule(adaptive_rule)
        self.adaptive_policies[pattern_id] = adaptive_policy
        
        return adaptive_policy
    
    def update_policy_effectiveness(self, policy_id: str, rule_id: str, successful: bool):
        """Update policy rule effectiveness based on outcomes"""
        
        if policy_id in self.policies:
            policy = self.policies[policy_id]
            for rule in policy.rules:
                if rule.rule_id == rule_id:
                    rule.update_success_rate(successful)
                    break
        
        # Also update adaptive policies
        for adaptive_policy in self.adaptive_policies.values():
            for rule in adaptive_policy.rules:
                if rule.rule_id == rule_id:
                    rule.update_success_rate(successful)
                    break
    
    def get_compliance_policies(self, regulations: List[str]) -> List[SecurityPolicy]:
        """Get policies required for specific regulatory compliance"""
        
        compliance_policies = []
        
        for regulation in regulations:
            if regulation.upper() == "HIPAA":
                hipaa_policy = self._create_hipaa_policy()
                compliance_policies.append(hipaa_policy)
            elif regulation.upper() == "GDPR":
                gdpr_policy = self._create_gdpr_policy()
                compliance_policies.append(gdpr_policy)
            elif regulation.upper() == "SOX":
                sox_policy = self._create_sox_policy()
                compliance_policies.append(sox_policy)
        
        return compliance_policies
    
    def analyze_policy_patterns(self) -> Dict[str, Any]:
        """Analyze policy decision patterns for insights"""
        
        if len(self.decision_history) < 10:
            return {"insufficient_data": True}
        
        # Analyze decision frequencies
        decision_frequencies = {}
        for record in self.decision_history:
            for decision_key in record["decisions"]:
                if decision_key not in decision_frequencies:
                    decision_frequencies[decision_key] = 0
                decision_frequencies[decision_key] += 1
        
        # Analyze evaluation performance
        evaluation_times = [r["evaluation_time"] for r in self.decision_history]
        avg_evaluation_time = sum(evaluation_times) / len(evaluation_times)
        
        # Find most effective rules
        rule_effectiveness = {}
        for policy in self.policies.values():
            for rule in policy.rules:
                rule_effectiveness[rule.rule_id] = {
                    "usage_count": rule.usage_count,
                    "success_rate": rule.success_rate,
                    "effectiveness_score": rule.usage_count * rule.success_rate
                }
        
        return {
            "total_decisions": len(self.decision_history),
            "decision_frequencies": decision_frequencies,
            "average_evaluation_time": avg_evaluation_time,
            "active_policies": len([p for p in self.policies.values() if p.enabled]),
            "adaptive_policies": len(self.adaptive_policies),
            "rule_effectiveness": rule_effectiveness,
            "policy_conflicts": len(self.policy_conflicts)
        }
    
    def export_policy_configuration(self) -> Dict[str, Any]:
        """Export current policy configuration"""
        
        config = {
            "agent_id": self.agent_identity.agent_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "policies": {},
            "adaptive_policies": {},
            "templates": self.policy_templates
        }
        
        # Export regular policies
        for policy_id, policy in self.policies.items():
            config["policies"][policy_id] = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "policy_type": policy.policy_type.value,
                "scope": policy.scope.value,
                "enabled": policy.enabled,
                "version": policy.version,
                "metadata": policy.metadata,
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "description": rule.description,
                        "conditions": [
                            {
                                "field": cond.field,
                                "operator": cond.operator,
                                "value": cond.value,
                                "metadata": cond.metadata
                            }
                            for cond in rule.conditions
                        ],
                        "actions": {action.value: value for action, value in rule.actions.items()},
                        "priority": rule.priority.value,
                        "enabled": rule.enabled,
                        "usage_count": rule.usage_count,
                        "success_rate": rule.success_rate
                    }
                    for rule in policy.rules
                ]
            }
        
        # Export adaptive policies
        for pattern_id, policy in self.adaptive_policies.items():
            config["adaptive_policies"][pattern_id] = {
                "pattern_id": pattern_id,
                "policy_id": policy.policy_id,
                "success_rate": policy.metadata.get("success_rate", 0.0),
                "learned_from": policy.metadata.get("learned_from", {}),
                "rules_count": len(policy.rules)
            }
        
        return config
    
    def import_policy_configuration(self, config: Dict[str, Any]):
        """Import policy configuration"""
        
        # Import regular policies
        for policy_data in config.get("policies", {}).values():
            try:
                policy = self.create_policy(policy_data)
                print(f"Imported policy: {policy.name}")
            except Exception as e:
                print(f"Failed to import policy {policy_data.get('name', 'unknown')}: {e}")
        
        # Import templates
        if "templates" in config:
            self.policy_templates.update(config["templates"])
    
    # Private helper methods
    
    def _find_applicable_policies(self, context: Dict[str, Any]) -> List[SecurityPolicy]:
        """Find policies applicable to given context"""
        
        applicable = []
        
        # Check all regular policies
        for policy in self.policies.values():
            if policy.enabled and self._policy_applies_to_context(policy, context):
                applicable.append(policy)
        
        # Check adaptive policies
        for policy in self.adaptive_policies.values():
            if policy.enabled and self._policy_applies_to_context(policy, context):
                applicable.append(policy)
        
        # Sort by policy type priority
        type_priority = {
            PolicyType.COMPLIANCE: 10,
            PolicyType.RISK_BASED: 8,
            PolicyType.TRUST_BASED: 6,
            PolicyType.CONTEXT_SENSITIVE: 4,
            PolicyType.ADAPTIVE_LEARNING: 2
        }
        
        applicable.sort(key=lambda p: type_priority.get(p.policy_type, 1), reverse=True)
        return applicable
    
    def _policy_applies_to_context(self, policy: SecurityPolicy, context: Dict[str, Any]) -> bool:
        """Check if policy applies to given context based on scope"""
        
        if policy.scope == PolicyScope.GLOBAL:
            return True
        elif policy.scope == PolicyScope.CONTEXT_TYPE:
            return context.get("context_type") in policy.metadata.get("context_types", [])
        elif policy.scope == PolicyScope.AGENT_PAIR:
            agent_pair = f"{context.get('from_agent')}_{context.get('to_agent')}"
            return agent_pair in policy.metadata.get("agent_pairs", [])
        elif policy.scope == PolicyScope.ENVIRONMENT:
            return context.get("environment") in policy.metadata.get("environments", [])
        elif policy.scope == PolicyScope.TIME_WINDOW:
            current_time = datetime.now(timezone.utc).time()
            start_time = policy.metadata.get("start_time")
            end_time = policy.metadata.get("end_time")
            if start_time and end_time:
                return start_time <= current_time <= end_time
        
        return True  # Default to applicable if scope not specified
    
    def _resolve_policy_conflicts(self, decision_key: str, value1: Any, value2: Any, context: Dict[str, Any]) -> Any:
        """Resolve conflicts between policy decisions"""
        
        # Record the conflict for analysis
        conflict = {
            "decision_key": decision_key,
            "value1": value1,
            "value2": value2,
            "context": context,
            "timestamp": datetime.now(timezone.utc),
            "resolution": None
        }
        
        # Apply resolution logic
        if decision_key == "require_encryption":
            # Higher encryption level wins
            try:
                level1 = EncryptionLevel(value1)
                level2 = EncryptionLevel(value2)
                levels = list(EncryptionLevel)
                resolution = levels[max(levels.index(level1), levels.index(level2))].value
            except:
                resolution = value1  # Fallback
        elif decision_key == "minimum_trust":
            # Higher trust requirement wins
            resolution = max(value1, value2)
        elif decision_key == "time_limit":
            # Shorter time limit wins (more restrictive)
            resolution = min(value1, value2)
        else:
            # Default: first value wins
            resolution = value1
        
        conflict["resolution"] = resolution
        self.policy_conflicts.append(conflict)
        
        # Keep only recent conflicts
        if len(self.policy_conflicts) > 100:
            self.policy_conflicts = self.policy_conflicts[-80:]
        
        return resolution
    
    def _create_rule_from_config(self, rule_config: Dict[str, Any]) -> PolicyRule:
        """Create policy rule from configuration"""
        
        conditions = []
        for cond_config in rule_config.get("conditions", []):
            condition = PolicyCondition(
                field=cond_config["field"],
                operator=cond_config["operator"],
                value=cond_config["value"],
                metadata=cond_config.get("metadata", {})
            )
            conditions.append(condition)
        
        actions = {}
        for action_key, action_value in rule_config.get("actions", {}).items():
            action_enum = PolicyAction(action_key)
            actions[action_enum] = action_value
        
        return PolicyRule(
            rule_id=rule_config["rule_id"],
            name=rule_config["name"],
            description=rule_config["description"],
            conditions=conditions,
            actions=actions,
            priority=PolicyPriority(rule_config.get("priority", PolicyPriority.DEFAULT.value)),
            enabled=rule_config.get("enabled", True)
        )
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        
        # Default trust-based policy
        default_policy = SecurityPolicy(
            policy_id="default_trust_based",
            name="Default Trust-Based Security",
            description="Default policy that adjusts security based on trust levels",
            policy_type=PolicyType.TRUST_BASED,
            scope=PolicyScope.GLOBAL
        )
        
        # High trust rule - minimal encryption
        high_trust_rule = PolicyRule(
            rule_id="high_trust_minimal",
            name="High Trust Minimal Security",
            description="Minimal encryption for high trust relationships",
            conditions=[
                PolicyCondition("trust_level", "greater_equal", 0.8)
            ],
            actions={
                PolicyAction.REQUIRE_ENCRYPTION: EncryptionLevel.OBFUSCATED.value
            },
            priority=PolicyPriority.DEFAULT
        )
        
        # Low trust rule - strong encryption
        low_trust_rule = PolicyRule(
            rule_id="low_trust_strong",
            name="Low Trust Strong Security", 
            description="Strong encryption for low trust relationships",
            conditions=[
                PolicyCondition("trust_level", "less_than", 0.4)
            ],
            actions={
                PolicyAction.REQUIRE_ENCRYPTION: EncryptionLevel.ASYMMETRIC.value,
                PolicyAction.AUDIT_REQUIRED: True
            },
            priority=PolicyPriority.DEFAULT
        )
        
        default_policy.add_rule(high_trust_rule)
        default_policy.add_rule(low_trust_rule)
        self.policies[default_policy.policy_id] = default_policy
    
    def _initialize_policy_templates(self):
        """Initialize policy templates for common scenarios"""
        
        self.policy_templates = {
            "high_security": {
                "name": "High Security Template",
                "description": "Template for high-security environments",
                "rules": [
                    {
                        "conditions": [{"field": "context_type", "operator": "in_list", "value": ["EMOTIONAL_STATE", "USER_PREFERENCE"]}],
                        "actions": {"require_encryption": "asymmetric", "audit_required": True}
                    }
                ]
            },
            "development": {
                "name": "Development Environment Template",
                "description": "Relaxed security for development environments",
                "rules": [
                    {
                        "conditions": [{"field": "environment", "operator": "equals", "value": "development"}],
                        "actions": {"require_encryption": "obfuscated"}
                    }
                ]
            },
            "production": {
                "name": "Production Environment Template",
                "description": "Strict security for production environments", 
                "rules": [
                    {
                        "conditions": [{"field": "environment", "operator": "equals", "value": "production"}],
                        "actions": {"require_encryption": "symmetric", "audit_required": True, "key_rotation": "daily"}
                    }
                ]
            }
        }
    
    def _create_hipaa_policy(self) -> SecurityPolicy:
        """Create HIPAA compliance policy"""
        
        hipaa_policy = SecurityPolicy(
            policy_id="hipaa_compliance",
            name="HIPAA Compliance Policy",
            description="Policy for HIPAA regulatory compliance",
            policy_type=PolicyType.COMPLIANCE,
            scope=PolicyScope.GLOBAL,
            metadata={"regulation": "HIPAA", "compliance_level": "strict"}
        )
        
        # PHI protection rule
        phi_rule = PolicyRule(
            rule_id="hipaa_phi_protection",
            name="PHI Protection",
            description="Protect Protected Health Information",
            conditions=[
                PolicyCondition("payload", "matches_regex", r".*(medical|health|patient|diagnosis).*")
            ],
            actions={
                PolicyAction.REQUIRE_ENCRYPTION: EncryptionLevel.ASYMMETRIC.value,
                PolicyAction.AUDIT_REQUIRED: True,
                PolicyAction.TIME_LIMIT: 3600  # 1 hour limit
            },
            priority=PolicyPriority.REGULATORY
        )
        
        hipaa_policy.add_rule(phi_rule)
        return hipaa_policy
    
    def _create_gdpr_policy(self) -> SecurityPolicy:
        """Create GDPR compliance policy"""
        
        gdpr_policy = SecurityPolicy(
            policy_id="gdpr_compliance",
            name="GDPR Compliance Policy", 
            description="Policy for GDPR regulatory compliance",
            policy_type=PolicyType.COMPLIANCE,
            scope=PolicyScope.GLOBAL,
            metadata={"regulation": "GDPR", "compliance_level": "strict"}
        )
        
        # Personal data protection rule
        personal_data_rule = PolicyRule(
            rule_id="gdpr_personal_data",
            name="Personal Data Protection",
            description="Protect personal data under GDPR",
            conditions=[
                PolicyCondition("payload", "matches_regex", r".*(email|phone|address|name|personal).*")
            ],
            actions={
                PolicyAction.REQUIRE_ENCRYPTION: EncryptionLevel.SYMMETRIC.value,
                PolicyAction.AUDIT_REQUIRED: True,
                PolicyAction.APPROVAL_REQUIRED: True
            },
            priority=PolicyPriority.REGULATORY
        )
        
        gdpr_policy.add_rule(personal_data_rule)
        return gdpr_policy
    
    def _create_sox_policy(self) -> SecurityPolicy:
        """Create SOX compliance policy"""
        
        sox_policy = SecurityPolicy(
            policy_id="sox_compliance",
            name="SOX Compliance Policy",
            description="Policy for Sarbanes-Oxley regulatory compliance",
            policy_type=PolicyType.COMPLIANCE,
            scope=PolicyScope.CONTEXT_TYPE,
            metadata={
                "regulation": "SOX", 
                "compliance_level": "strict",
                "context_types": ["TASK_KNOWLEDGE", "SYSTEM_STATE"]
            }
        )
        
        # Financial data protection rule
        financial_rule = PolicyRule(
            rule_id="sox_financial_data",
            name="Financial Data Protection",
            description="Protect financial and audit data under SOX",
            conditions=[
                PolicyCondition("payload", "matches_regex", r".*(financial|revenue|audit|earnings|accounting).*"),
                PolicyCondition("context_type", "in_list", ["TASK_KNOWLEDGE", "SYSTEM_STATE"])
            ],
            actions={
                PolicyAction.REQUIRE_ENCRYPTION: EncryptionLevel.MULTI_LAYER.value,
                PolicyAction.AUDIT_REQUIRED: True,
                PolicyAction.MULTI_FACTOR: True,
                PolicyAction.APPROVAL_REQUIRED: True
            },
            priority=PolicyPriority.REGULATORY
        )
        
        sox_policy.add_rule(financial_rule)
        return sox_policy


class PolicyConflictResolver:
    """
    Resolves conflicts between multiple policies when they provide contradictory guidance.
    """
    
    def __init__(self):
        self.resolution_strategies = {
            "most_restrictive": self._most_restrictive_strategy,
            "highest_priority": self._highest_priority_strategy,
            "context_aware": self._context_aware_strategy,
            "adaptive_learning": self._adaptive_learning_strategy
        }
        self.conflict_history: List[Dict[str, Any]] = []
    
    def resolve_conflict(self, 
                        conflicting_decisions: List[Dict[str, Any]], 
                        context: Dict[str, Any],
                        strategy: str = "context_aware") -> Dict[str, Any]:
        """
        Resolve conflict between multiple policy decisions
        """
        
        if not conflicting_decisions:
            return {}
        
        if len(conflicting_decisions) == 1:
            return conflicting_decisions[0]
        
        # Record conflict for learning
        conflict_record = {
            "conflicting_decisions": conflicting_decisions,
            "context": context,
            "strategy_used": strategy,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Apply resolution strategy
        if strategy in self.resolution_strategies:
            resolved_decision = self.resolution_strategies[strategy](
                conflicting_decisions, context
            )
        else:
            # Default to most restrictive
            resolved_decision = self._most_restrictive_strategy(
                conflicting_decisions, context
            )
        
        conflict_record["resolved_decision"] = resolved_decision
        self.conflict_history.append(conflict_record)
        
        # Keep only recent conflicts
        if len(self.conflict_history) > 200:
            self.conflict_history = self.conflict_history[-150:]
        
        return resolved_decision
    
    def _most_restrictive_strategy(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose the most restrictive security measures"""
        
        resolved = {}
        
        for decision in decisions:
            for key, value in decision.get("decisions", {}).items():
                if key not in resolved:
                    resolved[key] = value
                else:
                    # Apply most restrictive logic
                    if key == "require_encryption":
                        resolved[key] = self._higher_encryption_level(resolved[key], value)
                    elif key == "minimum_trust":
                        resolved[key] = max(resolved[key], value)
                    elif key == "time_limit":
                        resolved[key] = min(resolved[key], value)
                    elif isinstance(resolved[key], bool) and isinstance(value, bool):
                        resolved[key] = resolved[key] or value  # OR for boolean flags
        
        return {"decisions": resolved, "resolution_strategy": "most_restrictive"}
    
    def _highest_priority_strategy(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose decisions from highest priority policies"""
        
        # Sort decisions by policy priority (assuming it's included in the decision metadata)
        sorted_decisions = sorted(
            decisions, 
            key=lambda d: d.get("policy_priority", 0), 
            reverse=True
        )
        
        # Take decisions from highest priority policy
        if sorted_decisions:
            return {
                "decisions": sorted_decisions[0].get("decisions", {}),
                "resolution_strategy": "highest_priority",
                "winning_policy": sorted_decisions[0].get("policy_id")
            }
        
        return {"decisions": {}, "resolution_strategy": "highest_priority"}
    
    def _context_aware_strategy(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve based on context sensitivity and risk assessment"""
        
        # Assess context risk
        risk_score = self._assess_context_risk(context)
        
        if risk_score > 0.7:
            # High risk - use most restrictive
            return self._most_restrictive_strategy(decisions, context)
        elif risk_score < 0.3:
            # Low risk - use least restrictive
            return self._least_restrictive_strategy(decisions, context)
        else:
            # Medium risk - use highest priority
            return self._highest_priority_strategy(decisions, context)
    
    def _adaptive_learning_strategy(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve based on learned patterns from conflict history"""
        
        # Find similar past conflicts
        similar_conflicts = self._find_similar_conflicts(context)
        
        if similar_conflicts:
            # Use resolution that worked best in similar situations
            successful_resolutions = [c for c in similar_conflicts 
                                    if c.get("outcome_success", False)]
            
            if successful_resolutions:
                # Use the most common successful resolution pattern
                most_common_resolution = self._find_most_common_resolution(successful_resolutions)
                return {
                    "decisions": most_common_resolution,
                    "resolution_strategy": "adaptive_learning",
                    "based_on_similar_conflicts": len(successful_resolutions)
                }
        
        # Fallback to context-aware strategy
        return self._context_aware_strategy(decisions, context)
    
    def _least_restrictive_strategy(self, decisions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose the least restrictive security measures"""
        
        resolved = {}
        
        for decision in decisions:
            for key, value in decision.get("decisions", {}).items():
                if key not in resolved:
                    resolved[key] = value
                else:
                    # Apply least restrictive logic
                    if key == "require_encryption":
                        resolved[key] = self._lower_encryption_level(resolved[key], value)
                    elif key == "minimum_trust":
                        resolved[key] = min(resolved[key], value)
                    elif key == "time_limit":
                        resolved[key] = max(resolved[key], value)
                    elif isinstance(resolved[key], bool) and isinstance(value, bool):
                        resolved[key] = resolved[key] and value  # AND for boolean flags
        
        return {"decisions": resolved, "resolution_strategy": "least_restrictive"}
    
    def _assess_context_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk level of current context"""
        
        risk_score = 0.0
        
        # Risk factors
        sensitive_contexts = ["EMOTIONAL_STATE", "USER_PREFERENCE", "SYSTEM_STATE"]
        if context.get("context_type") in sensitive_contexts:
            risk_score += 0.3
        
        # Trust level risk (lower trust = higher risk)
        trust_level = context.get("trust_level", 0.5)
        risk_score += (1.0 - trust_level) * 0.4
        
        # Payload sensitivity
        payload_str = str(context.get("payload", "")).lower()
        sensitive_keywords = ["password", "secret", "private", "confidential", "personal", "medical", "financial"]
        keyword_matches = sum(1 for keyword in sensitive_keywords if keyword in payload_str)
        risk_score += min(0.3, keyword_matches * 0.1)
        
        return min(1.0, risk_score)
    
    def _find_similar_conflicts(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find conflicts similar to current context"""
        
        similar = []
        
        for conflict in self.conflict_history:
            similarity_score = self._calculate_context_similarity(
                context, conflict["context"]
            )
            
            if similarity_score > 0.7:  # 70% similarity threshold
                similar.append(conflict)
        
        return similar
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        
        similarity_factors = []
        
        # Context type similarity
        if context1.get("context_type") == context2.get("context_type"):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Trust level similarity
        trust1 = context1.get("trust_level", 0.5)
        trust2 = context2.get("trust_level", 0.5)
        trust_similarity = 1.0 - abs(trust1 - trust2)
        similarity_factors.append(trust_similarity)
        
        # Agent similarity
        if context1.get("from_agent") == context2.get("from_agent"):
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        return sum(similarity_factors) / len(similarity_factors)
    
    def _find_most_common_resolution(self, resolutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most commonly used resolution pattern"""
        
        resolution_counts = {}
        
        for resolution in resolutions:
            decisions = resolution.get("resolved_decision", {}).get("decisions", {})
            decisions_key = json.dumps(decisions, sort_keys=True)
            
            if decisions_key not in resolution_counts:
                resolution_counts[decisions_key] = {"count": 0, "decisions": decisions}
            resolution_counts[decisions_key]["count"] += 1
        
        if resolution_counts:
            most_common = max(resolution_counts.values(), key=lambda x: x["count"])
            return most_common["decisions"]
        
        return {}
    
    def _higher_encryption_level(self, level1: str, level2: str) -> str:
        """Return the higher of two encryption levels"""
        
        try:
            enc_level1 = EncryptionLevel(level1)
            enc_level2 = EncryptionLevel(level2)
            levels = list(EncryptionLevel)
            return levels[max(levels.index(enc_level1), levels.index(enc_level2))].value
        except:
            return level1  # Fallback
    
    def _lower_encryption_level(self, level1: str, level2: str) -> str:
        """Return the lower of two encryption levels"""
        
        try:
            enc_level1 = EncryptionLevel(level1)
            enc_level2 = EncryptionLevel(level2)
            levels = list(EncryptionLevel)
            return levels[min(levels.index(enc_level1), levels.index(enc_level2))].value
        except:
            return level1  # Fallback


# Example usage and testing
if __name__ == "__main__":
    # Create test agent
    agent = AgentIdentity(
        agent_id="policy-test-agent",
        agent_type="security_agent",
        version="1.0.0",
        capabilities=["policy_management", "security_analysis"]
    )
    
    # Create policy engine
    policy_engine = PolicyEngine(agent)
    
    # Test policy evaluation
    context = {
        "context_type": "EMOTIONAL_STATE",
        "trust_level": 0.6,
        "from_agent": "agent-001",
        "to_agent": "agent-002",
        "payload": {"mood": "stressed", "personal_info": "yes"},
        "environment": "production"
    }
    
    print("Testing policy evaluation...")
    result = policy_engine.evaluate_context(context)
    print("Policy decisions:", json.dumps(result["decisions"], indent=2))
    print("Policies evaluated:", result["evaluation_metadata"]["policies_evaluated"])
    
    # Test adaptive policy creation
    pattern_data = {
        "pattern_id": "agent_001_emotional_context",
        "context_type": "EMOTIONAL_STATE",
        "agent_pair": "agent-001_agent-002",
        "success_rate": 0.85,
        "optimal_encryption": "symmetric",
        "minimum_trust": 0.7
    }
    
    adaptive_policy = policy_engine.create_adaptive_policy(pattern_data)
    if adaptive_policy:
        print(f"Created adaptive policy: {adaptive_policy.name}")
    
    # Test compliance policies
    compliance_policies = policy_engine.get_compliance_policies(["HIPAA", "GDPR"])
    print(f"Created {len(compliance_policies)} compliance policies")
    
    # Test policy conflict resolution
    conflict_resolver = PolicyConflictResolver()
    
    conflicting_decisions = [
        {
            "decisions": {"require_encryption": "symmetric", "minimum_trust": 0.6},
            "policy_id": "policy_1",
            "policy_priority": 5
        },
        {
            "decisions": {"require_encryption": "asymmetric", "minimum_trust": 0.8},
            "policy_id": "policy_2", 
            "policy_priority": 8
        }
    ]
    
    resolved = conflict_resolver.resolve_conflict(
        conflicting_decisions, context, "context_aware"
    )
    print("Conflict resolution:", json.dumps(resolved["decisions"], indent=2))
    
    # Test policy analysis
    analysis = policy_engine.analyze_policy_patterns()
    print("Policy analysis:", json.dumps(analysis, indent=2, default=str))
    
    # Export policy configuration
    config = policy_engine.export_policy_configuration()
    print(f"Exported configuration with {len(config['policies'])} policies")
    
    print("Policy engine testing completed successfully!")
