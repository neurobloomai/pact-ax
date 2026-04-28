"""
PACT-AX: Agent Collaboration Layer
ContextShareManager - First Primitive

Enables organic context sharing with trust awareness and capability sensing.
Built on principles of dynamic evolution and natural harmony.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import time

from .schemas import (
    ContextType,
    TrustLevel,
    Priority,
    CollaborationOutcome,
    AgentIdentity,
    ContextMetadata,
    ContextPacket,
    AgentTrustProfile,
    TrustEvolution,
    CapabilitySensor,
    validate_context_packet,
)


class ContextShareManager:
    """
    Core primitive for agent collaboration in PACT-AX.
    Enables organic context sharing with evolving trust and capability awareness.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str = "generic",
        version: str = "1.0.0",
        capabilities: Optional[List[str]] = None,
        specializations: Optional[List[str]] = None,
    ):
        self.agent_id = agent_id
        self.identity = AgentIdentity(
            agent_id=agent_id,
            agent_type=agent_type,
            version=version,
            capabilities=capabilities or [],
            specializations=specializations or [],
        )
        self.trust_profiles: Dict[str, AgentTrustProfile] = {}
        self.context_buffer: List[ContextPacket] = []
        self.capability_sensors: Dict[str, CapabilitySensor] = {}
        self.collaboration_patterns: Dict[str, Any] = {}

    def create_context_packet(
        self,
        target_agent: str,
        context_type: ContextType,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        ttl_seconds: Optional[int] = None,
    ) -> ContextPacket:
        """Create a context packet with trust and capability awareness."""
        from datetime import datetime, timezone, timedelta

        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        trust_required = self._assess_required_trust(context_type, payload)

        packet = ContextPacket(
            from_agent=self.identity,
            to_agent=target_agent,
            context_type=context_type,
            payload=payload,
            metadata=ContextMetadata(),
            trust_required=trust_required,
            priority=priority,
            expires_at=expires_at,
        )

        return packet

    def assess_trust(
        self,
        target_agent: str,
        context_type: ContextType,
        current_situation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess trust level for sharing specific context with a target agent."""
        profile = self._get_or_create_profile(target_agent)
        base_trust = profile.get_trust_for_context(context_type)

        situation_adjustment = self._calculate_situation_adjustment(
            current_situation,
            [
                ix
                for evo in profile.trust_evolution.values()
                for ix in evo.interactions
            ],
        )

        adjusted_trust = max(0.0, min(1.0, base_trust + situation_adjustment))

        return {
            "agent_id": target_agent,
            "context_type": context_type.value,
            "base_trust": base_trust,
            "situation_adjustment": situation_adjustment,
            "final_trust": adjusted_trust,
            "recommendation": "share" if adjusted_trust > 0.6 else "caution",
        }

    def sense_capability_limit(
        self,
        current_task: str,
        confidence_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Sense approaching capability limits for proactive handoff."""
        sensor = self.capability_sensors.get(current_task)
        current_confidence = sensor.current_confidence if sensor else 1.0

        approaching_limit = current_confidence < confidence_threshold
        limit_proximity = (
            max(0.0, 1.0 - (current_confidence / confidence_threshold))
            if confidence_threshold > 0.0 else 0.0
        )

        return {
            "task": current_task,
            "current_confidence": current_confidence,
            "threshold": confidence_threshold,
            "approaching_limit": approaching_limit,
            "limit_proximity": limit_proximity,
            "recommendation": self._get_capability_recommendation(
                current_confidence, confidence_threshold
            ),
        }

    def prepare_handoff(
        self,
        target_agent: str,
        current_task: str,
        preserve_emotional_context: bool = True,
        transfer_ownership: bool = True,
    ) -> ContextPacket:
        """Prepare smooth handoff before hitting capability limits."""
        handoff_payload: Dict[str, Any] = {
            "current_task": current_task,
            "handoff_reason": "proactive_capability_limit",
            "task_progress": self._gather_task_progress(),
            "transfer_ownership": transfer_ownership,
            "recommendations": self._generate_handoff_recommendations(current_task),
        }

        if preserve_emotional_context:
            handoff_payload["emotional_context"] = self._gather_emotional_context()

        return self.create_context_packet(
            target_agent=target_agent,
            context_type=ContextType.HANDOFF_REQUEST,
            payload=handoff_payload,
            priority=Priority.HIGH,
        )

    def update_capability_confidence(self, task: str, confidence: float) -> None:
        """Update confidence level for a specific task."""
        if task not in self.capability_sensors:
            self.capability_sensors[task] = CapabilitySensor(task_type=task)
        self.capability_sensors[task].update_confidence(confidence, {})

    def record_collaboration_outcome(
        self,
        target_agent: str,
        context_type: ContextType,
        outcome: str,
        impact: float = 1.0,
    ) -> None:
        """Record collaboration outcome to evolve trust and patterns."""
        try:
            outcome_enum = CollaborationOutcome(outcome)
        except ValueError:
            outcome_enum = CollaborationOutcome.NEUTRAL

        profile = self._get_or_create_profile(target_agent)
        profile.update_trust(
            context_type=context_type,
            outcome=outcome_enum,
            impact=impact,
            context={"recorded_at": time.time()},
        )

        pattern_key = f"{target_agent}_{context_type.value}"
        if pattern_key not in self.collaboration_patterns:
            self.collaboration_patterns[pattern_key] = {
                "successes": 0,
                "failures": 0,
            }
        pattern = self.collaboration_patterns[pattern_key]
        if outcome_enum == CollaborationOutcome.POSITIVE:
            pattern["successes"] += 1
        elif outcome_enum == CollaborationOutcome.NEGATIVE:
            pattern["failures"] += 1

    def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights about collaboration patterns and trust evolution."""
        insights: Dict[str, Any] = {
            "trust_summary": {},
            "capability_status": {
                task: sensor.current_confidence
                for task, sensor in self.capability_sensors.items()
            },
            "collaboration_patterns": self.collaboration_patterns,
        }

        for agent_id, profile in self.trust_profiles.items():
            trust_values = [
                evo.current_level for evo in profile.trust_evolution.values()
            ]
            insights["trust_summary"][agent_id] = {
                "overall_trust": profile.overall_trust,
                "average_trust": sum(trust_values) / len(trust_values) if trust_values else 0.5,
                "interaction_count": sum(
                    len(evo.interactions) for evo in profile.trust_evolution.values()
                ),
                "last_interaction": (
                    profile.last_interaction.isoformat()
                    if profile.last_interaction
                    else None
                ),
            }

        return insights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_profile(self, target_agent: str) -> AgentTrustProfile:
        if target_agent not in self.trust_profiles:
            self.trust_profiles[target_agent] = AgentTrustProfile(agent_id=target_agent)
        return self.trust_profiles[target_agent]

    def _assess_required_trust(
        self, context_type: ContextType, payload: Dict[str, Any]
    ) -> TrustLevel:
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
        history: List[Dict],
    ) -> float:
        adjustment = 0.0

        if situation.get("stakes") == "high":
            adjustment -= 0.1
        elif situation.get("stakes") == "low":
            adjustment += 0.05

        now = time.time()

        def _to_unix(h: Dict) -> float:
            val = h.get("timestamp")
            if val is None:
                return now - 7200
            if isinstance(val, (int, float)):
                return float(val)
            try:
                from datetime import timezone as _tz
                dt = datetime.fromisoformat(str(val))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return dt.timestamp()
            except (ValueError, TypeError):
                return now - 7200

        recent = [h for h in history if now - _to_unix(h) < 3600]
        if recent:
            positive = sum(1 for h in recent if h.get("outcome") == CollaborationOutcome.POSITIVE.value)
            positive_ratio = positive / len(recent)
            adjustment += (positive_ratio - 0.5) * 0.2

        return adjustment

    def _get_capability_recommendation(self, confidence: float, threshold: float) -> str:
        if confidence > threshold * 1.2:
            return "continue"
        elif confidence > threshold:
            return "monitor"
        elif confidence > threshold * 0.8:
            return "prepare_handoff"
        else:
            return "immediate_handoff"

    def _gather_task_progress(self) -> Dict[str, Any]:
        if not self.capability_sensors:
            return {
                "completion_percentage": 1.0,
                "current_step": "idle",
                "next_steps": [],
                "blockers": [],
            }

        confidences = {t: s.current_confidence for t, s in self.capability_sensors.items()}
        avg_confidence = sum(confidences.values()) / len(confidences)

        # Task closest to its warning threshold is the active step
        def _proximity(sensor: "CapabilitySensor") -> float:
            return sensor.threshold_warning - sensor.current_confidence

        active_sensor = max(self.capability_sensors.values(), key=_proximity)
        current_step = active_sensor.task_type

        next_steps = [
            t for t, s in self.capability_sensors.items()
            if s.current_confidence >= s.threshold_warning and t != current_step
        ]

        blockers = [
            t for t, s in self.capability_sensors.items()
            if s.current_confidence < s.threshold_critical
        ]

        return {
            "completion_percentage": round(avg_confidence, 3),
            "current_step": current_step,
            "next_steps": next_steps,
            "blockers": blockers,
        }

    def _gather_emotional_context(self) -> Dict[str, Any]:
        total_successes = sum(
            p.get("successes", 0) for p in self.collaboration_patterns.values()
        )
        total_failures = sum(
            p.get("failures", 0) for p in self.collaboration_patterns.values()
        )
        total = total_successes + total_failures

        if total == 0:
            sentiment = "neutral"
        elif total_successes / total >= 0.7:
            sentiment = "positive"
        elif total_failures / total >= 0.5:
            sentiment = "frustrated"
        else:
            sentiment = "neutral"

        avg_trust = (
            sum(p.overall_trust for p in self.trust_profiles.values()) / len(self.trust_profiles)
            if self.trust_profiles else 0.5
        )
        interaction_tone = "collaborative" if avg_trust >= 0.7 else "cautious" if avg_trust >= 0.4 else "guarded"

        stress_indicators = [
            t for t, s in self.capability_sensors.items()
            if s.current_confidence < s.threshold_warning
        ]

        preferred_context_types: Dict[str, int] = {}
        for key, pattern in self.collaboration_patterns.items():
            if pattern.get("successes", 0) > pattern.get("failures", 0):
                context_type = key.split("_", 1)[-1] if "_" in key else key
                preferred_context_types[context_type] = preferred_context_types.get(context_type, 0) + 1

        return {
            "user_sentiment": sentiment,
            "interaction_tone": interaction_tone,
            "stress_indicators": stress_indicators,
            "preferences": preferred_context_types,
        }

    def _generate_handoff_recommendations(self, task: str) -> List[str]:
        recommendations: List[str] = []

        sensor = self.capability_sensors.get(task)
        if sensor:
            if sensor.current_confidence < sensor.threshold_critical:
                recommendations.append(f"Urgent: confidence on '{task}' is critically low ({sensor.current_confidence:.2f})")
            elif sensor.current_confidence < sensor.threshold_warning:
                recommendations.append(f"Monitor '{task}' closely — confidence approaching threshold ({sensor.current_confidence:.2f})")
            if sensor.degradation_rate > 0.05:
                recommendations.append(f"Capability degrading at rate {sensor.degradation_rate:.2f}; receiving agent should ramp up quickly")

        for key, pattern in self.collaboration_patterns.items():
            if task in key:
                successes = pattern.get("successes", 0)
                failures = pattern.get("failures", 0)
                if successes + failures > 0:
                    ratio = successes / (successes + failures)
                    context_type = key.split("_", 1)[-1] if "_" in key else key
                    if ratio >= 0.7:
                        recommendations.append(f"Prior '{context_type}' collaborations succeeded {ratio:.0%} — continue this pattern")
                    elif ratio < 0.4:
                        recommendations.append(f"Prior '{context_type}' collaborations struggled ({ratio:.0%}) — adjust approach")

        if not recommendations:
            recommendations.append(f"No prior history for '{task}' — proceed with standard capability thresholds")

        return recommendations
