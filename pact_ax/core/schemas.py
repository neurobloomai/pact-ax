"""
pact_ax/core/schemas.py
────────────────────────
Pydantic v2 validation models for the public PACT-AX wire format.

These models validate data at system boundaries (API requests/responses,
serialisation round-trips). Internal code uses the dataclass-based schemas
in pact_ax/primitives/context_share/schemas.py; these models handle the
HTTP / JSON surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations (string literals for JSON compatibility) ─────────────────────

CONTEXT_TYPES = {
    "task_knowledge", "emotional_state", "handoff_request",
    "trust_signal", "capability_signal",
}
PRIORITIES    = {"low", "normal", "high", "critical"}
TRUST_LEVELS  = {"emerging", "building", "strong", "deep"}
OUTCOMES      = {"positive", "negative", "neutral", "partial"}
HANDOFF_REASONS = {
    "continuation", "escalation", "pause", "completion", "load_balance",
}
CONFIDENCE_LEVELS = {"CERTAIN", "CONFIDENT", "MODERATE", "LOW", "UNKNOWN"}


# ── Shared validators ─────────────────────────────────────────────────────────

def _validate_enum(value: str, allowed: set, field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)!r}, got {value!r}")
    return value


# ── Agent identity ─────────────────────────────────────────────────────────────

class AgentIdentitySchema(BaseModel):
    agent_id: str = Field(..., min_length=1, description="Unique agent identifier")
    agent_type: str = Field(default="generic")
    version: str = Field(default="1.0.0")
    capabilities: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)


# ── Context packet ─────────────────────────────────────────────────────────────

class ContextPacketSchema(BaseModel):
    """Validated wire representation of a context packet."""

    from_agent: AgentIdentitySchema
    to_agent: str = Field(..., min_length=1)
    context_type: str
    payload: Dict[str, Any]
    priority: str = "normal"
    trust_required: str = "emerging"
    expires_at: Optional[datetime] = None

    @field_validator("context_type")
    @classmethod
    def validate_context_type(cls, v: str) -> str:
        return _validate_enum(v, CONTEXT_TYPES, "context_type")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        return _validate_enum(v, PRIORITIES, "priority")

    @field_validator("trust_required")
    @classmethod
    def validate_trust_required(cls, v: str) -> str:
        return _validate_enum(v, TRUST_LEVELS, "trust_required")

    @field_validator("payload")
    @classmethod
    def payload_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("payload must not be empty")
        return v


# ── Handoff packet ─────────────────────────────────────────────────────────────

class HandoffPacketSchema(BaseModel):
    """Validated wire representation of a state-transfer handoff packet."""

    packet_id: str = Field(..., min_length=1)
    from_agent_id: str = Field(..., min_length=1)
    to_agent_id: str = Field(..., min_length=1)
    reason: str
    state_data: Dict[str, Any]
    epistemic_payload: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: Dict[str, Any] = Field(default_factory=dict)
    trust_score: float = Field(..., ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    status: str = "in_flight"

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return _validate_enum(v, HANDOFF_REASONS, "reason")

    @field_validator("state_data")
    @classmethod
    def state_data_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("state_data must not be empty")
        return v

    @model_validator(mode="after")
    def sender_receiver_differ(self) -> "HandoffPacketSchema":
        if self.from_agent_id == self.to_agent_id:
            raise ValueError("from_agent_id and to_agent_id must be different")
        return self


# ── Policy decision ────────────────────────────────────────────────────────────

class PolicyDecisionSchema(BaseModel):
    """Validated wire representation of a policy decision."""

    decision: str = Field(..., min_length=1)
    confidence: str
    reasoning: str = Field(default="")
    agent_id: str = Field(..., min_length=1)
    domain: str = Field(default="general")
    alternatives_considered: List[str] = Field(default_factory=list)
    uncertainty_factors: List[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        return _validate_enum(v, CONFIDENCE_LEVELS, "confidence")


# ── Trust assessment response ──────────────────────────────────────────────────

class TrustAssessmentSchema(BaseModel):
    """Validated response from assess_trust()."""

    agent_id: str
    context_type: str
    base_trust: float = Field(..., ge=0.0, le=1.0)
    situation_adjustment: float
    final_trust: float = Field(..., ge=0.0, le=1.0)
    recommendation: str

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: str) -> str:
        allowed = {"share", "caution"}
        if v not in allowed:
            raise ValueError(f"recommendation must be one of {sorted(allowed)!r}")
        return v


# ── Capability limit response ──────────────────────────────────────────────────

class CapabilityLimitSchema(BaseModel):
    """Validated response from sense_capability_limit()."""

    task: str
    current_confidence: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    approaching_limit: bool
    limit_proximity: float = Field(..., ge=0.0, le=1.0)
    recommendation: str

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: str) -> str:
        allowed = {"continue", "monitor", "prepare_handoff", "immediate_handoff"}
        if v not in allowed:
            raise ValueError(f"recommendation must be one of {sorted(allowed)!r}")
        return v
