"""
pact_ax.coordination
─────────────────────
Complete humility-aware coordination system for PACT-AX agents.

Public API
──────────
    # Original four modules (unchanged)
    from pact_ax.coordination import (
        HumilityAwareCoordinator, DelegationChain, Query, Agent,
        PolicyAlignmentManager, PolicyDecision, PolicyConstraint,
        PolicyConflictResolution, PolicyLearning,
        GossipClarityProtocol, GossipMessage, ClarityAmplification,
        TrustNetwork, TrustScore, TrustInteractionOutcome,
        ReputationSystem, TrustDimension,
    )

    # New: consensus
    from pact_ax.coordination import (
        ConsensusProtocol, ConsensusStrategy, ConsensusOutcome,
        ConsensusResult, Vote,
    )

    # New: coordination bus
    from pact_ax.coordination import (
        CoordinationBus, CoordinationEvent, EventType, AgentSession,
    )
"""

# ── Original exports (unchanged) ─────────────────────────────────────────────

from pact_ax.coordination.humility_aware import (
    HumilityAwareCoordinator,
    DelegationChain,
    Query,
    Agent,
)

from pact_ax.coordination.policy_alignment import (
    PolicyAlignmentManager,
    PolicyDecision,
    PolicyConstraint,
    PolicyConflictResolution,
    PolicyLearning,
)

from pact_ax.coordination.gossip_clarity import (
    GossipClarityProtocol,
    GossipMessage,
    ClarityAmplification,
)

from pact_ax.coordination.trust_primitives import (
    TrustNetwork,
    TrustScore,
    TrustInteractionOutcome,
    ReputationSystem,
    TrustDimension,
)

# ── New: consensus ────────────────────────────────────────────────────────────

from pact_ax.coordination.consensus import (
    ConsensusProtocol,
    ConsensusStrategy,
    ConsensusOutcome,
    ConsensusResult,
    Vote,
)

# ── New: coordination bus ─────────────────────────────────────────────────────

from pact_ax.coordination.coordination_bus import (
    CoordinationBus,
    CoordinationEvent,
    EventType,
    AgentSession,
)

__all__ = [
    # humility_aware
    "HumilityAwareCoordinator", "DelegationChain", "Query", "Agent",
    # policy_alignment
    "PolicyAlignmentManager", "PolicyDecision", "PolicyConstraint",
    "PolicyConflictResolution", "PolicyLearning",
    # gossip_clarity
    "GossipClarityProtocol", "GossipMessage", "ClarityAmplification",
    # trust_primitives
    "TrustNetwork", "TrustScore", "TrustInteractionOutcome",
    "ReputationSystem", "TrustDimension",
    # consensus  (NEW)
    "ConsensusProtocol", "ConsensusStrategy", "ConsensusOutcome",
    "ConsensusResult", "Vote",
    # coordination_bus  (NEW)
    "CoordinationBus", "CoordinationEvent", "EventType", "AgentSession",
]
