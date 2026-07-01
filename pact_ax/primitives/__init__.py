"""
PACT-AX Primitives

Core collaboration primitives for multi-agent systems.
"""

from .story_keeper import StoryKeeper, StoryArc
from pact_ax.state import StateTransferManager, HandoffReason, HandoffPacket, TransferStatus
from .context_share import ContextShareManager, ContextType, TrustLevel, Priority
from .trust_score import TrustManager
from .trust_chain import TrustChainManager, TrustChain, ChainHop, ChainState, ChainScore, ChainVerification
from .trust_alignment import TrustAlignmentCheck, TrustDimension, DimensionScore, AlignmentResult, GateMode
from .trust_context import TrustContext, TrustScope, TrustOperatingMode, DimensionBreak, Action, ActionLevel
from .capability_registry import CapabilityRegistry, Capability
from .agent_router import AgentRouter, RouteDecision, RouteCandidate
from .dead_letter_queue import DeadLetterQueue, DLQEntry, DLQStatus
from .episodic_memory import EpisodicMemory, Episode

__all__ = [
    # Narrative continuity
    "StoryKeeper",
    "StoryArc",
    # Agent handoff (canonical: pact_ax.state)
    "StateTransferManager",
    "HandoffReason",
    "HandoffPacket",
    "TransferStatus",
    # Context sharing
    "ContextShareManager",
    "ContextType",
    "TrustLevel",
    "Priority",
    # Trust scoring
    "TrustManager",
    # Trust alignment (n/n gate)
    "TrustAlignmentCheck",
    "TrustDimension",
    "DimensionScore",
    "AlignmentResult",
    "GateMode",
    # Trust context (scoped propagatable contract)
    "TrustContext",
    "TrustScope",
    "TrustOperatingMode",
    "DimensionBreak",
    "Action",
    "ActionLevel",
    # Capability discovery
    "CapabilityRegistry",
    "Capability",
    # Trust-weighted routing
    "AgentRouter",
    "RouteDecision",
    "RouteCandidate",
    # Dead Letter Queue
    "DeadLetterQueue",
    "DLQEntry",
    "DLQStatus",
    # Episodic memory
    "EpisodicMemory",
    "Episode",
]
