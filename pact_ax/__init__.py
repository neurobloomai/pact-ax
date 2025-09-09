"""
PACT-AX: Agent Collaboration Layer

Provides primitives for multi-agent coordination:
- Context sharing
- State transfer
- Policy alignment
- Trust scoring
"""

from .version import __version__
from .core.base_primitive import PACTPrimitive, PACTConfig
from .primitives.context_share import ContextShareManager
from .primitives.state_transfer import StateTransferManager
from .primitives.policy_align import PolicyAlignmentManager

__all__ = [
    "__version__",
    "PACTPrimitive",
    "PACTConfig", 
    "ContextShareManager",
    "StateTransferManager",
    "PolicyAlignmentManager",
]
