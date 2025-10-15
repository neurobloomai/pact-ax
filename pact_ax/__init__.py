"""
PACT-AX: Agent Collaboration Layer

Provides primitives for multi-agent coordination:
- Context sharing
- State transfer  
- Policy alignment
- Trust scoring
- Story keeping (conscious continuity)
"""

__version__ = "0.1.0"

# Story Keeper - always available
from .primitives.story_keeper import StoryKeeper, StoryArc

# Core primitives - import only if they exist
try:
    from .core.base_primitive import PACTPrimitive, PACTConfig
    from .primitives.context_share import ContextShareManager
    from .primitives.state_transfer import StateTransferManager
    from .primitives.policy_align import PolicyAlignmentManager
    
    __all__ = [
        "__version__",
        "StoryKeeper",
        "StoryArc",
        "PACTPrimitive",
        "PACTConfig",
        "ContextShareManager",
        "StateTransferManager",
        "PolicyAlignmentManager",
    ]
except ImportError:
    # If other primitives not ready yet, just export Story Keeper
    __all__ = [
        "__version__",
        "StoryKeeper",
        "StoryArc",
    ]
