"""
PACT-AX: Protocol for AI Coordination and Trust - Agent eXtension

Session Integrity Layer for MCP (Model Context Protocol)

Unlike policy-based security that evaluates individual actions,
PACT-AX maintains relational context across sessions and detects
behavioral drift that policy engines miss.
"""

from .story_keeper import StoryKeeper, SessionStory, RelationalEvent, TrustTrajectory
from .proxy import PACTAXProxy, ProxyConfig

__version__ = "0.1.0"
__all__ = [
    "StoryKeeper",
    "SessionStory", 
    "RelationalEvent",
    "TrustTrajectory",
    "PACTAXProxy",
    "ProxyConfig"
]
