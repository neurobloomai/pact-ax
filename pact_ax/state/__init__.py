"""
pact_ax.state
─────────────
State management and transfer primitives for PACT-AX agents.

Public API
──────────
    from pact_ax.state import (
        StateTransferManager,
        HandoffPacket,
        IntegrationResult,
        ValidationResult,
        HandoffReason,
        TransferStatus,
        EpistemicStateTransfer,   # knowledge + confidence transfer
    )
"""

from pact_ax.state.state_transfer_manager import (
    HandoffPacket,
    HandoffReason,
    IntegrationResult,
    StateTransferManager,
    TransferStatus,
    ValidationResult,
)
from pact_ax.state.epistemic_transfer import EpistemicStateTransfer

__all__ = [
    "StateTransferManager",
    "HandoffPacket",
    "HandoffReason",
    "IntegrationResult",
    "ValidationResult",
    "TransferStatus",
    "EpistemicStateTransfer",
]
