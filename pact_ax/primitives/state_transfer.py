"""
pact_ax.primitives.state_transfer
───────────────────────────────────
Re-exports the canonical StateTransferManager from pact_ax.state.

The full lifecycle API (prepare / send / receive / rollback / checkpoint)
and the convenience wrappers (prepare_handoff / receive_handoff /
create_checkpoint) all live on the same class.

Canonical import::

    from pact_ax.state import StateTransferManager

This module exists for backward compatibility with code that imported
from pact_ax.primitives.state_transfer.
"""

from pact_ax.state import StateTransferManager

__all__ = ["StateTransferManager"]
