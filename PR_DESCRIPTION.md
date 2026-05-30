# feat: StateTransferManager — core wealth transfer protocol

## What was missing

`pact_ax/state/` had only `epistemic_transfer.py` (knowledge + confidence
fidelity, isolated).  `pact_ax/primitives/state_transfer.py` had narrative
scaffolding and a `restore_checkpoint()` placeholder.

Nothing tied them together into an actual, validated, lifecycle-tracked handoff
protocol.

## What this adds

### `pact_ax/state/state_transfer_manager.py` — the orchestrator

A production-ready `StateTransferManager` that combines:

| Layer | Source |
|-------|--------|
| Epistemic fidelity (knowledge + uncertainty) | `EpistemicStateTransfer` |
| Narrative continuity (story arc + context) | `StoryKeeper` |
| Trust scoring | heuristic baseline, pluggable |
| Lifecycle tracking | `TransferStatus` FSM |
| Validation | `ValidationResult` with structured issues |

**Handoff lifecycle:**
```
prepare() → PREPARING
send()    → IN_FLIGHT
receive() → RECEIVED → INTEGRATED
                    ↘ rollback() → ROLLED_BACK
(any step can → FAILED)
```

**Key types:**

| Class | Purpose |
|-------|---------|
| `StateTransferManager` | Main orchestrator per agent |
| `HandoffPacket` | Self-contained transfer unit (serialisable to/from dict) |
| `HandoffReason` | CONTINUATION / ESCALATION / PAUSE / COMPLETION / LOAD_BALANCE |
| `TransferStatus` | Full lifecycle enum |
| `IntegrationResult` | Rich receive() outcome including warnings |
| `ValidationResult` | Structured validation with issue list |

**Notable capabilities:**
- `packet.to_dict()` / `HandoffPacket.from_dict()` — full serialisation for cross-process / network delivery
- `POST /registry/reload` equivalent: `manager.rollback(packet_id)` for safe undo
- `checkpoint(label, state)` + `restore(ckpt_id)` — now fully implemented (was placeholder)
- `manager.summary()` — observability snapshot
- `clear_completed()` — TTL-based cleanup (configurable, default 120 min)
- Lazy import of optionals — works even if `EpistemicStateTransfer` or `StoryKeeper` aren't available

### `pact_ax/state/__init__.py` — clean public API

```python
from pact_ax.state import (
    StateTransferManager,
    HandoffPacket,
    HandoffReason,
    TransferStatus,
    IntegrationResult,
    ValidationResult,
    EpistemicStateTransfer,
)
```

### `tests/test_state_transfer_manager.py` — 35 tests

Coverage across: prepare/send, validate, receive, rollback, checkpoints,
observability, StoryKeeper integration (via mock), and all `HandoffReason`
variants via parametrize.

## Usage

```python
from pact_ax.state import StateTransferManager, HandoffReason

# ── Sender side ──────────────────────────────────────────────────────────────
sender = StateTransferManager(agent_id="agent-A")

packet_id = sender.prepare(
    to_agent_id       = "agent-B",
    state_data        = {"task": "analyse Q3 report", "progress": 0.6},
    reason            = HandoffReason.CONTINUATION,
    epistemic_states  = [my_epistemic_state],   # optional
    context           = {"priority": "high"},
)
packet = sender.send(packet_id)

# Deliver packet (serialize for transport if needed)
payload = packet.to_dict()

# ── Receiver side ─────────────────────────────────────────────────────────────
receiver = StateTransferManager(agent_id="agent-B")
result   = receiver.receive(HandoffPacket.from_dict(payload))

if result.success:
    state   = result.integrated_state["state_data"]
    context = result.integrated_state["handoff_provenance"]
else:
    print("Handoff failed:", result.error)

# ── Checkpoint & restore ──────────────────────────────────────────────────────
ckpt_id = sender.checkpoint("before-risky-op", current_state)
# ... do risky work ...
if something_went_wrong:
    snap = sender.restore(ckpt_id)
```

## Files changed

```
pact_ax/state/
├── __init__.py                  ← updated (was stub)
├── epistemic_transfer.py        ← unchanged
└── state_transfer_manager.py   ← NEW

tests/
└── test_state_transfer_manager.py   ← NEW (35 tests)
```
