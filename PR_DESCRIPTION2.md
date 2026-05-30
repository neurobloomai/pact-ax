# feat: consensus protocol + coordination bus (closes coordination gaps)

## Context

The coordination layer had 4 solid, isolated modules.  Two structural gaps
remained:

1. `PolicyConflictResolution.CONSENSUS_REQUIRED` was an enum value with no
   backing implementation — nothing could actually run consensus.
2. The 4 modules (trust, humility, gossip, policy) had no way to communicate
   with each other.  Wiring them together required bespoke glue code in every
   calling application.

This PR fills both gaps.

---

## New file: `pact_ax/coordination/consensus.py`

### `ConsensusProtocol`

Implements the concrete mechanism behind `CONSENSUS_REQUIRED`.

**Four strategies:**

| Strategy | Description |
|----------|-------------|
| `WEIGHTED_VOTE` | confidence × trust-score weighted majority (default) |
| `QUORUM` | simple vote count majority with configurable quorum fraction |
| `UNANIMOUS` | all agents must agree — for safety-critical decisions |
| `CONFIDENCE_THRESHOLD` | winner must clear a minimum average confidence bar |

**Outcome states:** `ACCEPTED` · `DEADLOCK` · `ESCALATE_TO_HUMAN` · `INSUFFICIENT_VOTES`

**Key types:**

| Class | Purpose |
|-------|---------|
| `Vote` | Agent's position (decision + confidence + trust-weighted) |
| `ConsensusResult` | Full outcome — breakdown, dissent map, confidence score |
| `ConsensusProtocol` | Runs rounds; tracks history and acceptance/escalation rates |

**Usage:**
```python
from pact_ax.coordination.consensus import ConsensusProtocol, ConsensusStrategy, Vote

protocol = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)
result = protocol.run(
    votes=[
        Vote("agent-A", "deploy-v2", confidence=0.85, reasoning="metrics stable"),
        Vote("agent-B", "deploy-v2", confidence=0.72),
        Vote("agent-C", "hold",      confidence=0.65),
    ],
    trust_scores={"agent-A": 0.9, "agent-B": 0.8, "agent-C": 0.75},
)
print(result.winning_decision)   # "deploy-v2"
print(result.confidence_score)   # 0.71
```

---

## New file: `pact_ax/coordination/coordination_bus.py`

### `CoordinationBus` + `AgentSession`

Event-driven backbone that connects all coordination modules.

**`CoordinationBus`** — typed pub/sub:
- Subscribe by `EventType` or wildcard (all events)
- `publish()` / `emit()` — synchronous delivery
- `recent_events()` — filterable event log
- `replay(from_event_id, handler)` — catch up from a known point
- Configurable `error_policy` — `"log"` (default) or `"raise"`

**20 `EventType` values** spanning the full coordination lifecycle:
```
trust.updated          gossip.initiated        query.routed
trust.below_floor      gossip.received         query.deferred
                       gossip.degraded         escalation.triggered
policy.decision_made   consensus.started       handoff.prepared
policy.violated        consensus.reached       handoff.completed
policy.conflict        consensus.failed        handoff.failed
                       consensus.vote_cast     handoff.rolled_back
agent.registered / agent.deregistered / agent.overloaded
```

**`AgentSession`** — one-stop wiring:
```python
session = AgentSession.create(
    bus         = bus,
    agent_id    = "orchestrator",
    trust_net   = TrustNetwork(),
    coordinator = HumilityAwareCoordinator(agents={}),
    gossip      = GossipClarityProtocol(agents={}, max_hops=4),
    policy_mgr  = PolicyAlignmentManager(),
    consensus   = ConsensusProtocol(),
)

# Standard cross-module reactions are wired automatically:
#   TRUST_UPDATED        → refresh humility routing hints
#   GOSSIP_RECEIVED      → trigger policy re-evaluation
#   POLICY_DECISION_MADE → record in trust network
#   CONSENSUS_REACHED    → update reputation for winning agents
#   CONSENSUS_FAILED     → flag divergence; suggest re-gossip
#   QUERY_ROUTED         → delegation step logged in trust network
#   HANDOFF_*            → state integrity tracking
```

**Convenience methods on `AgentSession`:**
```python
session.publish_trust_update("agent-B", new_score=0.82)
session.publish_gossip("q3-revenue", confidence=0.88)
session.run_consensus(votes, round_id="deploy-42")  # runs + publishes result
session.publish_handoff(EventType.HANDOFF_COMPLETED, packet_id="pkt-...")
```

---

## Updated: `pact_ax/coordination/__init__.py`

The 18 original exports are unchanged.  Added 9 new exports:

```python
# Consensus
ConsensusProtocol, ConsensusStrategy, ConsensusOutcome, ConsensusResult, Vote

# Coordination bus
CoordinationBus, CoordinationEvent, EventType, AgentSession
```

---

## Tests

```
tests/test_consensus.py         ~45 tests
tests/test_coordination_bus.py  ~30 tests (incl. full end-to-end round)
```

Coverage: all 4 strategies · all outcome states · trust-weighted voting ·
abstentions · serialisation roundtrip · subscriber isolation · error policy ·
event log filtering · session wiring/unwiring · end-to-end gossip→policy→consensus.

---

## Files changed

```
pact_ax/coordination/
├── __init__.py              ← updated (9 new exports)
├── consensus.py             ← NEW
├── coordination_bus.py      ← NEW
├── gossip_clarity.py        ← unchanged
├── humility_aware.py        ← unchanged
├── policy_alignment.py      ← unchanged
└── trust_primitives.py      ← unchanged

tests/
├── test_consensus.py           ← NEW
└── test_coordination_bus.py    ← NEW
```
