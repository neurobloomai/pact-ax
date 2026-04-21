# PACT-AX: Agent Collaboration Layer
### Part of the neurobloom.ai Open Source Ecosystem

![neurobloom.ai](https://img.shields.io/badge/neurobloom.ai-collaborative--intelligence-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active--development-orange)
![Tests](https://img.shields.io/badge/tests-125%20passing-brightgreen)

---

## Overview

PACT-AX provides primitives for safe collaboration, context sharing, and knowledge transfer between heterogeneous AI agents. Built on the principle that **trust scales while control just moves bottlenecks**, PACT-AX enables distributed AI collaboration that mirrors the best of human teamwork.

## Core Philosophy: EI + AI

**Emotional Intelligence + Artificial Intelligence**

PACT-AX integrates human collaboration wisdom with AI technical capabilities:
- **Jazz-like improvisation** for small-scale agent interactions
- **Symphonic coordination** for large-scale agent orchestration
- **MVI (Minimum Viable Intervention)** — maximum collaboration impact with minimal overhead
- **Organic trust building** through continuous interaction rather than one-time verification

---

## Architecture

### PACT Protocol Layers

```
neurobloom.ai Ecosystem
├── PACT-HX (Human Experience Layer)     [Planned]
│   ├── Collaborative improvisation frameworks
│   ├── Universal translator for human "operating systems"
│   ├── Designed serendipity with neural rewiring/unwiring
│   └── Leadership multiplication protocols
│
└── PACT-AX (Agent Communication Layer)  [This Repository]
    ├── pact_ax.primitives
    │   ├── StoryKeeper          — narrative continuity across turns
    │   ├── ContextShareManager  — trust-aware context exchange
    │   └── TrustManager         — network-wide trust scoring
    ├── pact_ax.state
    │   ├── StateTransferManager — full handoff lifecycle protocol
    │   └── EpistemicStateTransfer — knowledge + confidence fidelity
    └── pact_ax.coordination
        ├── ConsensusProtocol    — multi-agent decision making
        └── CoordinationBus      — event-driven agent messaging
```

---

## Key Features

### 📖 Story Keeper
Maintains narrative continuity across conversation turns so agents don't lose context between sessions.

```python
from pact_ax.primitives import StoryKeeper

keeper = StoryKeeper(agent_id="agent-001", session_id="user-session-42")

# Process turns — story state evolves automatically
keeper.process_turn("I want to build a startup in the health space")
keeper.process_turn("What should I focus on first?")

# Snapshot the story for handoff or persistence
story = keeper.get_story_state()
# {"characters": {...}, "arc": "Collaboration: startup, health",
#  "themes": ["startup", "health", "focus"], "context": "...", "last_beat": "..."}

# Restore in a new session
new_keeper = StoryKeeper(agent_id="agent-001", session_id="user-session-42")
new_keeper.load_story_state(story)

# Reset when starting fresh
keeper.reset_story()
```

---

### 🤝 Context Sharing
Trust-aware context exchange with validated packets, lineage tracking, and capability sensing.

```python
from pact_ax.primitives import ContextShareManager, ContextType, Priority

manager = ContextShareManager(
    agent_id="agent-001",
    agent_type="support_specialist",
    capabilities=["nlp", "customer_support"],
)

# Create a validated context packet
packet = manager.create_context_packet(
    target_agent="agent-002",
    context_type=ContextType.TASK_KNOWLEDGE,
    payload={
        "current_task": "customer_support",
        "priority": "high",
        "context": "User needs help with billing issue",
    },
    priority=Priority.HIGH,
)

# Assess trust before sharing sensitive context
trust = manager.assess_trust(
    target_agent="agent-002",
    context_type=ContextType.EMOTIONAL_STATE,
    current_situation={"stakes": "high"},
)
# {"final_trust": 0.62, "recommendation": "caution", ...}

# Sense capability limits for proactive handoff
status = manager.sense_capability_limit("billing_resolution", confidence_threshold=0.7)
# {"approaching_limit": True, "recommendation": "prepare_handoff", ...}

# Record outcomes to evolve trust over time
manager.record_collaboration_outcome("agent-002", ContextType.TASK_KNOWLEDGE, "positive")
```

---

### 🔄 State Transfer
Full handoff lifecycle — prepare, send, receive, integrate, rollback — with story awareness and epistemic state fidelity.

```python
from pact_ax.state import StateTransferManager, HandoffReason
from pact_ax.primitives import StoryKeeper

# Sender side
story_keeper = StoryKeeper("agent-A")
sender = StateTransferManager(agent_id="agent-A", story_keeper=story_keeper)

packet_id = sender.prepare(
    to_agent_id="agent-B",
    state_data={"task": "analyse Q3 revenue", "progress": 0.6},
    reason=HandoffReason.CONTINUATION,
    context={"priority": "high"},
)
packet = sender.send(packet_id)

# Receiver side
receiver = StateTransferManager(agent_id="agent-B")
result = receiver.receive(packet)

if result.success:
    print(result.integrated_state)  # full state + narrative + epistemic context

# Checkpoint before risky operations
ckpt_id = sender.checkpoint(label="before_v2_deploy", state_data=current_state)
sender.restore(ckpt_id)  # roll back if needed
```

**Convenience wrapper** (single-call API for simpler use cases):

```python
# prepare_handoff / receive_handoff wrap the full lifecycle
transfer = sender.prepare_handoff(
    target_agent="agent-B",
    state_data={"task": "billing_resolution", "progress": 0.75},
    handoff_reason="continuation",
)
confirmation = receiver.receive_handoff(transfer)
# {"received": True, "story_integrated": True, "ready_to_continue": True, ...}
```

---

### 🛡️ Trust Scoring
Network-wide trust that evolves from real collaboration outcomes, decays with inactivity, and infers reputation transitively.

```python
from pact_ax.primitives import TrustManager
from pact_ax.primitives.context_share import CollaborationOutcome, ContextType

tm = TrustManager(agent_id="agent-001")

# Record outcomes — trust evolves automatically
tm.update_trust("agent-002", CollaborationOutcome.POSITIVE, ContextType.TASK_KNOWLEDGE)
tm.record_outcome("agent-002", "negative", ContextType.HANDOFF_REQUEST)

# Query scores
score = tm.get_trust("agent-002")                                # overall
score = tm.get_trust("agent-002", ContextType.TASK_KNOWLEDGE)   # context-specific

# Decay inactive relationships toward neutral
tm.decay_trust(days_inactive=7)

# Network-level reputation (infers trust for unknown agents)
inferred = tm.get_network_trust("agent-unknown")

# Find your most trusted collaborators
trusted = tm.get_trusted_agents(min_trust=0.7, context_type=ContextType.TASK_KNOWLEDGE)

# Full insights across all relationships
insights = tm.get_trust_insights()
```

---

### 🗳️ Consensus Protocol
Structured multi-agent decision making with pluggable strategies and deadlock detection.

```python
from pact_ax.coordination.consensus import ConsensusProtocol, ConsensusStrategy, Vote

proto = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)

votes = [
    Vote("agent-A", "deploy-v2", confidence=0.85, reasoning="metrics look good"),
    Vote("agent-B", "deploy-v2", confidence=0.80, reasoning="tests pass"),
    Vote("agent-C", "hold",      confidence=0.65, reasoning="need more data"),
]

result = proto.run(votes, trust_scores={"agent-A": 0.9, "agent-B": 0.85, "agent-C": 0.7})

if result.reached:
    print(result.winning_decision)   # "deploy-v2"
    print(result.confidence_score)   # 0.83
else:
    print(result.outcome)            # DEADLOCK or ESCALATE_TO_HUMAN
```

Supported strategies: `WEIGHTED_VOTE`, `QUORUM`, `UNANIMOUS`, `CONFIDENCE_THRESHOLD`.

---

### 🚌 Coordination Bus
Event-driven pub/sub for loosely coupled agent coordination.

```python
from pact_ax.coordination.coordination_bus import CoordinationBus, AgentMessage

bus = CoordinationBus()

# Subscribe to message types
bus.subscribe("handoff.requested", lambda msg: handle_handoff(msg))

# Publish events
bus.publish(AgentMessage(
    sender="agent-A",
    message_type="handoff.requested",
    payload={"to": "agent-B", "task": "billing_resolution"},
))
```

---

## Quick Start

```python
from pact_ax.primitives import (
    StoryKeeper, ContextShareManager, TrustManager, ContextType, Priority
)
from pact_ax.state import StateTransferManager, HandoffReason

# 1. Build story context through conversation
keeper = StoryKeeper("agent-001")
keeper.process_turn("Help me analyse our Q3 revenue numbers")

# 2. Share context with another agent
manager = ContextShareManager("agent-001", agent_type="analyst")
packet = manager.create_context_packet(
    target_agent="agent-002",
    context_type=ContextType.TASK_KNOWLEDGE,
    payload={"task": "Q3 revenue analysis", "progress": 0.4},
)

# 3. Hand off when approaching capability limits
status = manager.sense_capability_limit("financial_modelling")
if status["approaching_limit"]:
    sender = StateTransferManager("agent-001", story_keeper=keeper)
    pid = sender.prepare(
        to_agent_id="agent-002",
        state_data={"task": "Q3 analysis", "progress": 0.4},
        reason=HandoffReason.ESCALATION,
    )
    packet = sender.send(pid)

# 4. Receive on the other side
receiver = StateTransferManager("agent-002")
result = receiver.receive(packet)
print(result.success, result.integrated_state)
```

---

## Installation

```bash
pip install pact-ax
```

Or from source:

```bash
git clone https://github.com/neurobloomai/pact-ax.git
cd pact-ax
pip install -e .
```

---

## Development Roadmap

### ✅ Implemented
- [x] **StoryKeeper** — narrative continuity, arc detection, multi-session persistence
- [x] **ContextShareManager** — trust-aware context packets, capability sensing, handoff preparation
- [x] **TrustManager** — per-context trust tracking, time-based decay, network reputation
- [x] **StateTransferManager** — full handoff lifecycle (prepare → send → receive → integrate → rollback), checkpoints, epistemic state transfer
- [x] **ConsensusProtocol** — weighted vote, quorum, unanimous, confidence-threshold strategies
- [x] **CoordinationBus** — event-driven pub/sub for agent coordination
- [x] **EpistemicStateTransfer** — knowledge + confidence fidelity across handoffs

### 🔄 In Progress
- [ ] Policy alignment — conflict resolution between agent policies
- [ ] Persistent trust store — trust scores that survive process restarts
- [ ] REST API layer — HTTP endpoints for cross-process agent coordination
- [ ] `pact_ax.primitives.context_share` schema validation (Pydantic)

### 🎯 Planned
- [ ] PACT-HX integration (Human Experience Layer)
- [ ] Real-time collaboration analytics dashboard
- [ ] Cross-platform agent discovery registry
- [ ] Jazz ↔ Symphony mode auto-detection
- [ ] Multi-agent orchestration patterns

---

## Package Structure

```
pact_ax/
├── primitives/
│   ├── story_keeper.py          # StoryKeeper
│   ├── context_share/           # ContextShareManager + full schema types
│   │   ├── manager.py
│   │   └── schemas.py           # ContextPacket, AgentIdentity, TrustEvolution, ...
│   └── trust_score.py           # TrustManager
├── state/
│   ├── state_transfer_manager.py  # StateTransferManager (canonical)
│   └── epistemic_transfer.py      # EpistemicStateTransfer
└── coordination/
    ├── consensus.py             # ConsensusProtocol
    └── coordination_bus.py      # CoordinationBus
```

---

## Philosophical Foundations

### The Collaboration Spectrum
- **Individual Mastery** → **Small Group Jazz** → **Large Scale Symphony**
- **Solo thinking** → **Intimate collaboration (3-4 agents)** → **Orchestrated coordination (100+ agents)**

### Learning Through Iteration
- **No failures, only iterations** of learning and expansion
- **Always arriving imperfect** but arriving beautifully
- **Organic timing** over forced milestones

### Trust as Infrastructure
- **Continuous trust building** through authentic interaction
- **Network effects** — each successful collaboration strengthens the whole
- **Quality over quantity** in collaboration partnerships

---

## Contributing

We welcome contributions from developers who share our vision of joyful, abundant collaboration between AI agents.

**Our Approach:**
- **Organic development** — let features emerge from real needs
- **Both technical excellence AND human wisdom** — EI+AI integration
- **Open source abundance** — share knowledge freely to create more value for everyone

See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

---

## Research & Inspiration

PACT-AX draws inspiration from diverse sources:
- **Organizational Learning Theory** (Ray Dalio's Principles)
- **Jazz Improvisation Dynamics** (collaborative creativity research)
- **Abundance Economics** (Naval Ravikant's leverage principles)
- **Systems Thinking** (complex adaptive systems)
- **Contemplative Traditions** (patience, presence, organic unfolding)

---

## Community

- **GitHub Discussions**: Share ideas and collaborate on features
- **Discord**: Real-time conversation with other builders
- **Newsletter**: Updates on neurobloom.ai ecosystem development

---

## License

MIT License — see [LICENSE](LICENSE) file.

Built with 🎵 by the neurobloom.ai community.

*Where Artificial Intelligence meets Emotional Intelligence, and collaboration becomes an art form.*

---

## Contact

**neurobloom.ai Team**
- Email: founders@neurobloom.ai
- Website: neurobloom.ai
- GitHub: [@neurobloomai](https://github.com/neurobloomai)

---

*"We are conduits of creation, building the infrastructure for human potential in the AI age."*
