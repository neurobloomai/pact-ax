# PACT-AX: Agent Collaboration Layer

### PACT-AX is the trust layer that makes agent coordination persistent, auditable, and drift-resistant.

### Part of the neurobloom.ai Open Source Ecosystem

![CI](https://github.com/neurobloomai/pact-ax/actions/workflows/ci.yml/badge.svg)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-743%20passing-brightgreen)
![PyPI](https://img.shields.io/pypi/v/pact-ax-client?label=pact-ax-client)

---

## Use the SDK

The fastest way to build on PACT-AX is the Python SDK — no server wiring required for your first experiment:

```bash
pip install pact-ax-client
```

```python
from pact_ax_client import Agent

agent = Agent("my-agent", base_url="http://localhost:8000")
agent.register_capability("contract_review", description="Reviews NDAs")

decision = agent.route("contract_review")
if decision.routed:
    result = agent.handoff(decision.best_agent, state_data={"doc": "..."})
    agent.remember("contract_review", partner_id=decision.best_agent, outcome="positive")
```

→ Full SDK docs: [neurobloomai/pact-ax-client](https://github.com/neurobloomai/pact-ax-client)

---

## The gap nobody named

The industry is solving token delegation — a signed credential at initialization, a binary check at T=0. That's authentication dressed as trust. A snapshot.

PACT-AX solves trust as a continuous process. Not "was this agent authorized?" but "is this relationship still coherent — across every hop, every handoff, every state change?"

Token says: you were trusted.  
PACT-AX says: are you still trustworthy?

This is the category error the entire industry is making right now. Every framework — LangChain, LangGraph, CrewAI, AutoGen — assumes trust at initialization and never verifies it again.

> "Safety is a moment. Trust is duration."

---

## What problem does this solve?

When you build a system with more than one AI agent, you hit three problems immediately — but underneath all three is one root cause: nobody is tracking whether agents remain trustworthy over time. PACT-AX is the substrate that makes trust computable, persistent, and continuous.

1. **Who handles this?** — no way to discover which agent has the right skill
2. **Can I trust the answer?** — no persistent record of how agents have performed before
3. **How do I hand off context?** — passing state between agents loses continuity

PACT-AX solves all three. It's a FastAPI server (84 routes) that your agents call to register capabilities, track trust, route tasks, hand off state, and record episodic memory. The Python SDK (`pact-ax-client`) wraps the API so you don't touch HTTP directly.

---

## The Category Error

The industry is solving token delegation — a signed credential, a binary check at T=0. That is authentication dressed as trust. A snapshot.

The question is not "was this agent authorized at initialization?"

The question is "is this relationship still coherent — across every interaction, every hop, every state change?"

**Token says: you were trusted.**  
**PACT-AX says: are you still trustworthy?**

Token delegation is a safety primitive. It answers a moment. The entire field is solving safety and calling it trust.

PACT-AX operates at the actual trust layer: duration, not moment.

> **Safety is a moment. Trust is duration.**

---

## Architecture

```
neurobloom.ai Ecosystem
├── PACT-HX (Human Experience Layer)
│   └── Personalized memory, emotional context, adaptive communication
│
└── PACT-AX (Agent Collaboration Layer)  [This Repository]
    ├── pact_ax.primitives
    │   ├── StoryKeeper          — narrative continuity across turns
    │   ├── ContextShareManager  — trust-aware context exchange
    │   ├── TrustManager         — network-wide trust scoring (persistent SQLite)
    │   ├── CapabilityRegistry   — agent skill registration and discovery
    │   └── AgentRouter          — trust-weighted task routing
    ├── pact_ax.state
    │   ├── StateTransferManager — full handoff lifecycle (prepare → send → receive)
    │   └── EpistemicStateTransfer — knowledge + confidence fidelity
    ├── pact_ax.coordination
    │   ├── ConsensusProtocol    — weighted-vote multi-agent decisions
    │   └── CoordinationBus      — event-driven agent messaging
    └── pact_ax.api              — 84-route REST API (FastAPI)
        ├── /capabilities        — register, find, search, deregister skills
        ├── /trust               — get, update, network trust, insights
        ├── /route               — trust-weighted and capability-only routing
        ├── /memory/episodes     — record and recall episodic memory
        ├── /consensus           — stateless run + stateful sessions
        ├── /dlq                 — dead letter queue with exponential backoff
        └── /transfer            — prepare, send, receive, checkpoint
```

---

## Key Features

### 🗺️ Capability Registry + Router

Register agent skills and route tasks to the best trusted+capable agent:

```python
from pact_ax.primitives import CapabilityRegistry, AgentRouter

registry = CapabilityRegistry()
registry.register("agent-a", "contract_review", tags=["legal"])
registry.register("agent-b", "contract_review", tags=["legal"])
registry.register("agent-b", "tax_analysis",   tags=["finance"])

router = AgentRouter(capability_db=":memory:", trust_db=":memory:")
decision = router.route(from_agent="orch", skill="contract_review", min_trust=0.6)
print(decision.best_agent, decision.strategy_used)
# "agent-b"  "trust_weighted"
```

### 📖 Story Keeper

Maintains narrative continuity across conversation turns:

```python
from pact_ax.primitives import StoryKeeper

keeper = StoryKeeper(agent_id="agent-001", session_id="user-session-42")
keeper.process_turn("I want to build a startup in the health space")
keeper.process_turn("What should I focus on first?")

story = keeper.get_story_state()
# {"arc": "Collaboration: startup, health", "themes": [...], ...}
```

### 🤝 Context Sharing

Trust-aware context exchange with validated packets:

```python
from pact_ax.primitives import ContextShareManager, ContextType, Priority

manager = ContextShareManager("agent-001", agent_type="support_specialist",
                               capabilities=["nlp", "customer_support"])
packet = manager.create_context_packet(
    target_agent="agent-002",
    context_type=ContextType.TASK_KNOWLEDGE,
    payload={"current_task": "customer_support", "context": "billing issue"},
    priority=Priority.HIGH,
)
```

### 🔄 State Transfer

Full handoff lifecycle — prepare, send, receive, checkpoint:

```python
from pact_ax.state import StateTransferManager, HandoffReason

sender = StateTransferManager(agent_id="agent-A")
packet_id = sender.prepare("agent-B", state_data={"task": "analyse Q3"}, reason=HandoffReason.CONTINUATION)
packet = sender.send(packet_id)

receiver = StateTransferManager(agent_id="agent-B")
result = receiver.receive(packet)
print(result.success, result.integrated_state)
```

### 🛡️ Trust Scoring

Persistent, network-wide trust that evolves from real collaboration outcomes:

```python
from pact_ax.primitives import TrustManager

tm = TrustManager(agent_id="agent-001")
tm.update_trust("agent-002", "positive", "task_knowledge")
score = tm.get_trust("agent-002")       # overall
inferred = tm.get_network_trust("agent-unknown")   # transitive
trusted = tm.get_trusted_agents(min_trust=0.7)
```

### 🧠 Episodic Memory

Record and recall past interactions with filters:

```python
from pact_ax.primitives import EpisodicMemory

mem = EpisodicMemory()
mem.record("agent-a", "reviewed_nda", partner_id="agent-b",
           outcome="positive", importance=0.8, tags=["legal"])
episodes = mem.recall("agent-a", outcome="positive", min_importance=0.5)
summary  = mem.summary("agent-a")
# {"total_episodes": 12, "avg_importance": 0.72, "outcome_breakdown": {...}}
```

### 📬 Dead Letter Queue

Park failed deliveries for retry with exponential backoff:

```python
from pact_ax.primitives import DeadLetterQueue

dlq = DeadLetterQueue(max_attempts=3, base_seconds=30)
entry = dlq.enqueue("pkt-xyz", "agent-a", "agent-b", reason="timeout")
dlq.retry(entry.id)   # next_retry = now + 30s, 60s, 120s...
```

### 🗳️ Consensus Protocol

Weighted-vote multi-agent decisions:

```python
from pact_ax.coordination.consensus import ConsensusProtocol, ConsensusStrategy, Vote

proto = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE)
votes = [
    Vote("agent-A", "deploy-v2", confidence=0.85),
    Vote("agent-B", "deploy-v2", confidence=0.80),
    Vote("agent-C", "hold",      confidence=0.65),
]
result = proto.run(votes, trust_scores={"agent-A": 0.9, "agent-B": 0.85, "agent-C": 0.7})
print(result.reaching, result.winning_decision)  # True, "deploy-v2"
```

---

## REST API

PACT-AX ships a 84-route FastAPI server. Run it:

```bash
git clone https://github.com/neurobloomai/pact-ax
cd pact-ax
pip install -r requirements.txt
uvicorn pact_ax.api.server:app --reload
```

Swagger docs at `http://localhost:8000/docs`.

Or use the SDK instead of calling HTTP directly:

```bash
pip install pact-ax-client
```

---

## MCP Server

PACT-AX ships an MCP server that exposes trust primitives as tools — so Claude Code, Cursor, and any MCP-native client can call trust, routing, memory, and handoff operations directly without touching HTTP.

**Start the PACT-AX server first:**

```bash
uvicorn pact_ax.api.server:app --reload
```

**Run the MCP server:**

```bash
python -m pact_ax.mcp.server
```

**Wire into Claude Code** (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "pact-ax": {
      "command": "python",
      "args": ["-m", "pact_ax.mcp.server"],
      "env": { "PACT_AX_URL": "http://localhost:8000" }
    }
  }
}
```

**Wire into Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "pact-ax": {
      "command": "python",
      "args": ["-m", "pact_ax.mcp.server"],
      "env": { "PACT_AX_URL": "http://localhost:8000" }
    }
  }
}
```

**Available tools (12):**

| Tool | What it does |
|---|---|
| `trust_get` | Get trust score between two agents |
| `trust_update` | Record a collaboration outcome and update trust |
| `trust_network` | Get transitive network trust for an indirect agent |
| `trust_insights` | Full trust relationship breakdown for an agent |
| `trust_agents` | List all agents trusted above a threshold |
| `route_task` | Route a task to the best trusted+capable agent (exact skill) |
| `route_any` | Route by fuzzy keyword search across all capabilities |
| `capability_register` | Register a skill for an agent |
| `capability_find` | Find agents registered for a skill |
| `memory_record` | Record an episodic interaction |
| `memory_recall` | Recall past episodes with filters |
| `transfer_prepare` | Prepare a state handoff packet |

**Environment:**

| Variable | Default | Description |
|---|---|---|
| `PACT_AX_URL` | `http://localhost:8000` | Base URL of the PACT-AX server |

---

## Demos

All runnable demos live in **[neurobloomai/pact-demos](https://github.com/neurobloomai/pact-demos)**:

```bash
git clone https://github.com/neurobloomai/pact-demos
cd pact-demos
pip install -r requirements.txt
python demos/capability_routing/demo.py
python demos/orchestrate_rest/demo.py
```

---

## Development

```bash
git clone https://github.com/neurobloomai/pact-ax
cd pact-ax
pip install -r requirements.txt
pytest tests/ -v                          # 743 tests
pytest tests/unit/ -v                     # unit only
pytest tests/integration/ -v             # integration only
```

---

## Roadmap

### ✅ Built
- StoryKeeper — narrative continuity, arc detection, multi-session persistence
- ContextShareManager — trust-aware context packets, capability sensing
- TrustManager — per-context trust, time-based decay, network reputation, **persistent SQLite**
- StateTransferManager — full handoff lifecycle, checkpoints, epistemic state transfer
- ConsensusProtocol — weighted vote, quorum, unanimous, confidence-threshold
- CoordinationBus — event-driven pub/sub
- CapabilityRegistry — skill registration, semantic search, tag filtering
- AgentRouter — trust-weighted routing, fuzzy search
- EpisodicMemory — record, recall, summary, partner analytics
- DeadLetterQueue — enqueue, retry, exhaustion, exponential backoff
- REST API — 84 routes across all primitives
- **pact-ax-client** — Python SDK on PyPI (`pip install pact-ax-client`)

### 🎯 Next
- TypeScript SDK (`npm install pact-ax-client`)
- Docker / one-command server setup
- Real integration (GitHub Actions, Slack bot)
- PACT-HX integration (Human Experience Layer)

---

## Package Structure

```
pact_ax/
├── primitives/
│   ├── story_keeper.py
│   ├── context_share/
│   ├── trust_score.py
│   ├── capability_registry.py
│   ├── agent_router.py
│   ├── episodic_memory.py
│   └── dead_letter_queue.py
├── state/
│   ├── state_transfer_manager.py
│   └── epistemic_transfer.py
├── coordination/
│   ├── consensus.py
│   └── coordination_bus.py
└── api/
    ├── server.py
    └── routes/
        ├── capabilities.py
        ├── trust.py
        ├── agent_router.py
        ├── episodic_memory.py
        ├── dead_letter.py
        ├── consensus.py
        └── transfer.py
```

---

## License

MIT — built by the [neurobloom.ai](https://neurobloom.ai) community.
