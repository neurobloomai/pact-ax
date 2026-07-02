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

## Why current agent networks work — and where they stop

Today's orchestrator/subagent patterns (Claude Code, LangGraph-style, CrewAI)
succeed because they are **stars, not meshes**:

- Subagents cannot talk to each other
- Nothing persists beyond the task
- Everything routes through one synthesizer audited by one human

Trust is never encoded — it is implied by topology. The human root *is* the
trust substrate. That is why it works, and that is exactly why it stops working
at scale.

Every implicit guarantee evaporates at the star→mesh transition: agents from
different owners, sessions, and organizations, with no shared human root, no
shared lineage, no mechanism to verify that a delegated agent still carries the
intent it was issued. The star is doing PACT's job by brute force. It works
until one human's working memory can no longer be the substrate.

That transition is where PACT-AX begins.

| Gap in star topology | PACT-AX primitive |
|---|---|
| Stale authorization ("token says you were trusted") | `TrustAlignmentCheck` — continuous n/n gate, not a T=0 snapshot |
| Unscoped context dumping between agents | `TrustContext` — scoped, propagatable intent-preservation contracts |
| Assumed alignment via shared lineage | `TrustIntent` — load-bearing constraints that travel with the delegation |
| Silent intent decay per hop | `verify_intent_integrity()` — detects omissions and modifications downstream |

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
    │   ├── TrustChainManager    — relational coherence across agent hops (A→B→C)
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

### 🔄 Trust-Gated State Transfer

Full handoff lifecycle with trust verification baked in — not bolted on after.

Every `receive()` runs an authoritative trust gate: the receiver checks its own `TrustManager` for the sender's history-backed score (not the heuristic the sender stamped on the packet), then optionally checks chain coherence via `TrustChainManager`. Broken chains are rejected. Degraded chains pass with a warning. Both checks surface in `TrustGateResult` on the `IntegrationResult`.

```python
from pact_ax.state import StateTransferManager, HandoffReason
from pact_ax.primitives import TrustManager, TrustChainManager

# Sender: TrustManager wires history-backed score into the packet
sender_tm = TrustManager(agent_id="agent-A")
sender    = StateTransferManager(agent_id="agent-A", trust_manager=sender_tm)
packet_id = sender.prepare("agent-B", state_data={"task": "analyse Q3"}, reason=HandoffReason.CONTINUATION)
packet    = sender.send(packet_id)

# Receiver: checks its own TrustManager + chain coherence before integrating
receiver_tm    = TrustManager(agent_id="agent-B")
chain_mgr      = TrustChainManager(trust_resolver=lambda f, t: receiver_tm.get_trust(t))
receiver       = StateTransferManager(
    agent_id="agent-B",
    trust_manager=receiver_tm,
    trust_chain_manager=chain_mgr,
    trust_floor=0.4,
)
result = receiver.receive(packet)

print(result.success)                          # True / False
print(result.trust_gate.sender_trust)         # score from receiver's TrustManager
print(result.trust_gate.sender_trust_source)  # "trust_manager" | "heuristic"
print(result.trust_gate.chain_state)          # "active" | "degraded" | "broken"
print(result.warnings)                        # degraded chain → warn but don't reject
```

**Trust gate layers:**
1. `validate()` — structural checks (address, TTL, non-empty state). Heuristic floor when no TrustManager.
2. `_run_trust_gate()` — receiver's own `TrustManager.get_trust(sender)` against `trust_floor`. Broken `TrustChain` → reject. Degraded → warn.
3. `TrustGateResult` — surfaces `sender_trust`, `sender_trust_source`, `chain_trust`, `chain_state`, `passed`, `rejection_reason` on every `IntegrationResult`.

Works without any trust primitives — falls back to heuristic, full backward compatibility.

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

### 🔗 Trust Chain

TrustManager answers point-in-time questions about pairs. TrustChain answers the next question: when Agent A → Agent B → Agent C, is the full chain still coherent — not at initialization, but now?

```python
from pact_ax.primitives import TrustChainManager

def resolver(from_id, to_id):
    return trust_managers[from_id].get_trust(to_id)

mgr = TrustChainManager(trust_resolver=resolver)

# Score without recording — chain_trust, coherence, weakest hop
score = mgr.score(["agent-a", "agent-b", "agent-c"])
print(score.chain_trust)    # geometric mean across hops
print(score.coherence)      # 1.0 = all hops equal strength
print(score.weakest_pair)   # ("agent-b", "agent-c")
print(score.state)          # active / degraded / broken

# Record a chain and verify it later
chain = mgr.record(["agent-a", "agent-b", "agent-c"])

# Re-verify against current trust — detects drift since baseline
verify = mgr.verify(chain.chain_id)
print(verify.state_changed)     # True if chain degraded
print(verify.hop_drift)         # per-hop baseline vs current
```

**Scoring model:**
- `chain_trust` — geometric mean of hop scores. A chain of [0.9, 0.3, 0.9] scores 0.63, not 0.9. Weak links cannot hide.
- `coherence` — 1.0 minus normalised std-dev. Flags uneven chains even when the average looks fine.
- `weakest_hop` — names the bottleneck agent relationship.
- `ChainState` — `ACTIVE` (≥0.7 trust, ≥0.6 coherence) / `DEGRADED` / `BROKEN` / `COMPLETED`.

REST API: `POST /trust-chain/score`, `POST /trust-chain/record`, `POST /trust-chain/{id}/verify`

MCP tools: `trust_chain_score`, `trust_chain_verify`

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

PACT-AX ships a 84-route FastAPI server.

### One-command start (Docker)

```bash
git clone https://github.com/neurobloomai/pact-ax
cd pact-ax
docker compose up
```

Server starts at `http://localhost:8000`. SQLite DBs are written to `./data/` and persist across restarts.

Swagger docs at `http://localhost:8000/docs`.

### Run without Docker

```bash
git clone https://github.com/neurobloomai/pact-ax
cd pact-ax
pip install -e .
uvicorn pact_ax.api.server:app --reload
```

### Environment variables

| Variable | Default | What it controls |
|---|---|---|
| `PACT_PORT` | `8000` | Host port (Docker only) |
| `PACT_ENFORCE_AUTH` | `0` | `1` = require API keys on every request |
| `PACT_TRUST_DB` | `trust.db` | SQLite path for trust scores |
| `PACT_CAP_DB` | `capabilities.db` | SQLite path for capability registry |
| `PACT_STORY_DB` | `story_keeper.db` | SQLite path for narrative memory |
| `PACT_MEMORY_DB` | `episodic.db` | SQLite path for episodic memory |
| `PACT_DLQ_DB` | `dlq.db` | SQLite path for dead letter queue |
| `PACT_ACCESS_DB` | `access.db` | SQLite path for API key store |

Copy `.env.example` to `.env` to configure locally.

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
- TrustChainManager — relational coherence across agent hops; geometric mean scoring, drift detection, chain state transitions (active/degraded/broken)
- StateTransferManager — trust-gated handoff lifecycle; receiver-side TrustManager + TrustChain verification, TrustGateResult on every IntegrationResult
- ConsensusProtocol — weighted vote, quorum, unanimous, confidence-threshold
- CoordinationBus — event-driven pub/sub
- CapabilityRegistry — skill registration, semantic search, tag filtering
- AgentRouter — trust-weighted routing, fuzzy search
- EpisodicMemory — record, recall, summary, partner analytics
- DeadLetterQueue — enqueue, retry, exhaustion, exponential backoff
- REST API — 84 routes across all primitives
- **pact-ax-client** — Python SDK on PyPI (`pip install pact-ax-client`)
- Docker — `docker compose up` one-command start; `./data/` volume for SQLite persistence

### 🎯 Next
- TypeScript SDK (`npm install pact-ax-client`)
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
