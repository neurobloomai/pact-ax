# Four-Primitive Integration Demo

Shows all four PACT-AX primitives working together at a single handoff seam.

## The scenario

Marcus is three months into a career pivot, coached by Agent-A (generalist).
A crisis arrives: competing job offer, 48-hour deadline, a non-compete clause
he didn't fully read. Agent-A hits its capability limit. Handoff to Agent-B
(compensation specialist) must happen — but Marcus's story has to survive it.

## The four primitives

| Primitive | What it does here | What breaks without it |
|---|---|---|
| **StoryKeeper** | Maintains 3 months of narrative across the handoff | Agent-B sees an offer negotiation, not a person |
| **StateTransfer** | Carries structured offer terms cleanly and verifiably | Agent-B starts cold |
| **ContextShare** | Surfaces only the context relevant to Agent-B's role | Agent-B is overwhelmed or under-informed |
| **Trust** | Gates the handoff; records the outcome for future routing | Any agent hands off to any agent, no accountability |

## The core moment

Step 6 shows Agent-B's context three ways — without story, with story, with story + pact-hx memory.
Step 7 shows how Agent-B responds differently at each level.

The progression:
- Without story → Agent-B saw an offer negotiation.
- With story → Agent-B saw Marcus.
- With story + pact-hx → Agent-B saw the pattern Marcus can't see himself.

## Run

```bash
# With real LLM calls
export ANTHROPIC_API_KEY=sk-ant-...
python examples/four_primitive_demo/demo.py

# Dry-run (no API key needed)
python examples/four_primitive_demo/demo.py --dry-run
```
