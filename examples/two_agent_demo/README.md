# PACT-AX · Two-Agent Billing Escalation Demo

A real-world scenario: two Claude agents coordinate through the PACT-AX
coordination layer to handle a customer billing dispute.

## What happens

| Step | Who | What |
|------|-----|------|
| 1 | PACT-AX | Register Agent-A (tier-1) and Agent-B (billing specialist) |
| 2 | **Agent-A** (Claude) | Reads the billing dispute and decides: resolve or escalate? |
| 3 | PACT-AX policy | Evaluates and aligns the decision through the policy layer |
| 4 | PACT-AX | Agent-A senses its capability limit — triggers handoff |
| 5 | PACT-AX | State transferred from A → B with full context |
| 6 | **Agent-B** (Claude) | Receives the handoff, resolves the case |
| 7 | PACT-AX | Outcome recorded, Agent-A's trust in Agent-B evolves |

## Run it

```bash
# With real Claude agents
export ANTHROPIC_API_KEY=sk-ant-...
python examples/two_agent_demo/demo.py

# Dry-run (no API key needed — uses canned agent responses)
python examples/two_agent_demo/demo.py --dry-run
```

## What to watch for

- **Agent-A's decision** changes based on the case — sometimes it resolves directly,
  sometimes it escalates. Watch the confidence level and reasoning.
- **Trust score** starts at 0.500 (neutral) and moves up after a successful handoff.
  Run the demo multiple times and the trust will keep evolving.
- **Policy layer** would block a LOW-confidence decision if you add a constraint:
  ```bash
  # Try adding this before running:
  curl -X POST http://localhost/policy/constraint \
    -H 'Content-Type: application/json' \
    -d '{"name":"high-conf","description":"test","min_confidence":0.9}'
  ```
- **Agent-A calibration** — accuracy and tendency (overconfident/underconfident)
  accumulate across runs once you have an API key and record real outcomes.
