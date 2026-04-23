# PACT-AX · Two-Agent Billing Escalation Demo

Two real Claude agents — a tier-1 support agent and a billing specialist —
coordinate through the PACT-AX layer to handle a customer billing dispute.
The demo is designed to accumulate history across runs: trust, calibration,
and capability confidence all evolve from real outcomes, not configuration.

---

## The scenario

**Customer:** Sarah Chen  
**Issue:** Duplicate charge — billed twice for the Pro plan ($349.99) on March 15  
**Agent-A:** Tier-1 support. Refund authority capped at $100. No payment processor access.  
**Agent-B:** Billing specialist. Full refund authority. Payment processor access.

---

## The 8-step flow

```
Agent-A                    PACT-AX                      Agent-B
   │                          │                             │
   │── registers ────────────►│                             │
   │                          │◄── registers ───────────────│
   │                          │                             │
   │  LLM call: what do I do? │                             │
   │── decision ─────────────►│ evaluate → align            │
   │                          │                             │
   │◄── capability check ─────│                             │
   │    (approaching limit?)   │                             │
   │                          │                             │
   │── prepare handoff ───────►│                             │
   │── send packet ───────────►│── receive ─────────────────►│
   │                          │                             │
   │                          │     LLM call: resolve case  │
   │                          │◄── outcome ─────────────────│
   │◄── trust updated ────────│                             │
   │◄── calibration recorded ─│                             │
   │◄── capability adjusted ──│ (loop 3)                    │
```

---

## The three feedback loops

PACT-AX runs three feedback loops simultaneously. Each one feeds the next.

### Loop 1 · Trust evolution  *(live from run #1)*

Every successful handoff moves Agent-A's trust in Agent-B upward.
Every failed or unnecessary escalation moves it down.
Inactivity decays trust back toward neutral (0.500) over time.

```
Run  1:  Trust 0.500 → 0.600  (first successful handoff)
Run  2:  Trust 0.600 → 0.700  (ledger restored, history continues)
Run  5:  Trust 0.900 → 0.950  (ceiling at 0.95 — trust stays informative)
Run 11:  Trust 1.000 → 0.950  (ceiling applied retroactively)
```

Trust is bounded: **floor 0.05, ceiling 0.95**.  
At 1.0 or 0.0 there's nowhere to go — trust stops being useful.

### Loop 2 · Policy calibration  *(accumulates from run #1, meaningful at ~10)*

`PolicyLearning` records every decision outcome:
- Was Agent-A's decision correct?
- What confidence did it claim vs. what accuracy did it deliver?

```
After 11 runs:
  Total decisions:        11
  Correct:                10  (91% accuracy)
  Avg stated confidence:  76%
  Calibration error:      14%  (underconfident)
  Tendency:               underconfident
```

This data feeds directly into loop 3.

### Loop 3 · Capability confidence  *(closes at run #11, feeds from loop 2)*

The cross-loop connection. After each run, calibration data adjusts
Agent-A's stored capability score for `billing_dispute`:

```
adjustment  = actual_accuracy − avg_stated_confidence
            = 91% − 76%  =  +14%

capability  = base(0.70) + adjustment(0.14)  =  0.840
```

If the adjusted capability exceeds the handoff threshold (0.70),
Agent-A is allowed to handle the case directly — no escalation needed.
If a bad streak drops accuracy below stated confidence, capability
falls below threshold and escalation becomes mandatory again.

**This is the system earning its own trust.**  
Not configuration. Not hardcoded thresholds. Behavior derived from track record.

---

## Scenarios to watch

### Scenario A · Healthy escalation  *(most runs)*
Agent-A correctly identifies it can't handle a $349.99 dispute.
Decision: `escalate` or `investigate`. Handoff fires. Agent-B resolves.
Trust rises. Calibration records a correct decision.

### Scenario B · Capability threshold crossed  *(after ~15+ consistent runs)*
Loop 3 has pushed capability above threshold by enough margin that
`approaching_limit` stays False even when Agent-A states moderate confidence.
Agent-A resolves directly. No handoff. The system trusts its own track record.

### Scenario C · Bad streak  *(simulate by editing DRY_RUN_B outcome to "denied")*
Agent-B denies a case. `was_correct_to_escalate = False`. Calibration accuracy
drops. Loop 3 adjustment shrinks or goes negative. Capability score falls.
Next run: handoff threshold tightens automatically.

### Scenario D · Policy constraint blocks a weak decision  *(manual setup)*
Add a high-confidence constraint before running:

```python
from starlette.testclient import TestClient
from pact_ax.api.server import app
pax = TestClient(app)
pax.post("/policy/constraint", json={
    "name": "high-confidence-required",
    "description": "Only confident decisions pass",
    "min_confidence": 0.9,
})
```

Then run the demo. A `MODERATE` or `LOW` confidence decision will be blocked
by the policy layer and `final_decision` returns `NO_VALID_DECISIONS`.

---

## Run it

```bash
# From the repo root — real Claude agents
export ANTHROPIC_API_KEY=sk-ant-...
python examples/two_agent_demo/demo.py

# Dry-run — no API key needed, uses canned responses
python examples/two_agent_demo/demo.py --dry-run
```

History persists to `demo_history.db` in this directory.
Each run restores trust profiles and policy outcomes from the previous run.

To reset and start fresh:
```bash
rm examples/two_agent_demo/demo_history.db
```

---

## What the ledger stores

```
demo_history.db
├── trust_profiles      (TrustStore — AgentTrustProfile per agent pair)
└── policy_outcomes     (PolicyLearning.outcomes — one row per run)
```

Both are plain SQLite. You can inspect them directly:

```bash
sqlite3 examples/two_agent_demo/demo_history.db \
  "SELECT agent_id, outcome_json FROM policy_outcomes ORDER BY id"
```

---

## The point

A multi-agent system that only uses static trust scores and fixed thresholds
is just routing with extra steps. What makes coordination real is when the
system *updates its own behavior* based on what actually happened.

That's what these three loops do together:

```
Outcome recorded
    → Trust evolves          (who do I rely on?)
    → Calibration accumulates (how accurate am I?)
    → Capability adjusts      (what should I handle myself?)
    → Next decision is different from the last
```

Rare signal still needs a ledger to become legacy.
