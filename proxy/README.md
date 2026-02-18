# PACT-AX Proxy

**Session Integrity Layer for MCP**

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌─────────────┐
│             │     │           PACT-AX PROXY             │     │             │
│   CURSOR    │────▶│  ┌─────────────────────────────┐   │────▶│  GITHUB     │
│  (Client)   │     │  │       Story Keeper          │   │     │  MCP Server │
│             │◀────│  │  - Session State            │   │◀────│             │
└─────────────┘     │  │  - Relational Context       │   │     └─────────────┘
                    │  │  - Trust Evolution          │   │
                    │  └─────────────────────────────┘   │
                    │  ┌─────────────────────────────┐   │
                    │  │    Coherence Monitor        │   │
                    │  │  - Drift Detection          │   │
                    │  │  - Pattern Anomalies        │   │
                    │  │  - Context Violations       │   │
                    │  └─────────────────────────────┘   │
                    │  ┌─────────────────────────────┐   │
                    │  │    State Transfer (RLP-0)   │   │
                    │  │  - Relational Primitives    │   │
                    │  │  - Trust Snapshots          │   │
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
```

## What This Demonstrates

1. **Session State Persistence** - Unlike stateless policy checks, PACT-AX maintains relational context across the entire session

2. **Coherence Monitoring** - Detects when agent behavior drifts from established patterns (not just policy violations, but *relational* violations)

3. **Trust Evolution Tracking** - Records how trust develops or degrades over interaction history

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (edit config.yaml)
cp config.example.yaml config.yaml

# Run proxy
python -m src.proxy

# Point Cursor to localhost:3000 instead of direct GitHub MCP
```

## Integration with Existing Security Tools

PACT-AX doesn't replace Wiz/WorkOS policy engines. It *enriches* them:

```
Policy Engine asks: "Is this action allowed?"
PACT-AX provides:   "Here's the relational context for that decision"
                    - Session history
                    - Trust trajectory  
                    - Coherence score
                    - Drift indicators
```

## Demo Scenario: Drift Detection

1. Agent establishes pattern: reading specific repo files
2. Agent behavior shifts: suddenly requesting org-wide access
3. Policy engine sees: valid token, permitted scope
4. PACT-AX sees: **relational drift** from established context
5. Alert: "Behavior inconsistent with session trust pattern"

---

*NeuroBloom.ai - Relational Intelligence Infrastructure*
