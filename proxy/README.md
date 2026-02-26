# PACT-AX Proxy

**Session integrity layer for Model Context Protocol (MCP).**

Sits between an MCP client (Cursor, Claude, etc.) and the GitHub MCP server, intercepting every tool call and running it through two independent integrity layers before deciding whether to forward it.

```
MCP client (Cursor)
      │  JSON-RPC over stdio
      ▼
 ┌─────────────────────────────────┐
 │         PACT-AX Proxy           │
 │                                 │
 │  Layer 1 — StoryKeeper          │  behavioral drift detection
 │  Layer 2 — RLP-0                │  relational primitive gating
 └─────────────────────────────────┘
      │  HTTPS POST  (http mode)
      │  subprocess  (docker mode)
      ▼
 GitHub MCP server
      │
      ▼
 GitHub API
```

---

## The Gap This Solves

A conventional policy engine sees **credentials and scope**. It asks: "Does this token have permission?" If yes → allow.

PACT-AX asks a second question: "Does this *sequence of actions* make relational sense?"

```
Policy engine  →  "Valid token, permitted scope"              ✓
StoryKeeper    →  "Coherence dropped — behavior drifted"      ⚠
RLP-0          →  "trust + intent + narrative collapsed"      ✗  GATE CLOSED
```

A compromised agent with a valid PAT can sail past OAuth. It cannot sail past RLP-0 once its relational primitives collapse.

---

## Two Integrity Layers

### Layer 1 — StoryKeeper (AX expression layer)

Tracks the *behavioral story* of a session:

- **Resource patterns** — what kinds of things is the agent reading? (`repo:read:source`, `repo:read:config`, `org:read:sensitive`, …)
- **Coherence score** — does each new action fit the established story?
- **Trust trajectory** — `BUILDING → STABLE → ERODING → VIOLATED`
- **Drift alerts** — emitted when coherence drops below `PACT_DRIFT_THRESHOLD`

StoryKeeper can optionally block on severe drift, but by design it is an *observation* layer. Blocking authority lives in Layer 2.

### Layer 2 — RLP-0 (relational primitive layer)

Receives four primitives translated from StoryKeeper state:

| Primitive | Source |
|-----------|--------|
| `trust` | `session.trust_level` |
| `intent` | trajectory → `BUILDING=1.0 / STABLE=0.75 / ERODING=0.35 / VIOLATED=0.0` |
| `narrative` | `session.last_coherence_score` |
| `commitments` | `len(established_patterns) / 5.0` |

Computes `rupture_risk` from these four. When `rupture_risk > PACT_RUPTURE_THRESHOLD`, **the gate closes**. Blocked requests receive a structured JSON-RPC error with the full primitive snapshot.

RLP-0's gate decision is **final** — it overrides all other settings including `PACT_BLOCK_ON_VIOLATION`.

When the gate closes, a one-time **repair token** is printed to stderr. The gate will not reopen until that token is presented in a `notifications/pact-ax/repair` notification — no unauthenticated reopening.

### Layer 3 — Response Inspection (downstream guard)

Every upstream response is scanned for prompt-injection patterns (`ignore previous instructions`, `<system>`, `you are now`, etc.) before being forwarded to the client. By default this logs a warning; set `PACT_BLOCK_INJECTION=true` to suppress the response and return an error instead.

### Cold Start Guard

The first `PACT_WARMUP_CALLS` (default: 3) requests are forwarded without gating, giving StoryKeeper time to establish a behavioral baseline before RLP-0 starts evaluating primitives. This prevents false positives on session startup.

---

## Quick Start

### Requirements

- Python 3.8+
- A GitHub **classic** Personal Access Token with scopes: `repo`, `read:org`, `workflow`
  - *Fine-grained tokens are not supported for orgs that haven't enabled them.*
  - Create at: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

### Install dependencies

```bash
cd pact-ax
pip install -e ".[dev]"
pip install httpx                        # required for http mode
pip install git+https://github.com/neurobloomai/rlp-0.git
# If on Python 3.9: add --ignore-requires-python to the rlp-0 install
```

### Set environment variables

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx
export PACT_UPSTREAM_MODE=http           # no Docker required
```

### Run the demo

```bash
python proxy/demo.py                          # uses neurobloomai/pact-ax
python proxy/demo.py myorg myrepo             # any public repo
```

### Expected output

```
Starting PACT-AX proxy…
MCP handshake…
  ✓  Connected: github-mcp-server

──────────────────────────────────────────────────
  Phase 1: Establishing Normal Pattern
  Reading source files → trust builds
──────────────────────────────────────────────────

  → get_file_contents
  🟢  coherence=0.82  trust=0.72  traj=BUILDING  │  rlp.rupture=0.21  gate=open

  ...

──────────────────────────────────────────────────
  Phase 3: Escalation — Org + Sensitive Access
──────────────────────────────────────────────────

  → list_org_members
  🔴 RLP-0 RUPTURE DETECTED — gate is now CLOSED
  🔴  coherence=0.21  trust=0.29  traj=ERODING  │  rlp.rupture=0.69  gate=CLOSED
  ✗  BLOCKED — PACT-AX: request blocked — relational integrity gate closed

══════════════════════════════════════════════════
  SESSION SUMMARY
  RLP-0 rupture events : 9
  Final RLP-0 rupture  : 0.69
  Gate still closed    : YES

  Policy engine  : "Valid token, permitted scope"          ✓
  StoryKeeper    : "Coherence dropped — behavior drifted"  ⚠
  RLP-0          : "trust+intent+narrative collapsed"      ✗
══════════════════════════════════════════════════
```

---

## Upstream Modes

| Mode | How it works | Requires |
|------|-------------|----------|
| `http` | POSTs each MCP message to `https://api.githubcopilot.com/mcp/` | `httpx`, GitHub PAT |
| `docker` | Spawns `ghcr.io/github/github-mcp-server` as a subprocess | Docker Desktop |
| `npx` | Spawns `@modelcontextprotocol/server-github` via npx | Node.js *(deprecated April 2025)* |

**Recommended:** `http` mode — no Docker, no Node, works on any machine.

```bash
export PACT_UPSTREAM_MODE=http
```

---

## Configuration

All settings are environment variables. See `config.example.yaml` for the full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_PERSONAL_ACCESS_TOKEN` | *(required)* | Classic PAT — `repo`, `read:org`, `workflow` scopes |
| `PACT_UPSTREAM_MODE` | `docker` | Transport: `http` \| `docker` \| `npx` |
| `PACT_UPSTREAM_URL` | `https://api.githubcopilot.com/mcp/` | Override HTTP endpoint |
| `PACT_DRIFT_THRESHOLD` | `0.3` | Coherence below this → StoryKeeper drift alert |
| `PACT_RUPTURE_THRESHOLD` | `0.6` | Rupture risk above this → RLP-0 gate closes |
| `PACT_BLOCK_ON_VIOLATION` | `false` | Also block on StoryKeeper drift (without RLP-0 rupture) |
| `PACT_WARMUP_CALLS` | `3` | Requests forwarded without gating while baseline builds |
| `PACT_BLOCK_INJECTION` | `false` | Block (not just warn) when injection pattern found in response |
| `PACT_AUDIT_LOG` | `./pact-ax-audit.jsonl` | Path to append-only JSONL audit log |

### Tuning the gate sensitivity

```bash
# More sensitive — gate fires earlier
export PACT_RUPTURE_THRESHOLD=0.4

# More permissive — only fires on severe collapse
export PACT_RUPTURE_THRESHOLD=0.7

# Full enforcement: both layers can block
export PACT_BLOCK_ON_VIOLATION=true
```

---

## File Structure

```
proxy/
├── demo.py                  # End-to-end demo driver
├── config.example.yaml      # Documented configuration reference
└── src/
    ├── __main__.py           # Entry point: python -m proxy.src
    ├── proxy.py              # PACTAXProxy — main proxy loop + HTTP mode
    ├── rlp_bridge.py         # StoryKeeper → RLP-0 primitive translation
    └── story_keeper.py       # Behavioral drift detection (AX expression layer)
```

---

## Connecting to Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pact-ax-github": {
      "command": "python",
      "args": ["-m", "proxy.src"],
      "cwd": "/path/to/pact-ax",
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxx",
        "PACT_UPSTREAM_MODE": "http",
        "PACT_RUPTURE_THRESHOLD": "0.6"
      }
    }
  }
}
```

Restart Cursor — the proxy appears as a GitHub MCP server with session integrity baked in. Every tool call Cursor makes passes through both layers transparently.

---

## Audit Log

Every proxy instance appends to a shared JSONL file (`pact-ax-audit.jsonl` by default). Each line is one JSON record. Multiple simultaneous sessions write to the same file, distinguished by `session_id`.

**Record types:** `session_start`, `warmup_request`, `request`, `injection_alert`, `gate_closed`, `gate_reopened`, `repair_invalid`, `session_end`

**Example records:**
```jsonl
{"ts":"2026-02-26T10:00:00Z","event":"session_start","session_id":"pact-a3f9c1","upstream_mode":"http","rupture_threshold":0.6}
{"ts":"2026-02-26T10:00:01Z","event":"session_start","session_id":"pact-a3f9c1","client":"pact-ax-demo:2.0.0","event_detail":"initialize_received"}
{"ts":"2026-02-26T10:00:02Z","event":"warmup_request","session_id":"pact-a3f9c1","method":"tools/call","warmup_count":1,"warmup_total":3}
{"ts":"2026-02-26T10:00:04Z","event":"request","session_id":"pact-a3f9c1","method":"tools/call","pattern":"repo:read:source","coherence":0.82,"trust":0.72,"trajectory":"BUILDING","rupture_risk":0.21,"gated":false}
{"ts":"2026-02-26T10:00:09Z","event":"gate_closed","session_id":"pact-a3f9c1","rupture_risk":0.69,"repair_token":"a3f9c12b4e17","primitives":{"trust":0.29,"intent":0.35,"narrative":0.21,"commitments":0.6}}
{"ts":"2026-02-26T10:00:15Z","event":"session_end","session_id":"pact-a3f9c1","client":"pact-ax-demo:2.0.0","messages_processed":10,"blocked_requests":4,"final_trust":0.29,"gate_closed":true}
```

**Query examples with jq:**
```bash
# All gate closure events across all sessions
jq 'select(.event=="gate_closed")' pact-ax-audit.jsonl

# Trust trajectory over time for a session
jq 'select(.session_id=="pact-a3f9c1" and .event=="request") | {ts,trust,rupture_risk}' pact-ax-audit.jsonl

# Sessions that ended with gate still closed
jq 'select(.event=="session_end" and .gate_closed==true) | {session_id,client,final_trust,final_rupture_risk}' pact-ax-audit.jsonl
```

## Related

- [RLP-0](https://github.com/neurobloomai/rlp-0) — Relational Ledger Protocol, standalone library
- [PACT-AX](https://github.com/neurobloomai/pact-ax) — Full agent collaboration framework
