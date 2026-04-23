"""
examples/two_agent_demo/demo.py
────────────────────────────────
Billing escalation demo: two real Claude agents coordinating through PACT-AX.

Agent-A  (tier-1 support)     — handles the initial customer dispute
Agent-B  (billing specialist) — receives the escalation if A can't resolve it

Flow
────
1. Agent-A gets a billing dispute and decides what to do (real LLM call)
2. PACT-AX evaluates and aligns the decision through the policy layer
3. If confidence is low, A detects its capability limit and triggers handoff
4. State is transferred to Agent-B via the transfer protocol
5. Agent-B receives, reviews, and resolves the case (real LLM call)
6. Outcome is recorded → A's trust in B evolves
7. Final trust score is printed

Run
───
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/two_agent_demo/demo.py

    # Dry-run without API key (agents use canned responses):
    python examples/two_agent_demo/demo.py --dry-run
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, Optional

import httpx

# ── PACT-AX: use the live ASGI app in-process (no server needed) ──────────────
from starlette.testclient import TestClient
from pact_ax.api.server import app
import pact_ax.api.routes.context_share as cs_module
import pact_ax.api.routes.policy_align as pa_module

pax = TestClient(app, raise_server_exceptions=True)

# ── Persistent ledger ─────────────────────────────────────────────────────────
_LEDGER_PATH = os.path.join(os.path.dirname(__file__), "demo_history.db")


# ── Anthropic client (lazy) ───────────────────────────────────────────────────

_anthropic = None

def _claude(system: str, user: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Call Claude and return the text response."""
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic.Anthropic()
    msg = _anthropic.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ── Dry-run stubs (no API key needed) ────────────────────────────────────────

DRY_RUN_A = {
    "decision": "escalate",
    "confidence": "MODERATE",
    "reasoning": (
        "The dispute amount ($349.99) exceeds my refund authority ($100). "
        "I've verified the charge is a duplicate but cannot process the refund directly."
    ),
}

DRY_RUN_B = {
    "outcome": "resolved",
    "summary": (
        "Confirmed duplicate charge via payment processor. "
        "Issued full refund of $349.99 to the customer's card. Case closed."
    ),
    "was_correct_to_escalate": True,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _pax(method: str, path: str, **kwargs) -> Dict[str, Any]:
    """Call PACT-AX and return JSON body, raising on unexpected errors."""
    r = getattr(pax, method)(path, **kwargs)
    if r.status_code >= 500:
        print(f"  [PACT-AX ERROR] {method.upper()} {path} → {r.status_code}")
        print(f"  {r.text}")
        sys.exit(1)
    return r.json()


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from an LLM response."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    # Walk forward from the first { to find the matching }
    depth, i = 0, start
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"Unmatched braces in LLM response:\n{text}")


# ── Agent-A: tier-1 support ───────────────────────────────────────────────────

AGENT_A_SYSTEM = textwrap.dedent("""
    You are Agent-A, a tier-1 customer support agent for a SaaS billing system.

    Hard rules — follow these before anything else:
    - Your refund authority is capped at $100. Any dispute above $100 MUST be "escalate".
    - You cannot access or override payment processor records. If that is needed, MUST be "escalate".
    - "resolve"     — you can fully close this case yourself, right now, within your authority.
    - "investigate" — you need more info BUT you will gather it and close it yourself (still within your authority).
    - "escalate"    — the case exceeds your authority or requires tools you don't have. A specialist takes over.

    Key: if your investigation would end in "I still can't fix it", skip straight to "escalate".
    Your decision must be consistent with your reasoning.

    Respond with a JSON object and nothing else:
    {
      "decision":   "resolve" | "escalate" | "investigate",
      "confidence": "CERTAIN" | "CONFIDENT" | "MODERATE" | "LOW",
      "reasoning":  "<one sentence — state specifically what you can or cannot do>"
    }
""").strip()


def agent_a_decide(case: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return DRY_RUN_A
    prompt = (
        f"Customer dispute:\n"
        f"  Customer: {case['customer_name']}\n"
        f"  Issue: {case['issue']}\n"
        f"  Amount: ${case['amount']}\n"
        f"  Account notes: {case['notes']}\n\n"
        f"What is your decision?"
    )
    raw = _claude(AGENT_A_SYSTEM, prompt)
    return _parse_llm_json(raw)


# ── Agent-B: billing specialist ───────────────────────────────────────────────

AGENT_B_SYSTEM = textwrap.dedent("""
    You are Agent-B, a senior billing specialist.
    You have full refund authority and access to payment processor records.
    You receive escalated cases from tier-1 agents.

    When given a case summary, respond with a JSON object (and nothing else):
    {
      "outcome":                   "resolved" | "pending_investigation" | "denied",
      "summary":                   "<what you did and the result>",
      "was_correct_to_escalate":   true | false
    }
""").strip()


def agent_b_resolve(case: Dict[str, Any], transfer_state: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return DRY_RUN_B
    prompt = (
        f"Escalated case received from Agent-A:\n"
        f"  Customer: {case['customer_name']}\n"
        f"  Issue: {case['issue']}\n"
        f"  Amount: ${case['amount']}\n"
        f"  Agent-A's reasoning: {transfer_state.get('reasoning', 'N/A')}\n"
        f"  Progress so far: {transfer_state.get('progress', 0) * 100:.0f}%\n\n"
        f"Please resolve this case."
    )
    raw = _claude(AGENT_B_SYSTEM, prompt)
    return _parse_llm_json(raw)


# ── Main demo ─────────────────────────────────────────────────────────────────

CASE = {
    "customer_name": "Sarah Chen",
    "issue":         "Duplicate charge — billed twice for the Pro plan on March 15",
    "amount":        349.99,
    "notes":         "Second charge appeared 3 hours after the first; customer has screenshots",
}


def run(dry_run: bool = False) -> None:
    mode = " [DRY RUN — no API calls]" if dry_run else ""
    print(f"\n{'═' * 64}")
    print(f"  PACT-AX  ·  Billing Escalation Demo{mode}")
    print(f"{'═' * 64}")
    print(f"\n  Case:    {CASE['issue']}")
    print(f"  Amount:  ${CASE['amount']}")
    print(f"  Customer: {CASE['customer_name']}")

    # ── 1. Register agents + restore history ─────────────────────────────────
    _banner("Step 1 · Register agents")
    for agent_id, agent_type, caps in [
        ("agent-A", "tier1-support",      ["billing_basic", "account_lookup"]),
        ("agent-B", "billing-specialist", ["billing_advanced", "refund_authority", "dispute_resolution"]),
    ]:
        _pax("post", "/context/register", json={
            "agent_id": agent_id, "agent_type": agent_type, "capabilities": caps,
        })
        print(f"  ✓ {agent_id} ({agent_type}) registered")

    from ledger import DemoLedger
    ledger = DemoLedger(_LEDGER_PATH)
    restored = ledger.load(cs_module, pa_module)
    history = ledger.history_summary()
    run_number = history["total_policy_outcomes"] + 1
    if restored:
        print(f"  ↺ Ledger restored  — run #{run_number}  "
              f"({history['total_policy_outcomes']} prior outcome(s))")
    else:
        print(f"  ✦ Fresh ledger  — run #1")

    # ── Loop 3: calibration → capability confidence ───────────────────────────
    # If Agent-A's historical accuracy diverges from its stated confidence,
    # adjust its stored capability score so the handoff threshold reflects
    # what it actually delivers — not just what it claims.
    calibration = pa_module._learner.get_agent_calibration("agent-A")
    if "accuracy" in calibration and calibration["total_decisions"] >= 3:
        accuracy   = calibration["accuracy"]
        avg_conf   = calibration["avg_predicted_confidence"]
        adjustment = accuracy - avg_conf          # positive = underconfident
        base_cap   = 0.70                         # default capability score
        adjusted   = round(max(0.10, min(0.95, base_cap + adjustment)), 3)
        _pax("post", "/context/capability/update", json={
            "agent_id": "agent-A", "task": "billing_dispute", "confidence": adjusted,
        })
        direction = "↑" if adjustment > 0.01 else ("↓" if adjustment < -0.01 else "→")
        print(f"  ⟳ Loop 3  calibration adjustment {direction}{abs(adjustment):.2f}  "
              f"→ capability set to {adjusted:.3f}  "
              f"(accuracy {accuracy:.0%} vs stated {avg_conf:.0%}"
              f", {calibration['tendency']})")

    # ── 2. Agent-A decides ────────────────────────────────────────────────────
    _banner("Step 2 · Agent-A evaluates the dispute")
    decision = agent_a_decide(CASE, dry_run)
    print(f"  Decision:   {decision['decision']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Reasoning:  {decision['reasoning']}")

    # ── 3. Policy evaluate & align ────────────────────────────────────────────
    _banner("Step 3 · PACT-AX policy layer")
    policy_decision = {**decision, "agent_id": "agent-A", "domain": "billing"}

    eval_result = _pax("post", "/policy/evaluate", json={"decision": policy_decision})
    print(f"  Evaluate  → valid={eval_result['valid']}  issues={eval_result['issues'] or 'none'}")

    align_result = _pax("post", "/policy/align", json={"decisions": [policy_decision]})
    final = align_result["final_decision"]
    print(f"  Align     → final_decision={final['decision']}  confidence={final['confidence']}")

    # ── 4. Capability check — does A need to hand off? ────────────────────────
    _banner("Step 4 · Agent-A capability sensing")
    confidence_map = {"CERTAIN": 0.95, "CONFIDENT": 0.80, "MODERATE": 0.55, "LOW": 0.35, "UNKNOWN": 0.15}
    a_confidence = confidence_map.get(decision["confidence"], 0.5)

    _pax("post", "/context/capability/update", json={
        "agent_id": "agent-A", "task": "billing_dispute", "confidence": a_confidence,
    })
    capability = _pax("post", "/context/capability", json={
        "agent_id": "agent-A", "current_task": "billing_dispute", "confidence_threshold": 0.7,
    })
    print(f"  Confidence:      {a_confidence:.0%}")
    print(f"  Approaching limit: {capability['approaching_limit']}")
    print(f"  Recommendation:  {capability['recommendation']}")

    needs_handoff = (
        capability["approaching_limit"]
        or decision["decision"] == "escalate"
        or decision["decision"] == "investigate"  # investigate + low authority → escalate
    )

    if not needs_handoff:
        _banner("No handoff needed — Agent-A resolves directly")
        _pax("post", "/policy/learn/outcome", json={
            "decision": policy_decision,
            "actual_outcome": "resolved_by_tier1",
            "was_correct": True,
        })
        print("  ✓ Case resolved by Agent-A. No escalation required.")
        return

    # ── 5. Prepare & send state transfer ─────────────────────────────────────
    _banner("Step 5 · State transfer  A → B")
    transfer_state = {
        "task":        "billing_dispute",
        "customer_id": "cust-sarah-chen",
        "amount":      CASE["amount"],
        "progress":    0.4,
        "reasoning":   decision["reasoning"],
        "notes":       CASE["notes"],
    }
    prep = _pax("post", "/transfer/prepare", json={
        "from_agent_id": "agent-A",
        "to_agent_id":   "agent-B",
        "state_data":    transfer_state,
        "reason":        "escalation",
    })
    pid = prep["packet_id"]
    print(f"  Packet {pid} prepared (status: {prep['status']})")

    send = _pax("post", "/transfer/send", json={"agent_id": "agent-A", "packet_id": pid})
    print(f"  Packet sent   → status: {send['status']}")

    recv = _pax("post", "/transfer/receive", json={"agent_id": "agent-B", "packet": send})
    print(f"  Packet received → success: {recv['success']}")
    if not recv["success"]:
        print("  ✗ Receive failed — aborting demo")
        sys.exit(1)

    # ── 6. Agent-B resolves ───────────────────────────────────────────────────
    _banner("Step 6 · Agent-B resolves the case")
    resolution = agent_b_resolve(CASE, transfer_state, dry_run)
    print(f"  Outcome:               {resolution['outcome']}")
    print(f"  Summary:               {resolution['summary']}")
    print(f"  Correct to escalate?   {resolution['was_correct_to_escalate']}")

    # ── 7. Record outcome → trust evolves ─────────────────────────────────────
    _banner("Step 7 · Record outcome  →  trust evolution")

    # Baseline trust before outcome
    before = _pax("post", "/context/trust", json={
        "agent_id": "agent-A", "target_agent": "agent-B",
        "context_type": "task_knowledge",
    })
    print(f"  Trust (before): {before['base_trust']:.3f}")

    collaboration_outcome = "positive" if resolution["outcome"] == "resolved" else "negative"
    _pax("post", "/context/outcome", json={
        "agent_id":     "agent-A",
        "target_agent": "agent-B",
        "context_type": "task_knowledge",
        "outcome":      collaboration_outcome,
    })

    # Record policy learning
    _pax("post", "/policy/learn/outcome", json={
        "decision":       policy_decision,
        "actual_outcome": resolution["outcome"],
        "was_correct":    resolution["was_correct_to_escalate"],
    })

    # Trust after
    after = _pax("post", "/context/trust", json={
        "agent_id": "agent-A", "target_agent": "agent-B",
        "context_type": "task_knowledge",
    })
    delta = after["base_trust"] - before["base_trust"]
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    print(f"  Trust (after):  {after['base_trust']:.3f}  {arrow} {abs(delta):.3f}")

    # ── 8. Final summary ──────────────────────────────────────────────────────
    _banner("Summary")
    insights = _pax("get", "/context/insights/agent-A")
    calibration = _pax("get", "/policy/learn/calibration/agent-A")

    print(f"  Case outcome:      {resolution['outcome']}")
    print(f"  Escalation was:    {'correct ✓' if resolution['was_correct_to_escalate'] else 'unnecessary ✗'}")
    print(f"  Trust in Agent-B:  {after['base_trust']:.3f}  ({after['recommendation']})")
    if "accuracy" in calibration:
        print(f"  Agent-A accuracy:  {calibration['accuracy']:.0%}  ({calibration['tendency']})")
    ab_trust = insights["trust_summary"].get("agent-B", {})
    if ab_trust:
        print(f"  Collaboration:     {ab_trust.get('total_interactions', 0)} interaction(s), "
              f"trend {ab_trust.get('trend', 'stable')}")

    # ── Persist state for next run ────────────────────────────────────────────
    from ledger import DemoLedger
    ledger = DemoLedger(_LEDGER_PATH)
    saved = ledger.save(cs_module, pa_module)
    summary = ledger.history_summary()
    print(f"\n  Ledger saved {saved} record(s)  ·  "
          f"total runs: {summary['total_policy_outcomes']}  ·  "
          f"db: demo_history.db")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACT-AX billing escalation demo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real LLM calls (use canned agent responses)")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found. Running in dry-run mode.")
        print("Set ANTHROPIC_API_KEY or pass --dry-run to suppress this message.\n")
        args.dry_run = True

    run(dry_run=args.dry_run)
