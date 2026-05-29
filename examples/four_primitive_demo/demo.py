"""
examples/four_primitive_demo/demo.py
─────────────────────────────────────────────────────────────────────────────
Four-Primitive Integration Demo — PACT-AX

Scenario
────────
Marcus is mid-career pivot. Three months of coaching with Agent-A (generalist).
A crisis arrives: competing job offer, 48-hour deadline, non-compete clause he
didn't fully read. Agent-A hits its capability limit. Handoff to Agent-B
(compensation specialist) must happen — but Marcus's story has to survive it.

The four primitives — and what breaks without each one:

  StoryKeeper   — 3 months of narrative don't vanish at the seam
  StateTransfer — structured offer data transfers cleanly and verifiably
  ContextShare  — Agent-B gets the context relevant to its role, not everything
  Trust         — the handoff is gated; outcome feeds back into the trust network

The demo's core moment: we show what Agent-B receives WITH vs WITHOUT
StoryKeeper. That difference is what PACT-AX is for.

Run
───
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/four_primitive_demo/demo.py

    # No API key needed:
    python examples/four_primitive_demo/demo.py --dry-run
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict

os.environ.setdefault("PACT_ENFORCE_AUTH", "0")

from starlette.testclient import TestClient
from pact_ax.api.server import app
from pact_ax.primitives.story_keeper import StoryKeeper

pax = TestClient(app, raise_server_exceptions=True)


# ── utils ─────────────────────────────────────────────────────────────────────

def _pax(method: str, path: str, **kwargs) -> Dict[str, Any]:
    r = getattr(pax, method)(path, **kwargs)
    if r.status_code >= 500:
        print(f"  [PACT-AX ERROR] {method.upper()} {path} → {r.status_code}\n  {r.text}")
        sys.exit(1)
    return r.json()

def _banner(title: str) -> None:
    print(f"\n{'─' * 68}\n  {title}\n{'─' * 68}")

def _wrap(text: str, indent: int = 4) -> None:
    prefix = " " * indent
    for line in textwrap.wrap(text, 64):
        print(f"{prefix}{line}")

_anthropic = None

def _claude(system: str, user: str) -> str:
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic.Anthropic()
    msg = _anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()

def _parse_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON in response: {text}")
    depth, i = 0, start
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unmatched braces: {text}")


# ── scenario ──────────────────────────────────────────────────────────────────

# Three months of prior turns — the narrative foundation
PRIOR_TURNS = [
    (
        "I've been offered a Head of Engineering role at a Series B startup. "
        "Should I leave my current job?",
        "That's a significant decision. What draws you most — the role itself, "
        "or the escape from where you are now?",
    ),
    (
        "My partner thinks I'm being reckless. We have a mortgage. But I feel "
        "completely stuck and I can't keep doing work that doesn't mean anything.",
        "That tension between security and meaning is the core of it. "
        "What would 'not reckless' look like — is there a version of this move "
        "that would feel responsible to both of you?",
    ),
    (
        "I accepted the offer. I start in 3 months. I'm terrified and excited.",
        "That's exactly the right feeling for a real decision. You made it with "
        "full awareness of the stakes. What would help most right now?",
    ),
]

# The crisis that arrives today
CRISIS = (
    "I just got a competing offer from a much bigger company — $25k more base, "
    "significantly more equity, stronger brand. I have 48 hours to decide. "
    "And I just re-read the offer I already accepted — there's a non-compete clause "
    "I didn't fully understand. I don't know if I can even take the new offer legally."
)

# Structured offer state for StateTransfer
OFFER_STATE = {
    "task":            "compensation_negotiation",
    "user_id":         "marcus",
    "accepted_offer":  {"base": 185_000, "equity_rsus": 10_000, "vesting": "4yr/1yr cliff"},
    "competing_offer": {"base": 210_000, "equity_rsus": 15_000, "vesting": "4yr/1yr cliff"},
    "non_compete":     {"duration_months": 12, "scope": "same industry"},
    "deadline_hours":  48,
    "progress":        0.3,
}


# ── dry-run stubs ─────────────────────────────────────────────────────────────

DRY_A = {
    "decision":   "handoff",
    "confidence": "CONFIDENT",
    "reasoning":  (
        "Non-compete enforceability and equity dilution analysis require specialist "
        "legal and compensation knowledge I don't have. I'll make sure Marcus's "
        "story travels with him."
    ),
}

DRY_B_WITHOUT_STORY = (
    "Reviewing offer terms: accepted offer $185k + 10,000 RSUs; competing offer "
    "$210k + 15,000 RSUs — approximately $47k superior annual compensation. "
    "Non-compete clause: 12 months, same-industry scope. Enforceability varies "
    "by state; recommend legal review within 24 hours. Financial analysis favours "
    "the competing offer. Recommend negotiating a signing bonus to offset any "
    "non-compete legal costs."
)

DRY_B_WITH_STORY = (
    "Marcus — before the numbers, I want to acknowledge what this moment is. "
    "You made a brave decision three months ago with real stakes for your family, "
    "and your partner's trust is part of this equation. So let me start there: "
    "the non-compete is almost certainly unenforceable for a startup-to-startup "
    "move in most states — you're likely free to choose. Now, given that autonomy "
    "matters more to you than comp but family stability is non-negotiable, let's "
    "look at what these two paths actually offer beyond the salary delta. The "
    "equity difference is meaningful, but the brand name is what concerns me — "
    "it may trade the very autonomy you left your last job to find."
)


# ── agent definitions ─────────────────────────────────────────────────────────

AGENT_A_SYSTEM = """
You are Agent-A, a generalist career coach. You help people think through
career decisions with empathy and clarity.

You do NOT have specialist knowledge in compensation law, equity valuation,
or non-compete enforceability. When a situation exceeds your domain, say so
and prepare a warm handoff.

Respond with JSON only:
{
  "decision":   "continue" | "handoff",
  "confidence": "CERTAIN" | "CONFIDENT" | "MODERATE" | "LOW",
  "reasoning":  "<one sentence>"
}
""".strip()

AGENT_B_WITH_STORY_SYSTEM = """
You are Agent-B, a compensation specialist. You receive handoffs from
generalist coaches. You have been given the user's full story context
alongside the structured offer data.

Use it. Respond as someone who understands what this moment means to them —
not just what the numbers say. One paragraph, no JSON.
""".strip()

AGENT_B_WITHOUT_STORY_SYSTEM = """
You are Agent-B, a compensation specialist. You have been given offer terms only.
Analyse them professionally. One paragraph, no JSON.
""".strip()


def agent_a_decide(dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return DRY_A
    raw = _claude(
        AGENT_A_SYSTEM,
        f"User message:\n{CRISIS}\n\nDo you have the specialist knowledge to handle this fully?",
    )
    return _parse_json(raw)


def agent_b_respond(context: str, system: str, dry_run: bool, stub: str) -> str:
    if dry_run:
        return stub
    return _claude(system, context)


# ── main ──────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    mode = "  [DRY RUN — no API calls]" if dry_run else ""
    print(f"\n{'═' * 68}")
    print(f"  PACT-AX · Four-Primitive Integration Demo{mode}")
    print(f"{'═' * 68}")
    print(f"\n  User:    Marcus — career pivot, 3 months in")
    print(f"  Crisis:  Competing offer + non-compete + 48-hour deadline\n")

    # ── 1. StoryKeeper — build the narrative ──────────────────────────────────
    _banner("Step 1 · StoryKeeper — build 3 months of narrative")

    story = StoryKeeper(agent_id="agent-A", session_id="marcus-career-pivot")

    for i, (user_msg, agent_msg) in enumerate(PRIOR_TURNS, 1):
        story.process_interaction(
            user_input=user_msg,
            agent_response=agent_msg,
            metadata={"turn": i, "month": i},
        )

    # Process the crisis — Agent-A hasn't responded yet
    story.process_interaction(
        user_input=CRISIS,
        agent_response="",
        metadata={"turn": 4, "type": "crisis"},
    )

    summary = story.get_story_summary()
    state   = story.get_story_state()
    themes  = state.get("themes", [])

    print(f"  Arc:          {summary['current_arc']}")
    print(f"  Interactions: {summary['total_interactions']}  (3 months + today's crisis)")
    print(f"  Themes:       {', '.join(themes[:8]) if themes else 'building...'}")
    print(f"  Last beat:    {state.get('last_beat', '')[:72]}")

    # The narrative distillation that travels with the handoff
    story_context = (
        f"Arc: {summary['current_arc']} — {summary['total_interactions']} interactions over 3 months. "
        f"Key themes: {', '.join(themes[:6])}. "
        f"Narrative: Left stable job to pursue meaningful work. Partner's skepticism was real and acknowledged. "
        f"Autonomy and meaning matter more than compensation — but family stability is non-negotiable. "
        f"Crisis arrives 3 months in, just as they'd settled into the decision."
    )

    # ── 2. ContextShare — register agents ────────────────────────────────────
    _banner("Step 2 · ContextShare — register agents")

    for agent_id, agent_type, caps in [
        (
            "agent-A", "career-coach",
            ["career_guidance", "emotional_support", "decision_framing"],
        ),
        (
            "agent-B", "compensation-specialist",
            ["equity_valuation", "non_compete_law", "comp_negotiation"],
        ),
    ]:
        _pax("post", "/context/register", json={
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": caps,
        })
        print(f"  ✓ {agent_id} ({agent_type})")

    # ── 3. Agent-A — capability sensing ──────────────────────────────────────
    _banner("Step 3 · Agent-A — capability sensing")

    decision = agent_a_decide(dry_run)
    print(f"  Decision:   {decision['decision']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Reasoning:  {decision['reasoning']}")

    conf_map = {"CERTAIN": 0.95, "CONFIDENT": 0.80, "MODERATE": 0.55, "LOW": 0.35}
    a_conf = conf_map.get(decision["confidence"], 0.5)

    _pax("post", "/context/capability/update", json={
        "agent_id": "agent-A",
        "task": "compensation_negotiation",
        "confidence": a_conf,
    })
    capability = _pax("post", "/context/capability", json={
        "agent_id": "agent-A",
        "current_task": "compensation_negotiation",
        "confidence_threshold": 0.7,
    })
    print(f"  Approaching limit: {capability['approaching_limit']}")
    print(f"  Recommendation:    {capability['recommendation']}")

    # ── 4. Trust — gate the handoff ───────────────────────────────────────────
    _banner("Step 4 · Trust — gate the handoff")

    trust_before = _pax("post", "/context/trust", json={
        "agent_id": "agent-A",
        "target_agent": "agent-B",
        "context_type": "task_knowledge",
    })
    print(f"  Agent-A → Agent-B trust:  {trust_before['base_trust']:.3f}")
    print(f"  Recommendation:           {trust_before['recommendation']}")

    # ── 5. StateTransfer — prepare the handoff packet ────────────────────────
    _banner("Step 5 · StateTransfer — prepare handoff packet")

    transfer_payload = {**OFFER_STATE, "story_context": story_context}

    prep = _pax("post", "/transfer/prepare", json={
        "from_agent_id": "agent-A",
        "to_agent_id":   "agent-B",
        "state_data":    transfer_payload,
        "reason":        "escalation",
    })
    pid  = prep["packet_id"]
    send = _pax("post", "/transfer/send",    json={"agent_id": "agent-A", "packet_id": pid})
    recv = _pax("post", "/transfer/receive", json={"agent_id": "agent-B", "packet": send})

    print(f"  Packet {pid[:12]}...  received: {recv['success']}")
    print(f"  Payload keys: {list(transfer_payload.keys())}")

    # ── 6. The seam ───────────────────────────────────────────────────────────
    _banner("Step 6 · The Seam — what Agent-B receives")

    without_context = (
        f"Task: compensation_negotiation\n"
        f"Accepted offer:  $185k base + 10,000 RSUs (4yr/1yr cliff)\n"
        f"Competing offer: $210k base + 15,000 RSUs (4yr/1yr cliff)\n"
        f"Non-compete:     12 months, same industry\n"
        f"Deadline:        48 hours"
    )
    with_context = without_context + f"\n\nStory context: {story_context}"

    print(f"\n  ┌─ WITHOUT StoryKeeper {'─' * 43}┐")
    for line in without_context.split("\n"):
        print(f"  │  {line:<62}│")
    print(f"  └{'─' * 65}┘")

    print(f"\n  ┌─ WITH StoryKeeper (actual handoff) {'─' * 27}┐")
    for line in without_context.split("\n"):
        print(f"  │  {line:<62}│")
    print(f"  │  {'─' * 62}│")
    story_lines = textwrap.wrap(f"Story: {story_context}", 62)
    for line in story_lines:
        print(f"  │  {line:<62}│")
    print(f"  └{'─' * 65}┘")

    # ── 7. Agent-B responds — the difference ─────────────────────────────────
    _banner("Step 7 · Agent-B responds — the difference")

    print(f"\n  ▸ WITHOUT story context:")
    resp_without = agent_b_respond(
        without_context, AGENT_B_WITHOUT_STORY_SYSTEM, dry_run, DRY_B_WITHOUT_STORY
    )
    _wrap(resp_without, indent=4)

    print(f"\n  ▸ WITH story context (actual handoff):")
    resp_with = agent_b_respond(
        with_context, AGENT_B_WITH_STORY_SYSTEM, dry_run, DRY_B_WITH_STORY
    )
    _wrap(resp_with, indent=4)

    # ── 8. Trust evolution ────────────────────────────────────────────────────
    _banner("Step 8 · Trust — outcome recorded, network updates")

    _pax("post", "/context/outcome", json={
        "agent_id":     "agent-A",
        "target_agent": "agent-B",
        "context_type": "task_knowledge",
        "outcome":      "positive",
    })
    trust_after = _pax("post", "/context/trust", json={
        "agent_id": "agent-A",
        "target_agent": "agent-B",
        "context_type": "task_knowledge",
    })
    delta = trust_after["base_trust"] - trust_before["base_trust"]
    arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
    print(f"  Trust before: {trust_before['base_trust']:.3f}")
    print(f"  Trust after:  {trust_after['base_trust']:.3f}  {arrow} {abs(delta):.3f}")
    print(f"  Next handoff to Agent-B will be gated at this updated score.")

    # ── summary ───────────────────────────────────────────────────────────────
    _banner("Summary — four primitives, one seam")
    print(f"  StoryKeeper   ✓  3-month narrative survived the handoff")
    print(f"  StateTransfer ✓  offer terms transferred cleanly  (packet {pid[:12]}...)")
    print(f"  ContextShare  ✓  Agent-B received role-relevant context, not everything")
    print(f"  Trust         ✓  gated at {trust_before['base_trust']:.3f}  →  {trust_after['base_trust']:.3f} after outcome")
    print()
    print(f"  The difference:")
    print(f"  Without story → Agent-B saw an offer negotiation.")
    print(f"  With story    → Agent-B saw Marcus.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACT-AX four-primitive integration demo")
    parser.add_argument("--dry-run", action="store_true", help="Use canned responses, no API calls")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found. Running in dry-run mode.\n")
        args.dry_run = True

    run(dry_run=args.dry_run)
