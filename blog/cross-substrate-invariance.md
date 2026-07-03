---
layout: post
title: "Cross-Substrate Invariance: Intent Decays the Same Way Everywhere"
date: 2026-07-02
categories: [trust, architecture, agents]
excerpt: "The same intent-decay physics appear in agent chains, human outreach, and human–AI collaboration. Three substrates, one failure mode."
published: true
---

The most defensible claims are the ones that appear across unrelated domains without anyone
planning them that way. When the same failure mode surfaces in agent chains, human outreach,
and long-running human–AI collaborations — with the same shape, the same cause, the same
uninspected assumption — that is not a coincidence. That is a physics.

The physics is this: **any system that delegates through hops, without a machine-readable
contract, will decay intent**. The substrate does not matter. The decay rate changes. The
mechanism is always the same.

---

## How it appears in agent chains

An orchestrator issues a task to a sub-agent with a clear constraint: the result must satisfy
condition X. The constraint is expressed in the task prompt as prose — a sentence, maybe two.

The sub-agent re-summarizes the task for the next hop. The re-summarization preserves the
goal and drops the constraint — not maliciously, but because no mechanism existed to mark it
as load-bearing. Natural language is lossy. Each re-summary is a compression. Compressions
drop things.

By hop three, the constraint does not exist. The task completes. The constraint is violated.
The orchestrator has no way to see this, because the constraint was never in anything
propagatable. It lived in prose. Prose does not travel intact.

The PACT-AX fix is direct:

```python
from pact_ax.primitives.trust_context import TrustIntent, TrustConstraint

intent = TrustIntent(
    purpose="run analysis on dataset",
    constraints=[
        TrustConstraint(
            key="exclude_invalid_records",
            description="Records failing validation must not appear in output",
            load_bearing=True,   # must survive every hop verbatim
        ),
    ],
)
```

The constraint is now machine-readable, propagatable, and verifiable at every hop:

```python
# At any downstream agent — before acting
violations = child.verify_intent_integrity(parent_intent)
if violations:
    child.signal_break(
        "intent_integrity",
        f"load-bearing constraints dropped: {violations}",
    )
```

If the constraint is dropped at any hop, it is detectable. The downstream agent does not
need access to the original orchestrator — the contract travels with the context.

---

## How it appears in human outreach

A message is drafted with a specific framing: a clear ask, a specific angle, a constraint on
how the request should be understood. It is sent. Someone forwards it. The forward drops the
framing — not deliberately, but because forwarding is a compression. The recipient replies to
the re-summarized version. The reply misses the point.

The mechanism is identical. Natural language delegation chain. No load-bearing flag. No way
for any hop to distinguish "this is the core of the request" from "this is context I can
drop." The goal survives (there is a request). The constraints decay (the specific framing
is gone).

The fix does not exist yet in human outreach infrastructure. That is the point.

---

## How it appears in human–AI collaboration

A long session begins with a clear constraint: "we are operating within these boundaries,
do not suggest X." Forty turns later, the agent's context has been compressed and
reweighted. The constraint from turn one has been treated as background. The agent suggests X.

It did not forget. It re-summarized, and the re-summarization was lossy. The constraint was
not marked as load-bearing — it was expressed in natural language at turn one, and natural
language at turn one does not survive forty turns of context compression intact.

The same physics. The same mechanism. A different substrate.

---

## The invariant

Three substrates. One failure mode.

The common cause: **intent expressed as prose travels through a compression function at
every hop**. The compression preserves goal and degrades constraints. The goal is always
the most salient thing; constraints are context, and context is what compression
sacrifices first.

The invariant is not "systems lose intent." It is more specific: **load-bearing constraints
decay faster than goals, because goals are compression-stable and constraints are not.**

The fix is not "write clearer prose." That addresses the input, not the compression function.
The fix is to remove the compression function from the path of load-bearing constraints —
express them in a form that does not pass through re-summarization at all.

That is what `TrustConstraint(load_bearing=True)` does. A machine-readable contract does not
go through the compression function. It propagates intact or it raises a detectable violation.

**Prose is lossy. Contracts propagate.**

---

## Why this matters beyond agents

If intent decay were specific to AI agents, it would be an AI safety concern — interesting
but bounded. If it appears across human outreach, agent chains, and human–AI collaboration
with the same mechanism, it is something else: a **fundamental property of natural-language
delegation systems**.

AI agents did not introduce intent decay. They made it visible, faster, and at scale. They
also — for the first time — make the fix tractable: machine-readable contracts that travel
with the delegation, not inside it.

The infrastructure we are building for agent trust is also, precisely, the infrastructure
that was missing from every other natural-language delegation system. That is not a feature.
That is what it means for a solution to address the root cause.

---

*Part of the PACT-AX doctrine series — [neurobloom.ai](https://neurobloom.ai)*  
*Code: [github.com/neurobloomai/pact-ax](https://github.com/neurobloomai/pact-ax)*
