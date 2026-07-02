# Cross-Substrate Invariance: Intent Decays the Same Way Everywhere

*DRAFT — needs Amarnath's edit pass before publishing. Personal examples marked [PERSONAL EDIT NEEDED].*

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

An orchestrator issues a task to a sub-agent with a clear constraint: "delisted tickers must
not appear in the scan results." This constraint lives in the orchestrator's context, expressed
in the task prompt as prose.

The sub-agent re-summarizes the task for the next hop. The re-summarization preserves the
goal ("scan for MA proximity setups") and drops the constraint — not maliciously, but because
no mechanism existed to mark it as load-bearing. The constraint was expressed in natural
language. Natural language is lossy. Each re-summary is a compression. Compressions drop
things.

By hop three, the constraint does not exist. The scan runs. MASI appears in the results.
The orchestrator's guarantee has been violated by a mechanism the orchestrator had no way to
see, because the constraint was never in anything propagatable.

This is not an edge case. It is the default behavior of any multi-hop delegation system that
passes intent as prose.

---

## How it appears in human outreach

[PERSONAL EDIT NEEDED: the specifics of where you've observed this in outreach / referral
chains / DM threads. The structure of the claim: you reach out with a specific intent and
framing, each person in the chain re-summarizes it to the next, and by the time it arrives
at the target, the original framing is gone. The target responds to the re-summarized version.
You get a reply that misses the point — not because anyone was careless, but because the
re-summarization chain had no mechanism to preserve what was load-bearing.]

The invariant: natural language delegation chains compress at every hop. There is no
load-bearing flag. There is no mechanism for the downstream party to distinguish "this is
the core of the request" from "this is context I can drop."

---

## How it appears in long-running human–AI collaboration

[PERSONAL EDIT NEEDED: your specific experience of what happens to intent across a long
session, or across multiple sessions. The claim: what you said in turn 1 — the actual
constraint you were operating under — is not what the agent is acting on in turn 47. The
agent's context has been compressed, summarized, reweighted. Some of what was load-bearing
at the start has been treated as background and dropped. You notice this when the agent
makes a suggestion that would have been obviously wrong given what you said at the start.
The agent did not forget — it re-summarized, and the re-summarization was lossy.]

The invariant holds here too. The substrate is different — one agent, one human, one
continuous session — but the mechanism is the same. Compression. No load-bearing flag.
No way to distinguish core constraints from context.

---

## The invariant

Three substrates. One failure mode.

The common cause: **intent expressed as prose travels through a compression function at
every hop**. The compression preserves goal (the what) and degrades constraints (the
why-not and the must-not). The goal is always the most salient thing; constraints are
context, and context is what compression sacrifices first.

The invariant is not "systems lose intent." It is more specific: **load-bearing constraints
decay faster than goals, because goals are compression-stable and constraints are not.**

This has a precise fix. Not "write clearer prose" — that addresses the input, not the
compression function. The fix is to remove the compression function from the path of
load-bearing constraints.

```python
TrustConstraint(
    key="delisted_tickers_invalid",
    description="Delisted tickers must not appear in scan results",
    load_bearing=True,   # survives every hop verbatim; omission is detectable
)
```

A machine-readable contract does not go through the compression function. It propagates
intact or it raises a detectable violation. The decay physics cannot reach it.

**Prose is lossy. Contracts propagate.**

---

## Why this matters beyond agents

The reason cross-substrate invariance is worth naming is that it changes the category of
the problem.

If intent decay were specific to AI agents, it would be an AI safety concern — interesting
but bounded. If it appears across human outreach, agent chains, and human–AI collaboration
with the same mechanism, it is something else: a **fundamental property of natural-language
delegation systems**.

AI agents did not introduce intent decay. They made it visible, faster, and at scale.
They also — for the first time — make the fix tractable: machine-readable contracts that
travel with the delegation, not inside it.

The infrastructure we are building for agent trust is also, precisely, the infrastructure
that was missing from every other natural-language delegation system. That is not a feature.
That is what it means for a solution to address the root cause.

---

*DRAFT — not for publication without Amarnath's edit pass on the [PERSONAL EDIT NEEDED] sections.*  
*Part of the PACT-AX doctrine series — [neurobloom.ai](https://neurobloom.ai)*  
*Code: [github.com/neurobloomai/pact-ax](https://github.com/neurobloomai/pact-ax)*
