# The Star Is Doing PACT's Job by Brute Force

*Why today's agent networks feel safe — and why that changes the moment you cross a trust boundary.*

---

Current orchestrator/subagent patterns feel strong. They are strong. Claude Code, LangGraph,
CrewAI, AutoGen — the real-world experience of building with these is that things generally
work, agents generally stay aligned, handoffs generally preserve intent.

The question worth asking: *why* does it work? Because the answer determines where it stops.

---

## The topology is the trust layer

Look at what every working production agent network has in common:

**Subagents cannot talk to each other.** All communication routes through the orchestrator.  
**Nothing persists beyond the task.** Each invocation starts fresh; there is no durable agent identity.  
**Everything routes through one synthesizer audited by one human.** The human is the root.

Trust is never encoded in these systems. It is implied by the shape of the wiring. The star
topology is doing what a trust layer would do — but by construction, not by design. The human
at the root is manually re-injecting lost intent at every checkpoint. That is what HITL
actually is: a human being the error-correction mechanism for intent decay the system has no
other way to handle.

This is not a criticism. It works. But it means the safety guarantee is load-bearing on one
constraint: **a single human's working memory can encompass the entire delegation graph**.

---

## MCP just changed the physics

Before MCP, trust was implicit in lineage. Same codebase. Same company. Same deployment.
Same model vendor. If an agent did something unexpected, you had one place to look and one
team to call. The implicit trust guarantee was: we control everything in the chain.

MCP standardized *connection*. Any model can now call any tool from any vendor. Any agent
can route to any other agent across org boundaries. The ecosystem is moving toward persistent
agents — agents with durable identity, cross-session memory, multi-org presence.

This is the right direction. It is also the moment the star topology's implicit guarantee
dissolves.

When a sub-agent comes from a different vendor, a different session, a different org — there
is no shared lineage. No shared human root. No single synthesizer the conversation routes
through. The human can no longer hold the full delegation graph in working memory, because
the delegation graph now crosses boundaries the human cannot audit.

**MCP standardized connection. Nothing standardizes whether the connected party should still
be listened to.**

---

## What the industry is building instead

The current answer to this is authentication: signed credentials, OAuth scopes, API keys.
Token delegation. A binary check at initialization.

That answers a different question.

Token says: this agent was authorized at T=0.  
The question is: is this agent still aligned at T=now?

Authorization is a snapshot. Trust is duration. The industry is building snapshots and calling
them trust infrastructure.

This is not a gap nobody has noticed — it is a gap nobody has *named*, because in star
topologies it never surfaces. The star suppresses the symptom. Every agent in the chain was
issued by the same orchestrator, audited by the same human, fresh-started for this task. Of
course it stays aligned. It has no opportunity not to.

The symptom appears at the star→mesh transition. Persistent agents. Cross-org delegation.
Multi-session context. Agents that outlive the task that created them. At that point, the
question "are you still trustworthy?" has no answer in any current framework.

---

## What PACT-AX adds

`TrustAlignmentCheck` is a continuous n/n gate — not a T=0 snapshot. You declare the
dimensions you care about (behavioral consistency, mode of engagement, capability coherence,
whatever the collaboration requires). The engine evaluates them. If one drifts, the gate
fails. The context marks itself as requiring a re-gate.

`TrustContext` is an intent-preservation contract. Not just "you're permitted to act" but
"here is the purpose and the load-bearing constraints on this delegation." It propagates
through every hop. Sub-agents can verify that the intent they received matches the intent
that was issued. Omissions are detectable. Modifications are detectable.

`TrustIntent` is the answer to the MASI-class failure: a constraint that lived in the
delegator's head — "delisted tickers invalidate the scan" — but had no propagatable form.
Prose is lossy. Contracts propagate.

None of this requires replacing star topologies. It is additive to them. What it does is
make the implicit guarantee of the star topology *explicit* — so it survives the moment the
star topology doesn't.

---

## The precise claim

The star is doing PACT's job by brute force. One human, working memory as the trust substrate,
fresh-start disposable subagents. This is a correct and effective strategy at current scale.

The industry roadmap is a march toward the point where it stops. Persistent agents.
Cross-org sessions. MCP-connected meshes. Every step in that direction is a step away from
the conditions under which the star topology's implicit trust guarantee holds.

PACT-AX is not a replacement for what works today. It is the substrate that makes what works
today survive tomorrow's topology.

---

*Part of the PACT-AX doctrine series — [neurobloom.ai](https://neurobloom.ai)*  
*Code: [github.com/neurobloomai/pact-ax](https://github.com/neurobloomai/pact-ax)*
