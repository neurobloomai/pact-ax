# PACT-AX Doctrine Library

Core doctrine lines — the precise, defensible claims behind the design.
Each line is a constraint, not a slogan. It should survive a rebuttal.

---

## Architecture resolves authorization. It cannot resolve fidelity.

Star topologies — one orchestrator, disposable subagents, human root — make
trust implicit in the wiring. Subagents cannot talk to each other. Everything
routes through one synthesizer audited by one human. This answers *who may
act* but says nothing about *whether the acting agent still carries the
original intent intact*.

Authorization is a gate at T=0. Fidelity is the question at every hop after
that. Star topology answers the first question by construction. It is silent
on the second. PACT-AX is the answer to the second question.

---

## Prose is lossy. Contracts propagate.

Intent passed as natural language degrades per hop — each agent re-summarizes,
re-scopes, and silently drops constraints it did not recognize as load-bearing.
There is no mechanism to detect the loss. The delegator's intent lived in their
head, not in anything the downstream agent could inspect.

Intent passed as a structured contract — purpose, constraints, load-bearing
flags — travels intact. Omissions and modifications are detectable at any
downstream gate. `TrustIntent` + `verify_intent_integrity()` is the
implementation of this principle.

See: `pact_ax.primitives.trust_context.TrustIntent`

---

## HITL is not a design. It is a patch.

Human-in-the-loop is a manual trust substrate — the human personally
re-injecting lost intent at every checkpoint. It works while the network is
small enough for one human's working memory to *be* the substrate.

Every roadmap toward persistent, cross-org, multi-vendor agent networks is a
march toward the point where this stops scaling. At that point, the implicit
trust guarantees of star topology evaporate: agents from different owners,
sessions, and orgs, with no shared human root, no shared lineage, no mechanism
to verify that the agent still carries the intent it was issued.

HITL is not wrong — it is the correct patch for today's scale. The doctrine
is: do not mistake a patch for a design.

---

## Intent fidelity decays per hop; the human is the only error-correction mechanism — today.

The precise, defensible form of the intent-decay claim. Observable now, in
single-vendor star topologies, without production multi-agent deployment.

*Avoid*: "coordination is utterly failing" — invites the rebuttal "works fine
for me." The real claim is narrower and harder to dismiss: each hop introduces
re-summarization loss; the human is the only party who detects and corrects it;
that correction does not scale.

`TrustIntent.load_bearing` and `verify_intent_integrity()` are the first
primitives that give agents a mechanism to perform this correction themselves.
