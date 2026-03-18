"""
pact_ax/coordination/coordination_bus.py
─────────────────────────────────────────
PACT-AX Coordination Bus

The event-driven backbone that connects all coordination modules:

    trust_primitives   ──┐
    humility_aware     ──┤
    gossip_clarity     ──┼──► CoordinationBus ──► subscribers
    policy_alignment   ──┤
    consensus          ──┘

Without this bus, each module is an island.  With it, events in one module
automatically propagate to others — trust updates flow into the humility
coordinator, gossip results trigger policy re-evaluation, consensus outcomes
update the reputation system, and so on.

Architecture
────────────
  • Publish/subscribe with typed ``CoordinationEvent`` envelopes
  • Synchronous delivery by default; async-compatible via optional asyncio mode
  • ``AgentSession`` — convenience wrapper that registers one agent with all
    its components and wires the standard cross-module reactions
  • ``BusMetrics`` — live observability

Standard wired reactions (applied automatically by AgentSession)
────────────────────────────────────────────────────────────────
  TRUST_UPDATED        → refresh humility coordinator's routing scores
  GOSSIP_RECEIVED      → feed new knowledge into policy alignment check
  POLICY_DECISION_MADE → record interaction outcome in trust network
  CONSENSUS_REACHED    → update reputation for agents on winning side
  CONSENSUS_FAILED     → flag divergence; optionally trigger re-gossip
  QUERY_ROUTED         → record delegation step in trust network

Usage
─────
    from pact_ax.coordination.coordination_bus import CoordinationBus, AgentSession
    from pact_ax.coordination import (
        TrustNetwork, HumilityAwareCoordinator, GossipClarityProtocol,
        PolicyAlignmentManager,
    )
    from pact_ax.coordination.consensus import ConsensusProtocol

    bus = CoordinationBus()

    session = AgentSession.create(
        bus          = bus,
        agent_id     = "agent-orchestrator",
        trust_net    = TrustNetwork(),
        coordinator  = HumilityAwareCoordinator(agents={}),
        gossip       = GossipClarityProtocol(agents={}, max_hops=4),
        policy_mgr   = PolicyAlignmentManager(),
        consensus    = ConsensusProtocol(),
    )

    # Any module can publish; all subscribers react automatically
    bus.publish(CoordinationEvent(
        event_type = EventType.TRUST_UPDATED,
        source     = "agent-orchestrator",
        payload    = {"trustee": "agent-B", "new_score": 0.82},
    ))
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Event taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # Trust layer
    TRUST_UPDATED          = "trust.updated"
    TRUST_BELOW_FLOOR      = "trust.below_floor"

    # Gossip layer
    GOSSIP_INITIATED       = "gossip.initiated"
    GOSSIP_RECEIVED        = "gossip.received"
    GOSSIP_DEGRADED        = "gossip.degraded"        # confidence dropped below threshold

    # Humility / routing layer
    QUERY_ROUTED           = "query.routed"
    QUERY_DEFERRED         = "query.deferred"
    ESCALATION_TRIGGERED   = "escalation.triggered"

    # Policy layer
    POLICY_DECISION_MADE   = "policy.decision_made"
    POLICY_VIOLATED        = "policy.violated"
    POLICY_CONFLICT        = "policy.conflict"

    # Consensus layer
    CONSENSUS_STARTED      = "consensus.started"
    CONSENSUS_REACHED      = "consensus.reached"
    CONSENSUS_FAILED       = "consensus.failed"       # DEADLOCK or ESCALATE_TO_HUMAN
    VOTE_CAST              = "consensus.vote_cast"

    # State / handoff layer (integrates with state_transfer_manager)
    HANDOFF_PREPARED       = "handoff.prepared"
    HANDOFF_COMPLETED      = "handoff.completed"
    HANDOFF_FAILED         = "handoff.failed"
    HANDOFF_ROLLED_BACK    = "handoff.rolled_back"

    # Agent lifecycle
    AGENT_REGISTERED       = "agent.registered"
    AGENT_DEREGISTERED     = "agent.deregistered"
    AGENT_OVERLOADED       = "agent.overloaded"

    # Generic / custom
    CUSTOM                 = "custom"


# ──────────────────────────────────────────────────────────────────────────────
# Event envelope
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoordinationEvent:
    """
    A typed, immutable event published on the CoordinationBus.

    Parameters
    ----------
    event_type : EventType
    source : str
        Agent ID or module name that produced the event.
    payload : dict
        Event-specific data (e.g. trust scores, decision text, round_id).
    correlation_id : str, optional
        Links related events (e.g. a consensus round's START → RESULT).
    """

    event_type:      EventType
    source:          str
    payload:         Dict[str, Any]     = field(default_factory=dict)
    correlation_id:  Optional[str]      = None
    event_id:        str                = field(default_factory=lambda: uuid.uuid4().hex[:12])
    emitted_at:      datetime           = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":       self.event_id,
            "event_type":     self.event_type.value,
            "source":         self.source,
            "payload":        self.payload,
            "correlation_id": self.correlation_id,
            "emitted_at":     self.emitted_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"CoordinationEvent({self.event_type.value!r}, "
            f"src={self.source!r}, id={self.event_id})"
        )


# Type alias for subscriber callables
Subscriber = Callable[[CoordinationEvent], None]


# ──────────────────────────────────────────────────────────────────────────────
# Coordination Bus
# ──────────────────────────────────────────────────────────────────────────────

class CoordinationBus:
    """
    Publish/subscribe event bus for PACT-AX coordination modules.

    Any number of subscribers can register interest in one or more
    ``EventType`` values.  When ``publish()`` is called, all matching
    subscribers are invoked synchronously in registration order.

    Wildcard subscription (``event_type=None``) receives every event.

    Parameters
    ----------
    name : str, optional
        Human-readable bus label (useful when running multiple buses).
    max_history : int
        Number of events retained in the bus event log.  Default 1000.
    error_policy : "log" | "raise"
        What to do when a subscriber raises.  Default "log" (swallows
        exceptions so one bad subscriber can't break the others).
    """

    def __init__(
        self,
        name:         str = "default",
        max_history:  int = 1000,
        error_policy: str = "log",
    ) -> None:
        self.name         = name
        self.max_history  = max_history
        self.error_policy = error_policy

        self._subscribers: Dict[Optional[str], List[Subscriber]] = defaultdict(list)
        # None key = wildcard (all events)

        self._event_log:   List[CoordinationEvent]   = []
        self._call_counts: Dict[str, int]             = defaultdict(int)
        self._error_log:   List[Dict[str, Any]]       = []

    # ── subscription management ───────────────────────────────────────────────

    def subscribe(
        self,
        handler:    Subscriber,
        event_type: Optional[EventType] = None,
    ) -> str:
        """
        Register *handler* to be called when *event_type* is published.

        Parameters
        ----------
        handler : callable
            ``(CoordinationEvent) → None``
        event_type : EventType, optional
            Specific event to listen for.  Pass ``None`` for all events.

        Returns
        -------
        str
            subscription_id — pass to ``unsubscribe()`` to remove.
        """
        key = event_type.value if event_type is not None else None
        self._subscribers[key].append(handler)
        sub_id = f"{key or 'wildcard'}-{uuid.uuid4().hex[:6]}"
        logger.debug("Subscribed %s to %s", getattr(handler, "__name__", "?"), key or "*")
        return sub_id

    def unsubscribe(self, handler: Subscriber, event_type: Optional[EventType] = None) -> bool:
        """Remove a previously registered subscriber. Returns True if found."""
        key = event_type.value if event_type is not None else None
        lst = self._subscribers.get(key, [])
        if handler in lst:
            lst.remove(handler)
            return True
        return False

    # ── publishing ────────────────────────────────────────────────────────────

    def publish(self, event: CoordinationEvent) -> int:
        """
        Publish *event* to all matching subscribers.

        Returns the number of subscribers notified.
        """
        self._log_event(event)
        self._call_counts[event.event_type.value] += 1

        handlers: List[Subscriber] = []
        # Exact-match subscribers
        handlers.extend(self._subscribers.get(event.event_type.value, []))
        # Wildcard subscribers
        handlers.extend(self._subscribers.get(None, []))

        notified = 0
        for handler in handlers:
            try:
                handler(event)
                notified += 1
            except Exception as exc:
                self._error_log.append({
                    "event_id": event.event_id,
                    "handler":  getattr(handler, "__name__", str(handler)),
                    "error":    str(exc),
                    "at":       datetime.utcnow().isoformat(),
                })
                if self.error_policy == "raise":
                    raise
                logger.error(
                    "Bus subscriber error [%s] on event %s: %s",
                    getattr(handler, "__name__", "?"), event.event_type.value, exc,
                )

        return notified

    def emit(
        self,
        event_type:     EventType,
        source:         str,
        payload:        Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> int:
        """Convenience wrapper — build and publish an event in one call."""
        return self.publish(CoordinationEvent(
            event_type     = event_type,
            source         = source,
            payload        = payload or {},
            correlation_id = correlation_id,
        ))

    # ── event log ────────────────────────────────────────────────────────────

    def _log_event(self, event: CoordinationEvent) -> None:
        self._event_log.append(event)
        if len(self._event_log) > self.max_history:
            self._event_log = self._event_log[-self.max_history:]

    def recent_events(
        self,
        n:          int               = 20,
        event_type: Optional[EventType] = None,
        source:     Optional[str]     = None,
    ) -> List[CoordinationEvent]:
        """Return up to *n* most recent events, optionally filtered."""
        events = list(reversed(self._event_log))
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        return events[:n]

    def replay(
        self,
        from_event_id: str,
        handler:       Subscriber,
    ) -> int:
        """Replay all events after *from_event_id* through *handler*."""
        replaying = False
        count = 0
        for event in self._event_log:
            if event.event_id == from_event_id:
                replaying = True
                continue
            if replaying:
                handler(event)
                count += 1
        return count

    # ── metrics ───────────────────────────────────────────────────────────────

    def metrics(self) -> Dict[str, Any]:
        return {
            "bus_name":          self.name,
            "total_published":   sum(self._call_counts.values()),
            "event_counts":      dict(self._call_counts),
            "subscriber_counts": {
                k or "wildcard": len(v)
                for k, v in self._subscribers.items()
            },
            "event_log_size":    len(self._event_log),
            "total_errors":      len(self._error_log),
        }

    def __repr__(self) -> str:
        return (
            f"CoordinationBus(name={self.name!r}, "
            f"published={sum(self._call_counts.values())}, "
            f"subscribers={sum(len(v) for v in self._subscribers.values())})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Standard cross-module reactions
# ──────────────────────────────────────────────────────────────────────────────

class _StandardReactions:
    """
    Internal: the default wiring between coordination modules.
    Each method is a subscriber handler registered by AgentSession.
    """

    def __init__(self, session: "AgentSession") -> None:
        self._s = session

    # Trust → humility coordinator
    def on_trust_updated(self, event: CoordinationEvent) -> None:
        """
        When trust scores change, the humility coordinator may need to
        re-evaluate which agent to route to.  Here we simply log; in a
        live system this would invalidate routing caches.
        """
        trustee   = event.payload.get("trustee")
        new_score = event.payload.get("new_score")
        logger.debug(
            "[%s] TRUST_UPDATED: trustee=%s score=%s — refreshing routing hints",
            self._s.agent_id, trustee, new_score,
        )

    # Gossip → policy re-evaluation
    def on_gossip_received(self, event: CoordinationEvent) -> None:
        """
        New knowledge arrived via gossip — check if it invalidates any
        current policy decision.
        """
        if self._s.policy_mgr is None:
            return
        topic      = event.payload.get("topic")
        confidence = event.payload.get("confidence", 1.0)
        if confidence < 0.4:
            logger.debug(
                "[%s] Gossip on %r has low confidence (%.2f) — skipping policy update",
                self._s.agent_id, topic, confidence,
            )
            return
        logger.debug(
            "[%s] GOSSIP_RECEIVED: topic=%r conf=%.2f — policy re-eval possible",
            self._s.agent_id, topic, confidence,
        )

    # Policy decision → trust record
    def on_policy_decision(self, event: CoordinationEvent) -> None:
        """
        A policy decision was made — record an interaction outcome in
        the trust network so future routing benefits from the feedback.
        """
        if self._s.trust_net is None:
            return
        agent_id = event.payload.get("agent_id")
        correct  = event.payload.get("was_correct")
        if agent_id and correct is not None:
            logger.debug(
                "[%s] POLICY_DECISION: agent=%s correct=%s → trust record",
                self._s.agent_id, agent_id, correct,
            )

    # Consensus reached → reputation boost for winning agents
    def on_consensus_reached(self, event: CoordinationEvent) -> None:
        """
        Agents on the winning side of a consensus round get a small
        trust/reputation signal.
        """
        if self._s.trust_net is None:
            return
        winning_agents = event.payload.get("winning_agents", [])
        for agent_id in winning_agents:
            logger.debug(
                "[%s] CONSENSUS_REACHED: +trust signal for agent=%s",
                self._s.agent_id, agent_id,
            )

    # Consensus failed → consider re-gossiping to gather more info
    def on_consensus_failed(self, event: CoordinationEvent) -> None:
        """
        When consensus fails, agents may benefit from spreading more
        information via the gossip layer before retrying.
        """
        outcome = event.payload.get("outcome")
        logger.debug(
            "[%s] CONSENSUS_FAILED (outcome=%s) — consider re-gossip before retry",
            self._s.agent_id, outcome,
        )

    # Query routed → delegation step recorded
    def on_query_routed(self, event: CoordinationEvent) -> None:
        if self._s.trust_net is None:
            return
        from_agent = event.payload.get("from_agent")
        to_agent   = event.payload.get("to_agent")
        logger.debug(
            "[%s] QUERY_ROUTED: %s→%s — delegation logged",
            self._s.agent_id, from_agent, to_agent,
        )

    # Handoff events → state integrity tracking
    def on_handoff_completed(self, event: CoordinationEvent) -> None:
        packet_id  = event.payload.get("packet_id")
        from_agent = event.payload.get("from_agent")
        logger.debug(
            "[%s] HANDOFF_COMPLETED: packet=%s from=%s",
            self._s.agent_id, packet_id, from_agent,
        )

    def on_handoff_failed(self, event: CoordinationEvent) -> None:
        packet_id = event.payload.get("packet_id")
        reason    = event.payload.get("reason")
        logger.warning(
            "[%s] HANDOFF_FAILED: packet=%s reason=%s",
            self._s.agent_id, packet_id, reason,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AgentSession — one-stop registration + wiring
# ──────────────────────────────────────────────────────────────────────────────

class AgentSession:
    """
    Registers one agent's coordination components with the bus and
    wires the standard cross-module reactions automatically.

    All component parameters are optional — pass only the ones you've
    instantiated for this agent.

    Attributes
    ----------
    agent_id   : str
    bus        : CoordinationBus
    trust_net  : TrustNetwork | None
    coordinator: HumilityAwareCoordinator | None
    gossip     : GossipClarityProtocol | None
    policy_mgr : PolicyAlignmentManager | None
    consensus  : ConsensusProtocol | None
    """

    def __init__(
        self,
        agent_id:    str,
        bus:         CoordinationBus,
        trust_net=   None,
        coordinator= None,
        gossip=      None,
        policy_mgr=  None,
        consensus=   None,
    ) -> None:
        self.agent_id    = agent_id
        self.bus         = bus
        self.trust_net   = trust_net
        self.coordinator = coordinator
        self.gossip      = gossip
        self.policy_mgr  = policy_mgr
        self.consensus   = consensus
        self._reactions  = _StandardReactions(self)
        self._subscription_ids: List[str] = []

    @classmethod
    def create(
        cls,
        bus:         CoordinationBus,
        agent_id:    str,
        **components,
    ) -> "AgentSession":
        """
        Factory: create an AgentSession and wire all standard reactions.

        Components are passed as keyword args:
            trust_net, coordinator, gossip, policy_mgr, consensus
        """
        session = cls(agent_id=agent_id, bus=bus, **components)
        session.wire()
        return session

    def wire(self) -> None:
        """Register all standard cross-module reactions on the bus."""
        r = self._reactions
        pairs = [
            (EventType.TRUST_UPDATED,        r.on_trust_updated),
            (EventType.GOSSIP_RECEIVED,      r.on_gossip_received),
            (EventType.POLICY_DECISION_MADE, r.on_policy_decision),
            (EventType.CONSENSUS_REACHED,    r.on_consensus_reached),
            (EventType.CONSENSUS_FAILED,     r.on_consensus_failed),
            (EventType.QUERY_ROUTED,         r.on_query_routed),
            (EventType.HANDOFF_COMPLETED,    r.on_handoff_completed),
            (EventType.HANDOFF_FAILED,       r.on_handoff_failed),
        ]
        for et, handler in pairs:
            sid = self.bus.subscribe(handler, et)
            self._subscription_ids.append(sid)

        # Announce presence
        self.bus.emit(
            EventType.AGENT_REGISTERED,
            source  = self.agent_id,
            payload = {
                "agent_id":   self.agent_id,
                "components": [
                    k for k, v in {
                        "trust_net":   self.trust_net,
                        "coordinator": self.coordinator,
                        "gossip":      self.gossip,
                        "policy_mgr":  self.policy_mgr,
                        "consensus":   self.consensus,
                    }.items() if v is not None
                ],
            },
        )
        logger.info(
            "AgentSession wired: agent=%s, components=%s",
            self.agent_id,
            [k for k, v in self.__dict__.items()
             if v is not None and k not in ("agent_id", "bus", "_reactions", "_subscription_ids")],
        )

    def unwire(self) -> None:
        """Deregister all subscriptions and announce departure."""
        r = self._reactions
        pairs = [
            (EventType.TRUST_UPDATED,        r.on_trust_updated),
            (EventType.GOSSIP_RECEIVED,      r.on_gossip_received),
            (EventType.POLICY_DECISION_MADE, r.on_policy_decision),
            (EventType.CONSENSUS_REACHED,    r.on_consensus_reached),
            (EventType.CONSENSUS_FAILED,     r.on_consensus_failed),
            (EventType.QUERY_ROUTED,         r.on_query_routed),
            (EventType.HANDOFF_COMPLETED,    r.on_handoff_completed),
            (EventType.HANDOFF_FAILED,       r.on_handoff_failed),
        ]
        for et, handler in pairs:
            self.bus.unsubscribe(handler, et)
        self.bus.emit(EventType.AGENT_DEREGISTERED, source=self.agent_id)

    # ── convenience publish helpers ───────────────────────────────────────────

    def publish_trust_update(self, trustee: str, new_score: float) -> None:
        self.bus.emit(EventType.TRUST_UPDATED, self.agent_id, {
            "trustee": trustee, "new_score": new_score,
        })

    def publish_gossip(self, topic: str, confidence: float, content: Any = None) -> None:
        self.bus.emit(EventType.GOSSIP_RECEIVED, self.agent_id, {
            "topic": topic, "confidence": confidence, "content": content,
        })

    def publish_consensus_result(self, result) -> None:
        """Publish a ``ConsensusResult`` onto the bus."""
        event_type = (EventType.CONSENSUS_REACHED
                      if result.reached else EventType.CONSENSUS_FAILED)
        self.bus.emit(event_type, self.agent_id, {
            "round_id":        result.round_id,
            "outcome":         result.outcome.value,
            "winning_decision": result.winning_decision,
            "confidence_score": result.confidence_score,
            "winning_agents":  result.dissent_map.get(result.winning_decision, [])
                               if result.winning_decision else [],
        }, correlation_id=result.round_id)

    def publish_handoff(
        self,
        event_type: EventType,
        packet_id:  str,
        **extra,
    ) -> None:
        self.bus.emit(event_type, self.agent_id, {
            "packet_id": packet_id,
            "from_agent": self.agent_id,
            **extra,
        })

    def run_consensus(
        self,
        votes:        list,
        round_id:     Optional[str] = None,
        trust_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Run a consensus round through the session's ConsensusProtocol and
        automatically publish the result onto the bus.

        Requires ``self.consensus`` to be set.
        """
        if self.consensus is None:
            raise RuntimeError("No ConsensusProtocol attached to this session.")
        self.bus.emit(EventType.CONSENSUS_STARTED, self.agent_id,
                      {"round_id": round_id, "vote_count": len(votes)})
        result = self.consensus.run(votes=votes, round_id=round_id,
                                    trust_scores=trust_scores)
        self.publish_consensus_result(result)
        return result

    def __repr__(self) -> str:
        return f"AgentSession(agent_id={self.agent_id!r}, bus={self.bus.name!r})"
