"""
tests/test_coordination_bus.py
───────────────────────────────
Test suite for CoordinationBus and AgentSession.

Run with: pytest tests/test_coordination_bus.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from pact_ax.coordination.coordination_bus import (
    AgentSession,
    CoordinationBus,
    CoordinationEvent,
    EventType,
)
from pact_ax.coordination.consensus import (
    ConsensusProtocol,
    ConsensusStrategy,
    Vote,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bus():
    return CoordinationBus(name="test-bus")

@pytest.fixture
def event():
    return CoordinationEvent(
        event_type=EventType.TRUST_UPDATED,
        source="agent-A",
        payload={"trustee": "agent-B", "new_score": 0.85},
    )


# ──────────────────────────────────────────────────────────────────────────────
# CoordinationEvent
# ──────────────────────────────────────────────────────────────────────────────

class TestCoordinationEvent:
    def test_event_has_id(self, event):
        assert event.event_id
        assert len(event.event_id) == 12

    def test_event_to_dict(self, event):
        d = event.to_dict()
        assert d["event_type"] == "trust.updated"
        assert d["source"] == "agent-A"
        assert "payload" in d
        assert "emitted_at" in d

    def test_two_events_have_different_ids(self):
        e1 = CoordinationEvent(EventType.TRUST_UPDATED, "a", {})
        e2 = CoordinationEvent(EventType.TRUST_UPDATED, "a", {})
        assert e1.event_id != e2.event_id


# ──────────────────────────────────────────────────────────────────────────────
# CoordinationBus — subscribe / publish
# ──────────────────────────────────────────────────────────────────────────────

class TestCoordinationBus:

    def test_subscribe_and_publish(self, bus):
        received = []
        bus.subscribe(received.append, EventType.TRUST_UPDATED)
        bus.emit(EventType.TRUST_UPDATED, "agent-A", {"score": 0.9})
        assert len(received) == 1
        assert received[0].event_type == EventType.TRUST_UPDATED

    def test_wildcard_subscriber_receives_all(self, bus):
        received = []
        bus.subscribe(received.append)  # no event_type = wildcard
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        bus.emit(EventType.GOSSIP_RECEIVED, "b", {})
        bus.emit(EventType.CONSENSUS_REACHED, "c", {})
        assert len(received) == 3

    def test_subscriber_only_gets_its_type(self, bus):
        trust_events = []
        gossip_events = []
        bus.subscribe(trust_events.append,  EventType.TRUST_UPDATED)
        bus.subscribe(gossip_events.append, EventType.GOSSIP_RECEIVED)
        bus.emit(EventType.TRUST_UPDATED,   "a", {})
        bus.emit(EventType.GOSSIP_RECEIVED, "b", {})
        assert len(trust_events)  == 1
        assert len(gossip_events) == 1

    def test_publish_returns_notified_count(self, bus):
        bus.subscribe(lambda e: None, EventType.TRUST_UPDATED)
        bus.subscribe(lambda e: None, EventType.TRUST_UPDATED)
        n = bus.emit(EventType.TRUST_UPDATED, "a", {})
        assert n == 2

    def test_unsubscribe(self, bus):
        received = []
        handler = received.append
        bus.subscribe(handler, EventType.TRUST_UPDATED)
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        bus.unsubscribe(handler, EventType.TRUST_UPDATED)
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        assert len(received) == 1  # only first delivery

    def test_error_policy_log_swallows(self, bus):
        def bad_handler(e):
            raise RuntimeError("boom")
        bus.subscribe(bad_handler, EventType.TRUST_UPDATED)
        # Should NOT raise
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        assert len(bus._error_log) == 1

    def test_error_policy_raise(self):
        strict_bus = CoordinationBus(error_policy="raise")
        strict_bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")),
                             EventType.TRUST_UPDATED)
        with pytest.raises(RuntimeError):
            strict_bus.emit(EventType.TRUST_UPDATED, "a", {})

    def test_event_logged(self, bus, event):
        bus.publish(event)
        recent = bus.recent_events(n=5)
        assert any(e.event_id == event.event_id for e in recent)

    def test_recent_events_filter_by_type(self, bus):
        bus.emit(EventType.TRUST_UPDATED,   "a", {})
        bus.emit(EventType.GOSSIP_RECEIVED, "b", {})
        only_trust = bus.recent_events(event_type=EventType.TRUST_UPDATED)
        assert all(e.event_type == EventType.TRUST_UPDATED for e in only_trust)

    def test_recent_events_filter_by_source(self, bus):
        bus.emit(EventType.TRUST_UPDATED, "agent-A", {})
        bus.emit(EventType.TRUST_UPDATED, "agent-B", {})
        from_a = bus.recent_events(source="agent-A")
        assert all(e.source == "agent-A" for e in from_a)

    def test_max_history_trim(self):
        small_bus = CoordinationBus(max_history=3)
        for i in range(10):
            small_bus.emit(EventType.TRUST_UPDATED, "a", {})
        assert len(small_bus._event_log) == 3

    def test_metrics_structure(self, bus):
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        m = bus.metrics()
        for key in ("bus_name", "total_published", "event_counts",
                    "subscriber_counts", "event_log_size", "total_errors"):
            assert key in m

    def test_replay_delivers_events(self, bus):
        # publish two events; replay from first into a collector
        bus.emit(EventType.TRUST_UPDATED, "a", {})
        first_id = bus._event_log[-1].event_id
        bus.emit(EventType.GOSSIP_RECEIVED, "b", {})
        bus.emit(EventType.POLICY_DECISION_MADE, "c", {})

        replayed = []
        bus.replay(first_id, replayed.append)
        assert len(replayed) == 2   # events AFTER first_id


# ──────────────────────────────────────────────────────────────────────────────
# AgentSession
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentSession:

    def test_create_publishes_agent_registered(self, bus):
        registered = []
        bus.subscribe(registered.append, EventType.AGENT_REGISTERED)
        AgentSession.create(bus=bus, agent_id="agent-X")
        assert len(registered) == 1
        assert registered[0].payload["agent_id"] == "agent-X"

    def test_standard_reactions_wired(self, bus):
        """Standard reactions should be subscribed after wire()."""
        session = AgentSession.create(bus=bus, agent_id="agent-Y")
        # Trust update should silently succeed (reaction is wired)
        session.publish_trust_update("agent-Z", 0.88)
        assert bus.metrics()["total_published"] >= 1

    def test_publish_trust_update(self, bus):
        events = []
        bus.subscribe(events.append, EventType.TRUST_UPDATED)
        session = AgentSession.create(bus=bus, agent_id="agent-A")
        session.publish_trust_update("agent-B", 0.75)
        trust_events = [e for e in events if e.source == "agent-A"]
        assert len(trust_events) == 1
        assert trust_events[0].payload["new_score"] == 0.75

    def test_publish_gossip(self, bus):
        events = []
        bus.subscribe(events.append, EventType.GOSSIP_RECEIVED)
        session = AgentSession.create(bus=bus, agent_id="agent-A")
        session.publish_gossip(topic="revenue", confidence=0.82, content={"q": 3})
        gossip = [e for e in events if e.source == "agent-A"]
        assert gossip[0].payload["topic"] == "revenue"

    def test_run_consensus_publishes_reached(self, bus):
        reached = []
        bus.subscribe(reached.append, EventType.CONSENSUS_REACHED)
        session = AgentSession.create(
            bus=bus,
            agent_id="agent-A",
            consensus=ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE),
        )
        votes = [
            Vote("agent-A", "go", 0.9),
            Vote("agent-B", "go", 0.8),
            Vote("agent-C", "hold", 0.4),
        ]
        result = session.run_consensus(votes)
        assert result.reached
        assert len(reached) >= 1

    def test_run_consensus_publishes_failed(self, bus):
        failed = []
        bus.subscribe(failed.append, EventType.CONSENSUS_FAILED)
        session = AgentSession.create(
            bus=bus,
            agent_id="agent-A",
            consensus=ConsensusProtocol(strategy=ConsensusStrategy.UNANIMOUS),
        )
        votes = [Vote("a", "X", 0.7), Vote("b", "Y", 0.7)]
        result = session.run_consensus(votes)
        assert not result.reached
        assert len(failed) >= 1

    def test_run_consensus_without_protocol_raises(self, bus):
        session = AgentSession.create(bus=bus, agent_id="agent-A")
        with pytest.raises(RuntimeError, match="No ConsensusProtocol"):
            session.run_consensus([Vote("a", "X", 0.9)])

    def test_unwire_deregisters(self, bus):
        deregistered = []
        bus.subscribe(deregistered.append, EventType.AGENT_DEREGISTERED)
        session = AgentSession.create(bus=bus, agent_id="agent-A")
        session.unwire()
        assert len(deregistered) == 1

    def test_multiple_sessions_isolated(self, bus):
        """Events from session-A should not confuse session-B's reactions."""
        s_a = AgentSession.create(bus=bus, agent_id="agent-A")
        s_b = AgentSession.create(bus=bus, agent_id="agent-B")
        # Should not raise
        s_a.publish_trust_update("agent-C", 0.6)
        s_b.publish_trust_update("agent-C", 0.7)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end: gossip → policy → consensus → bus
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_coordination_round(self, bus):
        """
        Simulate a realistic multi-agent coordination round:
          1. Agent-A gossips new knowledge
          2. Policy decision is made
          3. Agents vote and reach consensus
          4. Bus reflects all event types
        """
        event_log = []
        bus.subscribe(event_log.append)   # wildcard

        session_a = AgentSession.create(
            bus       = bus,
            agent_id  = "agent-A",
            consensus = ConsensusProtocol(strategy=ConsensusStrategy.WEIGHTED_VOTE),
        )
        session_b = AgentSession.create(bus=bus, agent_id="agent-B")

        # Step 1: gossip
        session_a.publish_gossip("q3-revenue", confidence=0.88)

        # Step 2: policy decision
        bus.emit(EventType.POLICY_DECISION_MADE, "agent-A", {
            "decision": "proceed-with-launch",
            "agent_id": "agent-A",
            "was_correct": True,
        })

        # Step 3: consensus
        votes = [
            Vote("agent-A", "proceed-with-launch", 0.88),
            Vote("agent-B", "proceed-with-launch", 0.80),
        ]
        result = session_a.run_consensus(votes, round_id="launch-42")
        assert result.reached
        assert result.winning_decision == "proceed-with-launch"

        # Step 4: check bus recorded everything
        types_seen = {e.event_type for e in event_log}
        assert EventType.AGENT_REGISTERED  in types_seen
        assert EventType.GOSSIP_RECEIVED   in types_seen
        assert EventType.POLICY_DECISION_MADE in types_seen
        assert EventType.CONSENSUS_STARTED in types_seen
        assert EventType.CONSENSUS_REACHED in types_seen
