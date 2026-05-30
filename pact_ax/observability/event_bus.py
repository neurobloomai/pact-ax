"""
pact_ax/observability/event_bus.py
────────────────────────────────────
Global event bus for the Seam Observer.
Routes emit events here; the SSE endpoint streams them to the browser.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SeamEvent:
    event_type: str
    data: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "event_id":   self.event_id,
            "event_type": self.event_type,
            "ts":         self.ts,
            "data":       self.data,
        }


class EventBus:
    def __init__(self):
        self._events: List[SeamEvent] = []
        self._agents: Dict[str, Dict] = {}
        self._edges:  Dict[str, Dict] = {}

    # ── emit ─────────────────────────────────────────────────────────────────

    def emit(self, event_type: str, **data) -> SeamEvent:
        event = SeamEvent(event_type=event_type, data=data)
        self._events.append(event)
        self._update_state(event)
        return event

    # ── state tracking ────────────────────────────────────────────────────────

    def _update_state(self, event: SeamEvent):
        d = event.data

        if event.event_type == "agent_registered":
            self._agents[d["agent_id"]] = {
                "role": d.get("role", ""),
                "capabilities": d.get("capabilities", []),
            }

        elif event.event_type == "trust_updated":
            key = f"{d['from_agent']}→{d['to_agent']}"
            e = self._edges.setdefault(key, {"from": d["from_agent"], "to": d["to_agent"], "packets": 0})
            e["trust"] = round(d.get("new_score", 0.5), 3)

        elif event.event_type in ("packet_prepared", "packet_sent", "packet_received"):
            fa, ta = d.get("from_agent", ""), d.get("to_agent", "")
            key = f"{fa}→{ta}"
            e = self._edges.setdefault(key, {"from": fa, "to": ta, "packets": 0})
            if event.event_type == "packet_received" and d.get("success"):
                e["packets"] = e.get("packets", 0) + 1

    # ── query ─────────────────────────────────────────────────────────────────

    def events_since(self, index: int) -> List[SeamEvent]:
        return self._events[index:]

    def total(self) -> int:
        return len(self._events)

    def snapshot(self) -> Dict:
        return {
            "agents":      self._agents,
            "edges":       self._edges,
            "event_count": len(self._events),
        }

    def reset(self):
        self._events.clear()
        self._agents.clear()
        self._edges.clear()


# ── module-level singleton ────────────────────────────────────────────────────

_bus = EventBus()


def get_bus() -> EventBus:
    return _bus
