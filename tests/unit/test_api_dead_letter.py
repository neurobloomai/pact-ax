"""
tests/unit/test_api_dead_letter.py
────────────────────────────────────
Tests for /dlq/* REST endpoints.

Run with:  pytest tests/unit/test_api_dead_letter.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.dead_letter as dlq_module
from pact_ax.primitives.dead_letter_queue import DeadLetterQueue

client = TestClient(app, raise_server_exceptions=True)

ENQUEUE = {
    "packet_id":  "pkt-001",
    "from_agent": "orchestrator",
    "to_agent":   "agent-b",
    "payload":    {"task": "review"},
    "reason":     "Connection timeout",
}


@pytest.fixture(autouse=True)
def fresh_dlq(tmp_path):
    dlq_module._dlq = DeadLetterQueue(tmp_path / "dlq.db", max_attempts=3, base_seconds=1)
    yield
    dlq_module._dlq = DeadLetterQueue(":memory:", max_attempts=3, base_seconds=1)


def enqueue(overrides=None):
    body = {**ENQUEUE, **(overrides or {})}
    return client.post("/dlq/enqueue", json=body)


# ── POST /dlq/enqueue ─────────────────────────────────────────────────────────

class TestEnqueue:

    def test_returns_200(self):
        assert enqueue().status_code == 200

    def test_response_has_id_and_status(self):
        body = enqueue().json()
        assert "id" in body
        assert body["status"] == "pending"
        assert body["attempt"] == 0

    def test_preserves_fields(self):
        body = enqueue().json()
        assert body["packet_id"]  == "pkt-001"
        assert body["from_agent"] == "orchestrator"
        assert body["to_agent"]   == "agent-b"

    def test_custom_max_attempts(self):
        body = enqueue({"max_attempts": 5}).json()
        assert body["max_attempts"] == 5

    def test_missing_packet_id_returns_422(self):
        r = client.post("/dlq/enqueue", json={"from_agent": "a", "to_agent": "b"})
        assert r.status_code == 422


# ── GET /dlq/pending ──────────────────────────────────────────────────────────

class TestPending:

    def test_newly_enqueued_is_pending(self):
        enqueue()
        r = client.get("/dlq/pending")
        assert r.json()["count"] >= 1

    def test_resolved_not_in_pending(self):
        eid = enqueue().json()["id"]
        client.post(f"/dlq/{eid}/resolve")
        r = client.get("/dlq/pending")
        ids = {e["id"] for e in r.json()["entries"]}
        assert eid not in ids


# ── GET /dlq/stats ────────────────────────────────────────────────────────────

class TestStats:

    def test_total_count(self):
        enqueue()
        enqueue({"packet_id": "pkt-002"})
        r = client.get("/dlq/stats")
        assert r.json()["total"] >= 2

    def test_resolved_counted(self):
        eid = enqueue().json()["id"]
        client.post(f"/dlq/{eid}/resolve")
        r = client.get("/dlq/stats")
        assert r.json()["resolved"] >= 1


# ── GET /dlq/{id} ─────────────────────────────────────────────────────────────

class TestGetEntry:

    def test_returns_entry(self):
        eid = enqueue().json()["id"]
        r = client.get(f"/dlq/{eid}")
        assert r.status_code == 200
        assert r.json()["id"] == eid

    def test_404_for_unknown(self):
        r = client.get("/dlq/nonexistent-id")
        assert r.status_code == 404


# ── POST /dlq/{id}/retry ──────────────────────────────────────────────────────

class TestRetry:

    def test_increments_attempt(self):
        eid = enqueue().json()["id"]
        body = client.post(f"/dlq/{eid}/retry").json()
        assert body["attempt"] == 1
        assert body["status"] == "retrying"

    def test_exhausted_after_max(self):
        eid = enqueue().json()["id"]
        for _ in range(3):
            client.post(f"/dlq/{eid}/retry")
        r = client.get(f"/dlq/{eid}")
        assert r.json()["status"] == "exhausted"

    def test_404_for_unknown(self):
        r = client.post("/dlq/nonexistent-id/retry")
        assert r.status_code == 404


# ── POST /dlq/{id}/resolve ────────────────────────────────────────────────────

class TestResolve:

    def test_marks_resolved(self):
        eid = enqueue().json()["id"]
        body = client.post(f"/dlq/{eid}/resolve").json()
        assert body["status"] == "resolved"

    def test_404_for_unknown(self):
        r = client.post("/dlq/nonexistent-id/resolve")
        assert r.status_code == 404


# ── DELETE /dlq/{id} ──────────────────────────────────────────────────────────

class TestDelete:

    def test_removes_entry(self):
        eid = enqueue().json()["id"]
        r = client.delete(f"/dlq/{eid}")
        assert r.status_code == 200
        assert client.get(f"/dlq/{eid}").status_code == 404

    def test_404_for_unknown(self):
        r = client.delete("/dlq/nonexistent-id")
        assert r.status_code == 404


# ── GET /dlq (full dump) ──────────────────────────────────────────────────────

class TestFullDump:

    def test_returns_all_entries(self):
        enqueue()
        enqueue({"packet_id": "pkt-002"})
        r = client.get("/dlq")
        assert r.json()["count"] >= 2
