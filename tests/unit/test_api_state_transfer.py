"""
tests/unit/test_api_state_transfer.py
──────────────────────────────────────
Integration tests for /transfer/* REST endpoints.

Run with:  pytest tests/unit/test_api_state_transfer.py -v
"""

import pytest
from fastapi.testclient import TestClient

from pact_ax.api.server import app
import pact_ax.api.routes.state_transfer as st_module

client = TestClient(app, raise_server_exceptions=True)


# ── Fixture: clear registry between tests ────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_registry():
    st_module._managers.clear()
    yield
    st_module._managers.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare(from_agent="sender", to_agent="receiver", state_data=None):
    return client.post("/transfer/prepare", json={
        "from_agent_id": from_agent,
        "to_agent_id":   to_agent,
        "state_data":    state_data or {"task": "analyse Q3", "progress": 0.6},
    })


def _full_packet(from_agent="sender", to_agent="receiver"):
    r_prep = _prepare(from_agent, to_agent)
    packet_id = r_prep.json()["packet_id"]
    r_send = client.post("/transfer/send", json={
        "agent_id": from_agent, "packet_id": packet_id,
    })
    return r_send.json()   # serialised HandoffPacket dict


# ── POST /transfer/prepare ────────────────────────────────────────────────────

class TestPrepare:

    def test_returns_200(self):
        assert _prepare().status_code == 200

    def test_response_has_packet_id(self):
        body = _prepare().json()
        assert "packet_id" in body
        assert body["packet_id"].startswith("pkt-")

    def test_status_is_preparing(self):
        body = _prepare().json()
        assert body["status"] == "preparing"

    def test_agents_in_response(self):
        body = _prepare("agent-A", "agent-B").json()
        assert body["from_agent"] == "agent-A"
        assert body["to_agent"]   == "agent-B"

    def test_invalid_reason_returns_422(self):
        r = client.post("/transfer/prepare", json={
            "from_agent_id": "a", "to_agent_id": "b",
            "state_data": {"x": 1}, "reason": "bad_reason",
        })
        assert r.status_code == 422

    def test_all_valid_reasons_accepted(self):
        for reason in ("continuation", "escalation", "pause",
                        "completion", "load_balance"):
            r = client.post("/transfer/prepare", json={
                "from_agent_id": "a", "to_agent_id": "b",
                "state_data": {"x": 1}, "reason": reason,
            })
            assert r.status_code == 200, f"Failed for reason={reason}"


# ── POST /transfer/send ───────────────────────────────────────────────────────

class TestSend:

    def test_returns_200(self):
        r_prep = _prepare()
        r_send = client.post("/transfer/send", json={
            "agent_id": "sender",
            "packet_id": r_prep.json()["packet_id"],
        })
        assert r_send.status_code == 200

    def test_status_is_in_flight(self):
        r_prep = _prepare()
        r_send = client.post("/transfer/send", json={
            "agent_id": "sender",
            "packet_id": r_prep.json()["packet_id"],
        })
        assert r_send.json()["status"] == "in_flight"

    def test_unknown_packet_id_returns_404(self):
        r = client.post("/transfer/send", json={
            "agent_id": "sender", "packet_id": "pkt-nonexistent",
        })
        assert r.status_code == 404

    def test_double_send_returns_409(self):
        r_prep = _prepare()
        pid = r_prep.json()["packet_id"]
        client.post("/transfer/send", json={"agent_id": "sender", "packet_id": pid})
        r2 = client.post("/transfer/send", json={"agent_id": "sender", "packet_id": pid})
        assert r2.status_code == 409


# ── POST /transfer/receive ────────────────────────────────────────────────────

class TestReceive:

    def test_returns_200(self):
        packet = _full_packet()
        r = client.post("/transfer/receive", json={
            "agent_id": "receiver", "packet": packet,
        })
        assert r.status_code == 200

    def test_success_true(self):
        packet = _full_packet()
        r = client.post("/transfer/receive", json={
            "agent_id": "receiver", "packet": packet,
        })
        assert r.json()["success"] is True

    def test_integrated_state_has_state_data(self):
        state = {"task": "analyse Q3", "progress": 0.6}
        packet = _full_packet()
        r = client.post("/transfer/receive", json={
            "agent_id": "receiver", "packet": packet,
        })
        assert r.json()["integrated_state"]["state_data"] == state

    def test_wrong_recipient_fails(self):
        packet = _full_packet()
        r = client.post("/transfer/receive", json={
            "agent_id": "wrong-agent", "packet": packet,
        })
        assert r.json()["success"] is False

    def test_invalid_packet_returns_422(self):
        r = client.post("/transfer/receive", json={
            "agent_id": "receiver", "packet": {"not": "a valid packet"},
        })
        assert r.status_code == 422


# ── POST /transfer/rollback ───────────────────────────────────────────────────

class TestRollback:

    def test_rollback_integrated_packet(self):
        packet = _full_packet()
        client.post("/transfer/receive", json={"agent_id": "receiver", "packet": packet})
        r = client.post("/transfer/rollback", json={
            "agent_id": "receiver", "packet_id": packet["packet_id"],
        })
        assert r.status_code == 200
        assert r.json()["rolled_back"] is True

    def test_rollback_unknown_id_returns_409(self):
        r = client.post("/transfer/rollback", json={
            "agent_id": "receiver", "packet_id": "pkt-unknown",
        })
        assert r.status_code == 409


# ── POST /transfer/checkpoint ─────────────────────────────────────────────────

class TestCheckpoint:

    def test_returns_checkpoint_id(self):
        r = client.post("/transfer/checkpoint", json={
            "agent_id": "agent-001",
            "label": "before-deploy",
            "state_data": {"step": 1},
        })
        assert r.status_code == 200
        assert r.json()["checkpoint_id"].startswith("ckpt-")

    def test_label_in_response(self):
        r = client.post("/transfer/checkpoint", json={
            "agent_id": "agent-001",
            "label": "my-label",
            "state_data": {"x": 1},
        })
        assert r.json()["label"] == "my-label"


# ── POST /transfer/restore ────────────────────────────────────────────────────

class TestRestore:

    def test_restore_returns_snapshot(self):
        r_ckpt = client.post("/transfer/checkpoint", json={
            "agent_id": "agent-001",
            "label": "snap",
            "state_data": {"step": 42},
        })
        ckpt_id = r_ckpt.json()["checkpoint_id"]

        r_restore = client.post("/transfer/restore", json={
            "agent_id": "agent-001",
            "checkpoint_id": ckpt_id,
        })
        assert r_restore.status_code == 200
        assert r_restore.json()["state_data"] == {"step": 42}

    def test_unknown_checkpoint_returns_404(self):
        r = client.post("/transfer/restore", json={
            "agent_id": "agent-001",
            "checkpoint_id": "ckpt-doesnotexist",
        })
        assert r.status_code == 404


# ── GET /transfer/status ──────────────────────────────────────────────────────

class TestStatus:

    def test_preparing_status(self):
        r_prep = _prepare()
        pid = r_prep.json()["packet_id"]
        r = client.get(f"/transfer/status/sender/{pid}")
        assert r.status_code == 200
        assert r.json()["status"] == "preparing"

    def test_in_flight_after_send(self):
        r_prep = _prepare()
        pid = r_prep.json()["packet_id"]
        client.post("/transfer/send", json={"agent_id": "sender", "packet_id": pid})
        r = client.get(f"/transfer/status/sender/{pid}")
        assert r.json()["status"] == "in_flight"

    def test_unknown_packet_returns_404(self):
        r = client.get("/transfer/status/sender/pkt-unknown")
        assert r.status_code == 404


# ── GET /transfer/checkpoints/{agent_id} ─────────────────────────────────────

class TestListCheckpoints:

    def test_empty_when_no_checkpoints(self):
        r = client.get("/transfer/checkpoints/agent-new")
        assert r.status_code == 200
        assert r.json() == []

    def test_lists_created_checkpoints(self):
        for label in ("v1", "v2"):
            client.post("/transfer/checkpoint", json={
                "agent_id": "agent-001",
                "label": label,
                "state_data": {"x": 1},
            })
        r = client.get("/transfer/checkpoints/agent-001")
        assert len(r.json()) == 2


# ── GET /transfer/summary/{agent_id} ─────────────────────────────────────────

class TestSummary:

    def test_returns_summary_keys(self):
        r = client.get("/transfer/summary/agent-001")
        assert r.status_code == 200
        for key in ("agent_id", "outbound_count", "inbound_count",
                    "active_transfers", "status_breakdown", "trust_floor"):
            assert key in r.json()

    def test_outbound_count_increments(self):
        _prepare()
        r = client.get("/transfer/summary/sender")
        assert r.json()["outbound_count"] == 1
