"""
tests/unit/test_dead_letter_queue.py
──────────────────────────────────────
Unit tests for DeadLetterQueue.

Run with:  pytest tests/unit/test_dead_letter_queue.py -v
"""

import pytest
from pact_ax.primitives.dead_letter_queue import DeadLetterQueue, DLQEntry, DLQStatus


@pytest.fixture
def dlq(tmp_path):
    return DeadLetterQueue(tmp_path / "dlq.db", max_attempts=3, base_seconds=1)


@pytest.fixture
def entry(dlq):
    return dlq.enqueue(
        packet_id="pkt-001",
        from_agent="orchestrator",
        to_agent="agent-b",
        payload={"task": "review"},
        reason="Connection timeout",
    )


# ── enqueue ───────────────────────────────────────────────────────────────────

class TestEnqueue:

    def test_returns_dlq_entry(self, dlq):
        e = dlq.enqueue("pkt-1", "orch", "b", {})
        assert isinstance(e, DLQEntry)

    def test_initial_status_is_pending(self, entry):
        assert entry.status == DLQStatus.PENDING

    def test_initial_attempt_is_zero(self, entry):
        assert entry.attempt == 0

    def test_preserves_payload(self, dlq):
        payload = {"key": "value", "nested": {"x": 1}}
        e = dlq.enqueue("pkt-x", "a", "b", payload)
        loaded = dlq.get(e.id)
        assert loaded.payload == payload

    def test_preserves_reason(self, entry):
        assert entry.reason == "Connection timeout"

    def test_custom_max_attempts(self, dlq):
        e = dlq.enqueue("pkt-x", "a", "b", {}, max_attempts=5)
        assert e.max_attempts == 5

    def test_has_next_retry(self, entry):
        assert entry.next_retry is not None

    def test_retryable_when_pending(self, entry):
        assert entry.retryable is True


# ── retry ─────────────────────────────────────────────────────────────────────

class TestRetry:

    def test_increments_attempt(self, dlq, entry):
        e2 = dlq.retry(entry.id)
        assert e2.attempt == 1

    def test_status_becomes_retrying(self, dlq, entry):
        e2 = dlq.retry(entry.id)
        assert e2.status == DLQStatus.RETRYING

    def test_exhausted_after_max_attempts(self, dlq, entry):
        dlq.retry(entry.id)
        dlq.retry(entry.id)
        e = dlq.retry(entry.id)
        assert e.status == DLQStatus.EXHAUSTED
        assert e.attempt == 3

    def test_exhausted_has_no_next_retry(self, dlq, entry):
        dlq.retry(entry.id)
        dlq.retry(entry.id)
        e = dlq.retry(entry.id)
        assert e.next_retry is None

    def test_retry_exhausted_entry_is_noop(self, dlq, entry):
        for _ in range(3):
            dlq.retry(entry.id)
        e = dlq.retry(entry.id)  # 4th call — already exhausted
        assert e.attempt == 3
        assert e.status == DLQStatus.EXHAUSTED

    def test_raises_for_missing_entry(self, dlq):
        with pytest.raises(KeyError):
            dlq.retry("nonexistent-id")

    def test_not_retryable_when_exhausted(self, dlq, entry):
        for _ in range(3):
            dlq.retry(entry.id)
        e = dlq.get(entry.id)
        assert e.retryable is False


# ── resolve ───────────────────────────────────────────────────────────────────

class TestResolve:

    def test_status_becomes_resolved(self, dlq, entry):
        e = dlq.resolve(entry.id)
        assert e.status == DLQStatus.RESOLVED

    def test_next_retry_cleared(self, dlq, entry):
        e = dlq.resolve(entry.id)
        assert e.next_retry is None

    def test_resolved_is_not_retryable(self, dlq, entry):
        e = dlq.resolve(entry.id)
        assert e.retryable is False

    def test_raises_for_missing_entry(self, dlq):
        with pytest.raises(KeyError):
            dlq.resolve("nonexistent-id")


# ── query methods ─────────────────────────────────────────────────────────────

class TestQueryMethods:

    def test_pending_includes_new_entry(self, dlq, entry):
        pending = dlq.pending()
        assert any(e.id == entry.id for e in pending)

    def test_pending_excludes_resolved(self, dlq, entry):
        dlq.resolve(entry.id)
        pending = dlq.pending()
        assert not any(e.id == entry.id for e in pending)

    def test_exhausted_list(self, dlq, entry):
        for _ in range(3):
            dlq.retry(entry.id)
        exhausted = dlq.exhausted()
        assert any(e.id == entry.id for e in exhausted)

    def test_get_by_id(self, dlq, entry):
        e = dlq.get(entry.id)
        assert e is not None
        assert e.id == entry.id

    def test_get_nonexistent_returns_none(self, dlq):
        assert dlq.get("nonexistent") is None

    def test_delete_removes_entry(self, dlq, entry):
        assert dlq.delete(entry.id) is True
        assert dlq.get(entry.id) is None

    def test_delete_nonexistent_returns_false(self, dlq):
        assert dlq.delete("nonexistent") is False


# ── stats ─────────────────────────────────────────────────────────────────────

class TestStats:

    def test_counts_by_status(self, dlq):
        e1 = dlq.enqueue("p1", "a", "b", {})
        e2 = dlq.enqueue("p2", "a", "c", {})
        dlq.resolve(e1.id)
        dlq.retry(e2.id)
        dlq.retry(e2.id)
        dlq.retry(e2.id)  # exhausted

        s = dlq.stats()
        assert s["resolved"] == 1
        assert s["exhausted"] == 1
        assert s["pending"] == 0
        assert s["total"] == 2

    def test_empty_queue_stats(self, dlq):
        s = dlq.stats()
        assert s["total"] == 0
        assert s["pending"] == 0


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_survives_new_instance(self, tmp_path):
        db = tmp_path / "dlq.db"
        d1 = DeadLetterQueue(db)
        e = d1.enqueue("pkt-x", "a", "b", {"data": "test"})
        d2 = DeadLetterQueue(db)
        loaded = d2.get(e.id)
        assert loaded is not None
        assert loaded.payload == {"data": "test"}
        assert loaded.status == DLQStatus.PENDING
