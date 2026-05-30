"""
pact_ax/primitives/dead_letter_queue.py
─────────────────────────────────────────
SQLite-backed Dead Letter Queue for failed agent handoffs.

When a StateTransfer packet cannot be delivered (target unreachable,
timeout, schema mismatch, etc.) it lands here for inspection and retry.

Features
────────
- Exponential backoff retry schedule
- Per-entry status: pending → retrying → exhausted | resolved
- Reason / error message preserved per attempt
- Configurable max_attempts (default 3)

Schema
──────
    dlq_entries (
        id           TEXT PRIMARY KEY,
        packet_id    TEXT,
        from_agent   TEXT,
        to_agent     TEXT,
        payload      TEXT,   -- JSON
        reason       TEXT,   -- human-readable failure description
        attempt      INTEGER,
        max_attempts INTEGER,
        status       TEXT,   -- pending | retrying | exhausted | resolved
        next_retry   TEXT,   -- ISO timestamp (NULL if resolved/exhausted)
        created_at   TEXT,
        updated_at   TEXT
    )
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DLQStatus:
    PENDING   = "pending"
    RETRYING  = "retrying"
    EXHAUSTED = "exhausted"
    RESOLVED  = "resolved"


@dataclass
class DLQEntry:
    id:           str
    packet_id:    str
    from_agent:   str
    to_agent:     str
    payload:      Dict[str, Any]
    reason:       str
    attempt:      int
    max_attempts: int
    status:       str
    next_retry:   Optional[str]
    created_at:   str
    updated_at:   str

    @property
    def exhausted(self) -> bool:
        return self.status == DLQStatus.EXHAUSTED

    @property
    def resolved(self) -> bool:
        return self.status == DLQStatus.RESOLVED

    @property
    def retryable(self) -> bool:
        return self.status in (DLQStatus.PENDING, DLQStatus.RETRYING) \
               and self.attempt < self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "retryable": self.retryable}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_retry(attempt: int, base_seconds: int = 30) -> str:
    """Exponential backoff: 30s, 60s, 120s, 240s …"""
    delay = base_seconds * (2 ** attempt)
    return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()


class DeadLetterQueue:
    """
    Persistent queue for failed StateTransfer packets.

    Usage
    -----
        dlq = DeadLetterQueue("dlq.db")
        entry = dlq.enqueue(
            packet_id="pkt-123",
            from_agent="orchestrator",
            to_agent="agent-b",
            payload={"data": "..."},
            reason="Connection refused",
        )
        # later
        dlq.retry(entry.id)          # increments attempt, updates next_retry
        dlq.resolve(entry.id)        # marks resolved
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS dlq_entries (
            id           TEXT PRIMARY KEY,
            packet_id    TEXT NOT NULL,
            from_agent   TEXT NOT NULL,
            to_agent     TEXT NOT NULL,
            payload      TEXT NOT NULL DEFAULT '{}',
            reason       TEXT NOT NULL DEFAULT '',
            attempt      INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            status       TEXT NOT NULL DEFAULT 'pending',
            next_retry   TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
    """

    def __init__(
        self,
        db_path:      Union[str, Path] = "dlq.db",
        max_attempts: int              = 3,
        base_seconds: int              = 30,
    ):
        self._db          = str(db_path)
        self._max_attempts = max_attempts
        self._base_seconds = base_seconds
        with sqlite3.connect(self._db) as conn:
            conn.execute(self._CREATE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    # ── write ─────────────────────────────────────────────────────────────────

    def enqueue(
        self,
        packet_id:    str,
        from_agent:   str,
        to_agent:     str,
        payload:      Dict[str, Any],
        reason:       str = "",
        max_attempts: Optional[int] = None,
    ) -> DLQEntry:
        now  = _now()
        eid  = str(uuid.uuid4())
        mxa  = max_attempts if max_attempts is not None else self._max_attempts
        entry = DLQEntry(
            id=eid,
            packet_id=packet_id,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            reason=reason,
            attempt=0,
            max_attempts=mxa,
            status=DLQStatus.PENDING,
            next_retry=_next_retry(0, self._base_seconds),
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dlq_entries
                  (id, packet_id, from_agent, to_agent, payload, reason,
                   attempt, max_attempts, status, next_retry, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (entry.id, entry.packet_id, entry.from_agent, entry.to_agent,
                 json.dumps(entry.payload), entry.reason,
                 entry.attempt, entry.max_attempts, entry.status,
                 entry.next_retry, entry.created_at, entry.updated_at),
            )
        return entry

    def retry(self, entry_id: str, reason: str = "") -> DLQEntry:
        """Increment attempt counter; mark exhausted if max_attempts reached."""
        entry = self._fetch(entry_id)
        if not entry:
            raise KeyError(f"DLQ entry {entry_id!r} not found")
        if entry.status in (DLQStatus.EXHAUSTED, DLQStatus.RESOLVED):
            return entry

        new_attempt = entry.attempt + 1
        if new_attempt >= entry.max_attempts:
            new_status     = DLQStatus.EXHAUSTED
            new_next_retry = None
        else:
            new_status     = DLQStatus.RETRYING
            new_next_retry = _next_retry(new_attempt, self._base_seconds)

        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dlq_entries
                   SET attempt=?, status=?, next_retry=?, reason=?, updated_at=?
                 WHERE id=?
                """,
                (new_attempt, new_status, new_next_retry,
                 reason or entry.reason, now, entry_id),
            )
        return self._fetch(entry_id)  # type: ignore[return-value]

    def resolve(self, entry_id: str) -> DLQEntry:
        """Mark a DLQ entry as successfully resolved."""
        entry = self._fetch(entry_id)
        if not entry:
            raise KeyError(f"DLQ entry {entry_id!r} not found")
        with self._connect() as conn:
            conn.execute(
                "UPDATE dlq_entries SET status=?, next_retry=NULL, updated_at=? WHERE id=?",
                (DLQStatus.RESOLVED, _now(), entry_id),
            )
        return self._fetch(entry_id)  # type: ignore[return-value]

    def delete(self, entry_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM dlq_entries WHERE id=?", (entry_id,))
            return cur.rowcount > 0

    # ── read ──────────────────────────────────────────────────────────────────

    def pending(self) -> List[DLQEntry]:
        return self._query("WHERE status IN ('pending','retrying') ORDER BY created_at")

    def exhausted(self) -> List[DLQEntry]:
        return self._query("WHERE status='exhausted' ORDER BY created_at")

    def resolved(self) -> List[DLQEntry]:
        return self._query("WHERE status='resolved' ORDER BY updated_at DESC")

    def all(self) -> List[DLQEntry]:
        return self._query("ORDER BY created_at DESC")

    def get(self, entry_id: str) -> Optional[DLQEntry]:
        return self._fetch(entry_id)

    def stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM dlq_entries GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        return {
            "pending":   counts.get(DLQStatus.PENDING, 0),
            "retrying":  counts.get(DLQStatus.RETRYING, 0),
            "exhausted": counts.get(DLQStatus.EXHAUSTED, 0),
            "resolved":  counts.get(DLQStatus.RESOLVED, 0),
            "total":     sum(counts.values()),
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _query(self, where_clause: str) -> List[DLQEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM dlq_entries {where_clause}"
            ).fetchall()
        return [self._row(r) for r in rows]

    def _fetch(self, entry_id: str) -> Optional[DLQEntry]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dlq_entries WHERE id=?", (entry_id,)
            ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> DLQEntry:
        return DLQEntry(
            id=row["id"],
            packet_id=row["packet_id"],
            from_agent=row["from_agent"],
            to_agent=row["to_agent"],
            payload=json.loads(row["payload"]),
            reason=row["reason"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            status=row["status"],
            next_retry=row["next_retry"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
