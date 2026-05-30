"""
pact_ax/primitives/episodic_memory.py
───────────────────────────────────────
Cross-session interaction memory for agents.

Different from StoryKeeper (which models narrative arcs), EpisodicMemory
records raw interaction episodes: who did what, when, with what outcome.
Agents can recall specific partners, filter by outcome or tags, and get
aggregate summaries for priming future interactions.

Schema
──────
    episodes (
        id              TEXT PRIMARY KEY,
        agent_id        TEXT,        -- the remembering agent
        partner_id      TEXT,        -- the other party (empty = solo action)
        action          TEXT,        -- what happened ("contract_review", "handoff", ...)
        context         TEXT,        -- JSON dict of free-form context
        outcome         TEXT,        -- positive | negative | neutral | partial
        importance      REAL,        -- 0.0–1.0 (caller-assigned salience)
        valence         TEXT,        -- positive | negative | neutral
        session_id      TEXT,        -- originating session / trace ID
        tags            TEXT,        -- JSON list of string tags
        timestamp       TEXT         -- ISO 8601
    )
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Outcome:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"
    PARTIAL  = "partial"


class Valence:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


@dataclass
class Episode:
    id:         str
    agent_id:   str
    partner_id: str
    action:     str
    context:    Dict[str, Any]
    outcome:    str
    importance: float
    valence:    str
    session_id: str
    tags:       List[str]
    timestamp:  str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicMemory:
    """
    SQLite-backed episodic memory store.

    Usage
    -----
        mem = EpisodicMemory("episodic.db")
        mem.record(
            agent_id="agent-a",
            partner_id="agent-b",
            action="contract_review",
            outcome="positive",
            importance=0.8,
            tags=["legal", "nda"],
            context={"jurisdiction": "CA"},
        )
        episodes = mem.recall(agent_id="agent-a", partner_id="agent-b")
        summary  = mem.summary("agent-a")
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS episodes (
            id         TEXT PRIMARY KEY,
            agent_id   TEXT NOT NULL,
            partner_id TEXT NOT NULL DEFAULT '',
            action     TEXT NOT NULL,
            context    TEXT NOT NULL DEFAULT '{}',
            outcome    TEXT NOT NULL DEFAULT 'neutral',
            importance REAL NOT NULL DEFAULT 0.5,
            valence    TEXT NOT NULL DEFAULT 'neutral',
            session_id TEXT NOT NULL DEFAULT '',
            tags       TEXT NOT NULL DEFAULT '[]',
            timestamp  TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Union[str, Path] = "episodic.db"):
        self._db = str(db_path)
        with sqlite3.connect(self._db) as conn:
            conn.execute(self._CREATE)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON episodes(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_partner ON episodes(agent_id, partner_id)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    # ── write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        agent_id:   str,
        action:     str,
        partner_id: str = "",
        outcome:    str = Outcome.NEUTRAL,
        importance: float = 0.5,
        valence:    str = Valence.NEUTRAL,
        session_id: str = "",
        tags:       Optional[List[str]] = None,
        context:    Optional[Dict[str, Any]] = None,
        timestamp:  Optional[str] = None,
    ) -> Episode:
        ep = Episode(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            partner_id=partner_id,
            action=action,
            context=context or {},
            outcome=outcome,
            importance=max(0.0, min(1.0, importance)),
            valence=valence,
            session_id=session_id,
            tags=tags or [],
            timestamp=timestamp or _now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes
                  (id, agent_id, partner_id, action, context, outcome,
                   importance, valence, session_id, tags, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (ep.id, ep.agent_id, ep.partner_id, ep.action,
                 json.dumps(ep.context), ep.outcome, ep.importance, ep.valence,
                 ep.session_id, json.dumps(ep.tags), ep.timestamp),
            )
        return ep

    def delete_episode(self, episode_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
            return cur.rowcount > 0

    def clear(self, agent_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM episodes WHERE agent_id=?", (agent_id,))
            return cur.rowcount

    # ── read ──────────────────────────────────────────────────────────────────

    def recall(
        self,
        agent_id:   str,
        partner_id: Optional[str] = None,
        outcome:    Optional[str] = None,
        tags:       Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit:      int = 50,
        offset:     int = 0,
    ) -> List[Episode]:
        """
        Retrieve episodes for agent_id, optionally filtered.
        Results are ordered by importance desc, then timestamp desc.
        """
        clauses = ["agent_id = ?"]
        params: List[Any] = [agent_id]

        if partner_id is not None:
            clauses.append("partner_id = ?")
            params.append(partner_id)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        if min_importance > 0.0:
            clauses.append("importance >= ?")
            params.append(min_importance)

        where = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM episodes WHERE {where} "
            f"ORDER BY importance DESC, timestamp DESC "
            f"LIMIT ? OFFSET ?"
        )
        params += [limit, offset]

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        episodes = [self._row(r) for r in rows]

        if tags:
            tag_set = set(tags)
            episodes = [e for e in episodes if tag_set.intersection(e.tags)]

        return episodes

    def recall_partner(self, agent_id: str, partner_id: str, limit: int = 20) -> List[Episode]:
        return self.recall(agent_id=agent_id, partner_id=partner_id, limit=limit)

    def summary(self, agent_id: str) -> Dict[str, Any]:
        """Aggregate statistics for an agent's episodic history."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE agent_id=?", (agent_id,)
            ).fetchone()[0]

            outcomes = conn.execute(
                "SELECT outcome, COUNT(*) as cnt FROM episodes "
                "WHERE agent_id=? GROUP BY outcome",
                (agent_id,),
            ).fetchall()

            partners = conn.execute(
                "SELECT partner_id, COUNT(*) as cnt, AVG(importance) as avg_imp "
                "FROM episodes WHERE agent_id=? AND partner_id != '' "
                "GROUP BY partner_id ORDER BY cnt DESC LIMIT 10",
                (agent_id,),
            ).fetchall()

            avg_imp = conn.execute(
                "SELECT AVG(importance) FROM episodes WHERE agent_id=?", (agent_id,)
            ).fetchone()[0]

        return {
            "agent_id":         agent_id,
            "total_episodes":   total,
            "avg_importance":   round(avg_imp or 0.0, 4),
            "outcome_breakdown": {r["outcome"]: r["cnt"] for r in outcomes},
            "top_partners": [
                {
                    "partner_id":   r["partner_id"],
                    "interactions": r["cnt"],
                    "avg_importance": round(r["avg_imp"], 4),
                }
                for r in partners
            ],
        }

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            agent_id=row["agent_id"],
            partner_id=row["partner_id"],
            action=row["action"],
            context=json.loads(row["context"]),
            outcome=row["outcome"],
            importance=row["importance"],
            valence=row["valence"],
            session_id=row["session_id"],
            tags=json.loads(row["tags"]),
            timestamp=row["timestamp"],
        )
