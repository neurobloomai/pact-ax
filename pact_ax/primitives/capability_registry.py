"""
pact_ax/primitives/capability_registry.py
──────────────────────────────────────────
Agent capability registration and discovery.

Agents register what they can do (skills/intents). Other agents query
"who can handle X?" — optionally filtered by minimum trust score.

SQLite-backed so registrations survive restarts.

Schema
──────
    capabilities (
        agent_id    TEXT,
        skill       TEXT,
        description TEXT,
        tags        TEXT,   -- JSON list
        version     TEXT,
        updated_at  TEXT,
        PRIMARY KEY (agent_id, skill)
    )
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class Capability:
    agent_id:    str
    skill:       str
    description: str = ""
    tags:        List[str] = field(default_factory=list)
    version:     str = "1.0"
    updated_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


class CapabilityRegistry:
    """
    Persistent registry mapping agents to their declared capabilities.

    Usage
    -----
        reg = CapabilityRegistry("capabilities.db")
        reg.register("agent-a", "contract_review", "Reviews NDAs and service agreements", tags=["legal"])
        candidates = reg.find_capable("contract_review")
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS capabilities (
            agent_id    TEXT NOT NULL,
            skill       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '[]',
            version     TEXT NOT NULL DEFAULT '1.0',
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (agent_id, skill)
        )
    """

    def __init__(self, db_path: Union[str, Path] = "capabilities.db"):
        self._db = str(db_path)
        # For :memory: keep one persistent connection; file DBs get a new one per call.
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self._db == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        with self._connect() as conn:
            conn.execute(self._CREATE)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    # ── write ─────────────────────────────────────────────────────────────────

    def register(
        self,
        agent_id:    str,
        skill:       str,
        description: str = "",
        tags:        Optional[List[str]] = None,
        version:     str = "1.0",
    ) -> Capability:
        cap = Capability(
            agent_id=agent_id,
            skill=skill,
            description=description,
            tags=tags or [],
            version=version,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO capabilities (agent_id, skill, description, tags, version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, skill) DO UPDATE SET
                    description = excluded.description,
                    tags        = excluded.tags,
                    version     = excluded.version,
                    updated_at  = excluded.updated_at
                """,
                (cap.agent_id, cap.skill, cap.description,
                 json.dumps(cap.tags), cap.version, cap.updated_at),
            )
        return cap

    def deregister(self, agent_id: str, skill: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM capabilities WHERE agent_id=? AND skill=?",
                (agent_id, skill),
            )
            return cur.rowcount > 0

    def deregister_agent(self, agent_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM capabilities WHERE agent_id=?", (agent_id,)
            )
            return cur.rowcount

    # ── read ──────────────────────────────────────────────────────────────────

    def get_agent_capabilities(self, agent_id: str) -> List[Capability]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM capabilities WHERE agent_id=? ORDER BY skill",
                (agent_id,),
            ).fetchall()
        return [self._row_to_cap(r) for r in rows]

    def find_capable(self, skill: str) -> List[Capability]:
        """Return all agents that registered the given skill (exact match)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM capabilities WHERE skill=? ORDER BY agent_id",
                (skill,),
            ).fetchall()
        return [self._row_to_cap(r) for r in rows]

    def search(self, query: str) -> List[Capability]:
        """
        Search skill names and descriptions for *query* (case-insensitive
        substring match). Useful when the caller doesn't know the exact skill name.
        """
        like = f"%{query.lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM capabilities
                WHERE lower(skill) LIKE ?
                   OR lower(description) LIKE ?
                   OR lower(tags) LIKE ?
                ORDER BY skill, agent_id
                """,
                (like, like, like),
            ).fetchall()
        return [self._row_to_cap(r) for r in rows]

    def all_capabilities(self) -> List[Capability]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM capabilities ORDER BY skill, agent_id"
            ).fetchall()
        return [self._row_to_cap(r) for r in rows]

    def all_skills(self) -> List[str]:
        """Unique skill names across all agents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT skill FROM capabilities ORDER BY skill"
            ).fetchall()
        return [r["skill"] for r in rows]

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_cap(row: sqlite3.Row) -> Capability:
        return Capability(
            agent_id=row["agent_id"],
            skill=row["skill"],
            description=row["description"],
            tags=json.loads(row["tags"]),
            version=row["version"],
            updated_at=row["updated_at"],
        )
