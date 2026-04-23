"""
pact_ax/primitives/trust_store.py
───────────────────────────────────
SQLite-backed persistence for TrustManager.

Profiles survive process restarts. The store is intentionally thin:
it serialises each AgentTrustProfile to JSON and writes one row per
(owner_agent, target_agent) pair. No ORM — just sqlite3.

Usage
─────
    from pact_ax.primitives.trust_score import TrustManager

    # Save after updates
    tm = TrustManager("agent-001")
    tm.update_trust("agent-002", CollaborationOutcome.POSITIVE, ContextType.TASK_KNOWLEDGE)
    tm.save("trust.db")

    # Restore in a new process
    tm2 = TrustManager.load("trust.db", agent_id="agent-001")
    print(tm2.get_trust("agent-002"))   # persisted score
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .context_share.schemas import (
    AgentTrustProfile,
    TrustEvolution,
    ContextType,
)


# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trust_profiles (
    owner_agent_id  TEXT NOT NULL,
    target_agent_id TEXT NOT NULL,
    profile_json    TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (owner_agent_id, target_agent_id)
);
"""


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _profile_to_dict(profile: AgentTrustProfile) -> Dict:
    return {
        "agent_id": profile.agent_id,
        "overall_trust": profile.overall_trust,
        "interaction_frequency": profile.interaction_frequency,
        "last_interaction": (
            profile.last_interaction.isoformat()
            if profile.last_interaction else None
        ),
        "trust_evolution": {
            ct.value: {
                "current_level":   evo.current_level,
                "trend_direction": evo.trend_direction,
                "volatility":      evo.volatility,
                "interactions":    evo.interactions,
                "last_updated": (
                    evo.last_updated.isoformat()
                    if evo.last_updated else None
                ),
            }
            for ct, evo in profile.trust_evolution.items()
        },
    }


def _profile_from_dict(data: Dict) -> AgentTrustProfile:
    profile = AgentTrustProfile(agent_id=data["agent_id"])
    profile.overall_trust = data.get("overall_trust", 0.5)
    profile.interaction_frequency = data.get("interaction_frequency", 0.0)

    raw_li = data.get("last_interaction")
    if raw_li:
        profile.last_interaction = datetime.fromisoformat(raw_li)

    for ct_value, evo_data in data.get("trust_evolution", {}).items():
        try:
            ct = ContextType(ct_value)
        except ValueError:
            continue

        evo = TrustEvolution(context_type=ct)
        evo.current_level   = evo_data.get("current_level", 0.5)
        evo.trend_direction = evo_data.get("trend_direction", "stable")
        evo.volatility      = evo_data.get("volatility", 0.1)
        evo.interactions    = evo_data.get("interactions", [])

        raw_lu = evo_data.get("last_updated")
        if raw_lu:
            evo.last_updated = datetime.fromisoformat(raw_lu)

        profile.trust_evolution[ct] = evo

    return profile


# ── TrustStore ────────────────────────────────────────────────────────────────

class TrustStore:
    """
    SQLite-backed store for AgentTrustProfile data.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file.  Use ``":memory:"`` for an in-memory store
        (useful in tests).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        # Keep a persistent connection for in-memory DBs; each new connect(":memory:")
        # would create a blank database, losing any previously created tables/rows.
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Context manager for connections ──────────────────────────────────────

    @contextmanager
    def _conn(self):
        if self._mem_conn is not None:
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception:
                self._mem_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(_SCHEMA)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save_profile(
        self,
        owner_agent_id: str,
        profile: AgentTrustProfile,
    ) -> None:
        """Upsert a single AgentTrustProfile."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trust_profiles
                    (owner_agent_id, target_agent_id, profile_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_agent_id, target_agent_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = excluded.updated_at
                """,
                (
                    owner_agent_id,
                    profile.agent_id,
                    json.dumps(_profile_to_dict(profile)),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_profiles(
        self, owner_agent_id: str
    ) -> Dict[str, AgentTrustProfile]:
        """Load all profiles for an owner agent. Returns {} if none found."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT target_agent_id, profile_json FROM trust_profiles "
                "WHERE owner_agent_id = ?",
                (owner_agent_id,),
            ).fetchall()

        result = {}
        for row in rows:
            try:
                data = json.loads(row["profile_json"])
                result[row["target_agent_id"]] = _profile_from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return result

    def save_all(
        self,
        owner_agent_id: str,
        profiles: Dict[str, AgentTrustProfile],
    ) -> None:
        """Upsert all profiles for an owner agent in a single transaction."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            for target_id, profile in profiles.items():
                conn.execute(
                    """
                    INSERT INTO trust_profiles
                        (owner_agent_id, target_agent_id, profile_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(owner_agent_id, target_agent_id) DO UPDATE SET
                        profile_json = excluded.profile_json,
                        updated_at   = excluded.updated_at
                    """,
                    (
                        owner_agent_id,
                        target_id,
                        json.dumps(_profile_to_dict(profile)),
                        now,
                    ),
                )

    def delete_profile(self, owner_agent_id: str, target_agent_id: str) -> bool:
        """Remove a single profile. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM trust_profiles "
                "WHERE owner_agent_id = ? AND target_agent_id = ?",
                (owner_agent_id, target_agent_id),
            )
        return cursor.rowcount > 0

    def list_owners(self) -> list:
        """Return all distinct owner_agent_ids in the store."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT owner_agent_id FROM trust_profiles"
            ).fetchall()
        return [row["owner_agent_id"] for row in rows]
