"""
pact_ax/storage/story_store.py
──────────────────────────────
SQLite persistence for StoryKeeper.

Schema
──────
  story_keepers  — one row per agent: arc, session, timestamps
  story_state    — latest story_state snapshot per agent (JSON)
  interactions   — append-only log of every processed interaction
  arc_history    — log of arc transitions
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


_DDL = """
CREATE TABLE IF NOT EXISTS story_keepers (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT,
    current_arc TEXT NOT NULL DEFAULT 'exploration',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS story_state (
    agent_id   TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES story_keepers(agent_id)
);

CREATE TABLE IF NOT EXISTS interactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    user_input     TEXT    NOT NULL,
    agent_response TEXT    NOT NULL DEFAULT '',
    arc            TEXT    NOT NULL,
    metadata_json  TEXT    NOT NULL DEFAULT '{}',
    FOREIGN KEY (agent_id) REFERENCES story_keepers(agent_id)
);

CREATE TABLE IF NOT EXISTS arc_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id          TEXT    NOT NULL,
    from_arc          TEXT    NOT NULL,
    to_arc            TEXT    NOT NULL,
    at                TEXT    NOT NULL,
    interaction_count INTEGER NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES story_keepers(agent_id)
);
"""


class StoryStore:
    """SQLite-backed persistence for StoryKeeper instances."""

    def __init__(self, db_path: str = "story_keeper.db"):
        self.db_path = db_path
        self._init_schema()

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ── write ─────────────────────────────────────────────────────────────────

    def save_keeper(self, keeper: Any) -> None:
        """Upsert keeper metadata + latest story_state."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO story_keepers (agent_id, session_id, current_arc, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    current_arc = excluded.current_arc,
                    session_id  = excluded.session_id,
                    updated_at  = excluded.updated_at
                """,
                (keeper.agent_id, keeper.session_id, keeper.current_arc.value, now, now),
            )
            conn.execute(
                """
                INSERT INTO story_state (agent_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (keeper.agent_id, json.dumps(keeper.story_state), now),
            )

    def save_interaction(self, agent_id: str, interaction: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions
                    (agent_id, timestamp, user_input, agent_response, arc, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    interaction["timestamp"].isoformat(),
                    interaction.get("user_input", ""),
                    interaction.get("agent_response", ""),
                    interaction["arc"].value,
                    json.dumps(interaction.get("metadata", {})),
                ),
            )

    def save_arc_transition(self, agent_id: str, transition: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO arc_history (agent_id, from_arc, to_arc, at, interaction_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    transition["from"].value,
                    transition["to"].value,
                    transition["at"].isoformat(),
                    transition["interaction_count"],
                ),
            )

    # ── read ──────────────────────────────────────────────────────────────────

    def load_keeper_data(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return all persisted data for agent_id, or None if unknown."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM story_keepers WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                return None

            state_row = conn.execute(
                "SELECT state_json FROM story_state WHERE agent_id = ?", (agent_id,)
            ).fetchone()

            interactions = conn.execute(
                "SELECT * FROM interactions WHERE agent_id = ? ORDER BY id", (agent_id,)
            ).fetchall()

            arc_history = conn.execute(
                "SELECT * FROM arc_history WHERE agent_id = ? ORDER BY id", (agent_id,)
            ).fetchall()

        return {
            "agent_id":    row["agent_id"],
            "session_id":  row["session_id"],
            "current_arc": row["current_arc"],
            "story_state": json.loads(state_row["state_json"]) if state_row else None,
            "interactions": [
                {
                    "timestamp":      datetime.fromisoformat(r["timestamp"]),
                    "user_input":     r["user_input"],
                    "agent_response": r["agent_response"],
                    "arc":            r["arc"],
                    "metadata":       json.loads(r["metadata_json"]),
                }
                for r in interactions
            ],
            "arc_history": [
                {
                    "from":              r["from_arc"],
                    "to":                r["to_arc"],
                    "at":                datetime.fromisoformat(r["at"]),
                    "interaction_count": r["interaction_count"],
                }
                for r in arc_history
            ],
        }

    def agent_exists(self, agent_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM story_keepers WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            return row is not None

    def list_agents(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT agent_id FROM story_keepers").fetchall()
            return [r["agent_id"] for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete_keeper(self, agent_id: str) -> None:
        with self._connect() as conn:
            for table in ("arc_history", "interactions", "story_state", "story_keepers"):
                conn.execute(f"DELETE FROM {table} WHERE agent_id = ?", (agent_id,))
