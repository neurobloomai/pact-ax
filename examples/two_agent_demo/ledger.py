"""
examples/two_agent_demo/ledger.py
───────────────────────────────────
SQLite-backed ledger for the billing escalation demo.

Persists two things across runs:
  1. Trust profiles  — via TrustStore (AgentTrustProfile per agent pair)
  2. Policy outcomes — PolicyLearning.outcomes list (JSON rows)

Usage
─────
    ledger = DemoLedger("demo_history.db")
    ledger.load(cs_module, pa_module)   # restore before run
    ...run the demo...
    ledger.save(cs_module, pa_module)   # persist after run
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pact_ax.primitives.trust_store import TrustStore


_OUTCOMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


class DemoLedger:
    """
    Thin persistence wrapper for the two-agent demo.

    Parameters
    ----------
    db_path : str or Path
        SQLite file.  Created on first use.  Use ":memory:" in tests.
    """

    def __init__(self, db_path: str | Path = "demo_history.db") -> None:
        self.db_path = str(db_path)
        self._trust_store = TrustStore(db_path)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
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
            conn.execute(_OUTCOMES_SCHEMA)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, cs_module: Any, pa_module: Any) -> int:
        """
        Restore persisted state into the live module registries.

        Returns total number of records loaded (trust profiles + outcomes).
        """
        loaded = 0

        # Trust profiles — for every agent registered in cs_module._managers
        for agent_id, mgr in cs_module._managers.items():
            profiles = self._trust_store.load_profiles(agent_id)
            if profiles:
                mgr.trust_profiles = profiles
                loaded += len(profiles)

        # Policy outcomes
        outcomes = self._load_outcomes()
        if outcomes:
            pa_module._learner.outcomes = outcomes
            loaded += len(outcomes)

        return loaded

    def _load_outcomes(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT outcome_json FROM policy_outcomes ORDER BY id"
            ).fetchall()
        results = []
        for row in rows:
            try:
                d = json.loads(row["outcome_json"])
                # Re-hydrate timestamp string → datetime
                if isinstance(d.get("timestamp"), str):
                    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                results.append(d)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return results

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, cs_module: Any, pa_module: Any) -> int:
        """
        Persist current in-memory state to SQLite.

        Returns total number of records saved.
        """
        saved = 0

        # Trust profiles
        for agent_id, mgr in cs_module._managers.items():
            if mgr.trust_profiles:
                self._trust_store.save_all(agent_id, mgr.trust_profiles)
                saved += len(mgr.trust_profiles)

        # Policy outcomes — append only (don't rewrite rows already saved)
        saved += self._append_new_outcomes(pa_module._learner.outcomes)

        return saved

    def _append_new_outcomes(self, outcomes: List[Dict]) -> int:
        """Insert outcomes that aren't already in the DB (by count offset)."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM policy_outcomes"
            ).fetchone()[0]

        new_outcomes = outcomes[existing:]  # only append the tail
        if not new_outcomes:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            for o in new_outcomes:
                serialisable = {
                    **o,
                    "timestamp": (
                        o["timestamp"].isoformat()
                        if isinstance(o["timestamp"], datetime)
                        else o["timestamp"]
                    ),
                }
                conn.execute(
                    "INSERT INTO policy_outcomes (agent_id, outcome_json, recorded_at) "
                    "VALUES (?, ?, ?)",
                    (o.get("agent_id", "unknown"), json.dumps(serialisable), now),
                )
        return len(new_outcomes)

    # ── Summary ───────────────────────────────────────────────────────────────

    def history_summary(self) -> Dict[str, Any]:
        """Return a quick summary of what's in the ledger."""
        with self._conn() as conn:
            outcome_count = conn.execute(
                "SELECT COUNT(*) FROM policy_outcomes"
            ).fetchone()[0]
            agent_counts = conn.execute(
                "SELECT agent_id, COUNT(*) as n FROM policy_outcomes GROUP BY agent_id"
            ).fetchall()

        trust_owners = self._trust_store.list_owners()
        return {
            "total_policy_outcomes": outcome_count,
            "outcomes_by_agent": {row["agent_id"]: row["n"] for row in agent_counts},
            "trust_owners": trust_owners,
        }
