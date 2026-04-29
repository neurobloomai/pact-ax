"""
pact_ax/access/keys.py
───────────────────────
API key generation and SQLite-backed storage for free tier users.

Key format: pax_<32 hex chars>
            └─ identifiable prefix  └─ 128 bits of entropy

Schema is created on first use. No migration framework needed at this scale.
"""

import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

_KEY_PREFIX = "pax_"
_DB_DEFAULT  = "access.db"


@dataclass
class APIKey:
    key:        str
    email:      str
    org:        str       # domain extracted from email
    tier:       str = "free"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active:     bool = True


def generate_key() -> str:
    """Return a new unique API key with the pax_ prefix."""
    return _KEY_PREFIX + secrets.token_hex(16)


class KeyStore:
    """SQLite-backed store for API keys."""

    def __init__(self, db_path: Union[str, Path] = _DB_DEFAULT):
        self._path = str(db_path)
        # Hold a single connection so :memory: databases survive across calls
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key         TEXT PRIMARY KEY,
                    email       TEXT NOT NULL UNIQUE,
                    org         TEXT NOT NULL,
                    tier        TEXT NOT NULL DEFAULT 'free',
                    created_at  TEXT NOT NULL,
                    active      INTEGER NOT NULL DEFAULT 1
                )
            """)

    def create(self, email: str, org: str, tier: str = "free") -> APIKey:
        """Generate a new key and persist it. Raises ValueError if email already registered."""
        key = generate_key()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO api_keys (key, email, org, tier, created_at, active) "
                    "VALUES (?, ?, ?, ?, ?, 1)",
                    (key, email.lower(), org, tier, now),
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"{email} is already registered")
        return APIKey(key=key, email=email.lower(), org=org, tier=tier)

    def get_by_key(self, key: str) -> Optional[APIKey]:
        """Look up an API key. Returns None if not found or inactive."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key = ? AND active = 1", (key,)
            ).fetchone()
        if row is None:
            return None
        return APIKey(
            key=row["key"],
            email=row["email"],
            org=row["org"],
            tier=row["tier"],
            created_at=datetime.fromisoformat(row["created_at"]),
            active=bool(row["active"]),
        )

    def get_by_email(self, email: str) -> Optional[APIKey]:
        """Look up a registration by email."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE email = ?", (email.lower(),)
            ).fetchone()
        if row is None:
            return None
        return APIKey(
            key=row["key"],
            email=row["email"],
            org=row["org"],
            tier=row["tier"],
            created_at=datetime.fromisoformat(row["created_at"]),
            active=bool(row["active"]),
        )

    def deactivate(self, key: str) -> bool:
        """Deactivate an API key. Returns True if the key existed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key = ?", (key,)
            )
        return cursor.rowcount > 0
