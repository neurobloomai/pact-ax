"""
pact_ax/access/keys.py
───────────────────────
API key generation, email verification, and SQLite-backed storage.

Registration flow:
  1. create_verification(email, org) → token   (pending, no key yet)
  2. confirm_verification(token)     → APIKey  (key issued only after proof)

Key format: pax_<32 hex chars> — identifiable prefix, 128 bits of entropy.
Token format: 64 hex chars — single-use, expires in 24 hours.
"""

import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

_KEY_PREFIX  = "pax_"
_DB_DEFAULT  = "access.db"
_TOKEN_TTL_H = 24  # hours before a verification token expires


@dataclass
class APIKey:
    key:        str
    email:      str
    org:        str
    tier:       str = "free"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active:     bool = True


def generate_key() -> str:
    return _KEY_PREFIX + secrets.token_hex(16)


def _generate_token() -> str:
    return secrets.token_hex(32)   # 64-char hex string


class KeyStore:
    """SQLite-backed store for API keys and pending verifications."""

    def __init__(self, db_path: Union[str, Path] = _DB_DEFAULT):
        self._path = str(db_path)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_verifications (
                    token       TEXT PRIMARY KEY,
                    email       TEXT NOT NULL UNIQUE,
                    org         TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT NOT NULL
                )
            """)

    # ── Verification flow ─────────────────────────────────────────────────────

    def create_verification(self, email: str, org: str) -> str:
        """
        Stage a registration and return a single-use verification token.

        The caller is responsible for delivering the token to the user
        (email, log, etc.). No API key is issued until confirm_verification().

        Raises ValueError if the email is already registered or has a
        pending (unexpired) verification.
        """
        email = email.lower()
        now   = datetime.now(timezone.utc)

        if self.get_by_email(email) is not None:
            raise ValueError(f"{email} is already registered")

        # Clean up any expired token for this email first
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_verifications WHERE email = ? AND expires_at < ?",
                (email, now.isoformat()),
            )
            # Check for still-valid pending token
            row = conn.execute(
                "SELECT token FROM pending_verifications WHERE email = ?", (email,)
            ).fetchone()
            if row:
                raise ValueError(
                    f"A verification email was already sent to {email}. "
                    "Check your inbox or wait for it to expire."
                )

        token      = _generate_token()
        expires_at = now + timedelta(hours=_TOKEN_TTL_H)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pending_verifications (token, email, org, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token, email, org, now.isoformat(), expires_at.isoformat()),
            )

        return token

    def confirm_verification(self, token: str) -> APIKey:
        """
        Validate token, issue an API key, and clean up the pending record.

        Raises ValueError if the token is invalid or expired.
        """
        now = datetime.now(timezone.utc)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT email, org, expires_at FROM pending_verifications WHERE token = ?",
                (token,),
            ).fetchone()

        if row is None:
            raise ValueError("Invalid verification token")

        if datetime.fromisoformat(row["expires_at"]) < now:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM pending_verifications WHERE token = ?", (token,)
                )
            raise ValueError("Verification token has expired")

        email = row["email"]
        org   = row["org"]

        # Issue the key
        api_key = self.create(email=email, org=org)

        # Consume the token
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_verifications WHERE token = ?", (token,)
            )

        return api_key

    # ── Direct key operations ─────────────────────────────────────────────────

    def create(self, email: str, org: str, tier: str = "free") -> APIKey:
        """Issue a key directly (used by confirm_verification and tests)."""
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
        """Return an active key record, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key = ? AND active = 1", (key,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_key(row)

    def get_by_email(self, email: str) -> Optional[APIKey]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE email = ?", (email.lower(),)
            ).fetchone()
        return self._row_to_key(row) if row else None

    def deactivate(self, key: str) -> bool:
        """Revoke a key. Returns True if it existed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key = ?", (key,)
            )
        return cursor.rowcount > 0

    def _row_to_key(self, row: sqlite3.Row) -> APIKey:
        return APIKey(
            key=row["key"],
            email=row["email"],
            org=row["org"],
            tier=row["tier"],
            created_at=datetime.fromisoformat(row["created_at"]),
            active=bool(row["active"]),
        )
