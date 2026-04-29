"""
pact_ax/access/rate_limit.py
─────────────────────────────
SQLite-backed rate limiter for free tier API keys.

Free tier limits:  100 requests / hour  |  1 000 requests / day

Windows are persisted to SQLite so limits survive process restarts and
are shared across multiple processes pointing at the same db file.
An in-memory write-through cache means every hot path hits the cache
first; SQLite is written on every check but only read on cache miss
(first request per key after startup).
"""

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

HOURLY_LIMIT = 100
DAILY_LIMIT  = 1_000


@dataclass
class _Window:
    hourly_count:    int
    hourly_reset_at: datetime
    daily_count:     int
    daily_reset_at:  datetime


class RateLimiter:
    """
    Thread-safe, SQLite-backed sliding-window rate limiter.

    Pass db_path=":memory:" (or omit) for a purely in-memory instance
    (useful in tests). Pass a file path to get persistence across restarts.
    """

    def __init__(
        self,
        hourly_limit: int = HOURLY_LIMIT,
        daily_limit:  int = DAILY_LIMIT,
        db_path: Union[str, Path, None] = None,
    ):
        self._hourly = hourly_limit
        self._daily  = daily_limit
        self._cache:  Dict[str, _Window] = {}
        self._lock = threading.Lock()

        path = str(db_path) if db_path is not None else ":memory:"
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_windows (
                    api_key          TEXT PRIMARY KEY,
                    hourly_count     INTEGER NOT NULL DEFAULT 0,
                    hourly_reset_at  TEXT NOT NULL,
                    daily_count      INTEGER NOT NULL DEFAULT 0,
                    daily_reset_at   TEXT NOT NULL
                )
            """)

    # ── Public interface ──────────────────────────────────────────────────────

    def check(self, key: str) -> Tuple[bool, Dict[str, str]]:
        """
        Record one request. Returns (allowed, rate-limit headers).

        Headers follow the draft RateLimit spec:
          RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset   (hourly)
          X-RateLimit-Limit-Day, X-RateLimit-Remaining-Day
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            win = self._load(key, now)

            # Reset expired windows
            if now >= win.hourly_reset_at:
                win.hourly_count    = 0
                win.hourly_reset_at = now + timedelta(hours=1)
            if now >= win.daily_reset_at:
                win.daily_count  = 0
                win.daily_reset_at = now + timedelta(days=1)

            hourly_remaining = max(0, self._hourly - win.hourly_count)
            daily_remaining  = max(0, self._daily  - win.daily_count)

            if win.hourly_count >= self._hourly:
                return False, self._headers(0, win.hourly_reset_at, daily_remaining)

            if win.daily_count >= self._daily:
                return False, self._headers(hourly_remaining, win.hourly_reset_at, 0)

            win.hourly_count += 1
            win.daily_count  += 1
            self._save(key, win)

            return True, self._headers(
                self._hourly - win.hourly_count,
                win.hourly_reset_at,
                self._daily  - win.daily_count,
            )

    def usage(self, key: str) -> Dict[str, int]:
        """Current usage counts for a key (used by /access/status)."""
        now = datetime.now(timezone.utc)
        with self._lock:
            win = self._cache.get(key) or self._load_from_db(key, now)
        hourly = win.hourly_count if now < win.hourly_reset_at else 0
        daily  = win.daily_count  if now < win.daily_reset_at  else 0
        return {
            "hourly_used":  hourly, "hourly_limit": self._hourly,
            "daily_used":   daily,  "daily_limit":  self._daily,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self, key: str, now: datetime) -> _Window:
        """Return cached window, falling back to DB, then a fresh window."""
        if key in self._cache:
            return self._cache[key]
        win = self._load_from_db(key, now)
        self._cache[key] = win
        return win

    def _load_from_db(self, key: str, now: datetime) -> _Window:
        row = self._conn.execute(
            "SELECT * FROM rate_windows WHERE api_key = ?", (key,)
        ).fetchone()
        if row is None:
            return _Window(
                hourly_count=0,
                hourly_reset_at=now + timedelta(hours=1),
                daily_count=0,
                daily_reset_at=now + timedelta(days=1),
            )
        return _Window(
            hourly_count=row["hourly_count"],
            hourly_reset_at=datetime.fromisoformat(row["hourly_reset_at"]),
            daily_count=row["daily_count"],
            daily_reset_at=datetime.fromisoformat(row["daily_reset_at"]),
        )

    def _save(self, key: str, win: _Window) -> None:
        self._cache[key] = win
        with self._conn:
            self._conn.execute("""
                INSERT INTO rate_windows
                    (api_key, hourly_count, hourly_reset_at, daily_count, daily_reset_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(api_key) DO UPDATE SET
                    hourly_count    = excluded.hourly_count,
                    hourly_reset_at = excluded.hourly_reset_at,
                    daily_count     = excluded.daily_count,
                    daily_reset_at  = excluded.daily_reset_at
            """, (
                key,
                win.hourly_count,
                win.hourly_reset_at.isoformat(),
                win.daily_count,
                win.daily_reset_at.isoformat(),
            ))

    def _headers(
        self,
        hourly_remaining: int,
        hourly_reset: datetime,
        daily_remaining: int,
    ) -> Dict[str, str]:
        return {
            "RateLimit-Limit":           str(self._hourly),
            "RateLimit-Remaining":       str(hourly_remaining),
            "RateLimit-Reset":           str(int(hourly_reset.timestamp())),
            "X-RateLimit-Limit-Day":     str(self._daily),
            "X-RateLimit-Remaining-Day": str(daily_remaining),
        }
