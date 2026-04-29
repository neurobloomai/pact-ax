"""
pact_ax/access/rate_limit.py
─────────────────────────────
In-memory sliding-window rate limiter for free tier API keys.

Free tier limits:
  100 requests / hour
  1 000 requests / day

No Redis dependency. State lives in the process — fine for a single-instance
deployment. When horizontal scaling is needed, swap the in-memory counters
for a Redis backend without changing the RateLimiter interface.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Dict, Optional, Tuple


HOURLY_LIMIT = 100
DAILY_LIMIT  = 1_000


@dataclass
class _Window:
    hourly_count:    int = 0
    hourly_reset_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    daily_count:     int = 0
    daily_reset_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=1)
    )


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Call check(key) before forwarding a request.
    Returns (allowed, headers) where headers should be added to the response.
    """

    def __init__(
        self,
        hourly_limit: int = HOURLY_LIMIT,
        daily_limit:  int = DAILY_LIMIT,
    ):
        self._hourly = hourly_limit
        self._daily  = daily_limit
        self._windows: Dict[str, _Window] = {}
        self._lock = Lock()

    def check(self, key: str) -> Tuple[bool, Dict[str, str]]:
        """
        Record one request for key and return (allowed, rate-limit headers).

        Headers follow the draft RateLimit spec (hourly window primary):
          RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset
          X-RateLimit-Limit-Day, X-RateLimit-Remaining-Day
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            win = self._windows.setdefault(key, _Window())

            if now >= win.hourly_reset_at:
                win.hourly_count    = 0
                win.hourly_reset_at = now + timedelta(hours=1)

            if now >= win.daily_reset_at:
                win.daily_count    = 0
                win.daily_reset_at = now + timedelta(days=1)

            hourly_remaining = max(0, self._hourly - win.hourly_count)
            daily_remaining  = max(0, self._daily  - win.daily_count)

            if win.hourly_count >= self._hourly:
                headers = self._headers(
                    hourly_remaining=0,
                    hourly_reset=win.hourly_reset_at,
                    daily_remaining=daily_remaining,
                )
                return False, headers

            if win.daily_count >= self._daily:
                headers = self._headers(
                    hourly_remaining=hourly_remaining,
                    hourly_reset=win.hourly_reset_at,
                    daily_remaining=0,
                )
                return False, headers

            win.hourly_count += 1
            win.daily_count  += 1

            headers = self._headers(
                hourly_remaining=self._hourly - win.hourly_count,
                hourly_reset=win.hourly_reset_at,
                daily_remaining=self._daily - win.daily_count,
            )
            return True, headers

    def usage(self, key: str) -> Dict[str, int]:
        """Return current usage counts for a key (for the /status endpoint)."""
        now = datetime.now(timezone.utc)
        with self._lock:
            win = self._windows.get(key)
            if win is None:
                return {
                    "hourly_used": 0, "hourly_limit": self._hourly,
                    "daily_used":  0, "daily_limit":  self._daily,
                }
            hourly = win.hourly_count if now < win.hourly_reset_at else 0
            daily  = win.daily_count  if now < win.daily_reset_at  else 0
            return {
                "hourly_used":  hourly, "hourly_limit": self._hourly,
                "daily_used":   daily,  "daily_limit":  self._daily,
            }

    def _headers(
        self,
        hourly_remaining: int,
        hourly_reset: datetime,
        daily_remaining: int,
    ) -> Dict[str, str]:
        reset_ts = str(int(hourly_reset.timestamp()))
        return {
            "RateLimit-Limit":          str(self._hourly),
            "RateLimit-Remaining":      str(hourly_remaining),
            "RateLimit-Reset":          reset_ts,
            "X-RateLimit-Limit-Day":    str(self._daily),
            "X-RateLimit-Remaining-Day": str(daily_remaining),
        }
