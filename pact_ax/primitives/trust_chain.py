"""
TrustChain: Relational coherence across agent hops.

TrustManager tracks point-in-time trust between pairs. TrustChain answers
the next question: when Agent A → Agent B → Agent C, is the full chain
still coherent — not just at initialization, but across every hop, every
handoff, every state change?

Key concepts
------------
ChainHop
    A single directed trust edge (from_agent → to_agent) with the score
    recorded at chain creation time (baseline) and the score last seen on
    re-verification (current).

TrustChain
    An ordered sequence of hops. Carries three derived signals:
      chain_trust  — geometric mean of hop scores. Penalises depth and
                     weak links proportionally; a chain of [0.9, 0.9, 0.9]
                     scores 0.9; a chain of [0.9, 0.5, 0.9] scores 0.72.
      coherence    — 1.0 − normalised std-dev of hop scores. 1.0 = all
                     hops equal strength; lower = uneven, harder to trust.
      weakest_hop  — the hop that limits chain trust (the bottleneck).

ChainState
    ACTIVE    — chain_trust ≥ 0.7 and coherence ≥ 0.6
    DEGRADED  — chain_trust in [0.4, 0.7) or coherence < 0.6
    BROKEN    — chain_trust < 0.4
    COMPLETED — explicitly closed

ChainVerification
    Result of re-running trust resolution against a recorded chain.
    Reports per-hop drift (current − baseline) and whether chain state
    has changed.

Usage
-----
    from pact_ax.primitives import TrustChain, TrustChainManager

    def my_resolver(from_id: str, to_id: str) -> float:
        return trust_managers[from_id].get_trust(to_id)

    mgr = TrustChainManager(trust_resolver=my_resolver)

    chain  = mgr.record(["agent-a", "agent-b", "agent-c"])
    verify = mgr.verify(chain.chain_id)
    score  = mgr.score(["agent-x", "agent-y", "agent-z"])
"""

from __future__ import annotations

import math
import sqlite3
import statistics
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union


# ── State ─────────────────────────────────────────────────────────────────────

class ChainState(str, Enum):
    ACTIVE    = "active"
    DEGRADED  = "degraded"
    BROKEN    = "broken"
    COMPLETED = "completed"


# ── Thresholds ────────────────────────────────────────────────────────────────

_ACTIVE_TRUST_THRESHOLD    = 0.70
_DEGRADED_TRUST_THRESHOLD  = 0.40
_ACTIVE_COHERENCE_THRESHOLD = 0.60


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ChainHop:
    """A single directed trust edge within a chain."""
    from_agent:      str
    to_agent:        str
    baseline_score:  float          # trust score at chain creation
    current_score:   float          # trust score at last verification
    recorded_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified:   Optional[datetime] = None

    @property
    def drift(self) -> float:
        """Signed drift from baseline (positive = trust grew, negative = decayed)."""
        return round(self.current_score - self.baseline_score, 4)

    @property
    def label(self) -> str:
        return f"{self.from_agent}→{self.to_agent}"


@dataclass
class TrustChain:
    """Recorded trust chain with derived coherence signals."""
    chain_id:       str
    hops:           List[ChainHop]
    chain_trust:    float           # geometric mean of hop scores
    coherence:      float           # 1.0 − normalised std-dev
    state:          ChainState
    created_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified:  Optional[datetime] = None

    @property
    def depth(self) -> int:
        return len(self.hops)

    @property
    def agents(self) -> List[str]:
        if not self.hops:
            return []
        path = [self.hops[0].from_agent]
        for hop in self.hops:
            path.append(hop.to_agent)
        return path

    @property
    def weakest_hop(self) -> Optional[ChainHop]:
        if not self.hops:
            return None
        return min(self.hops, key=lambda h: h.current_score)


@dataclass
class ChainScore:
    """Score of an agent path without recording it."""
    agents:         List[str]
    hop_scores:     List[float]
    chain_trust:    float
    coherence:      float
    state:          ChainState
    weakest_index:  int             # index into hop_scores of the weakest hop
    weakest_pair:   tuple           # (from_agent, to_agent) of the weakest hop


@dataclass
class HopDrift:
    """Drift report for a single hop."""
    from_agent:     str
    to_agent:       str
    baseline:       float
    current:        float
    drift:          float           # current − baseline
    drifted:        bool            # True if |drift| > threshold


@dataclass
class ChainVerification:
    """Result of re-verifying a chain against current trust scores."""
    chain_id:       str
    previous_state: ChainState
    current_state:  ChainState
    state_changed:  bool
    hop_drift:      List[HopDrift]
    chain_trust_was: float
    chain_trust_now: float
    coherence_was:  float
    coherence_now:  float
    verified_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Scoring ───────────────────────────────────────────────────────────────────

def _geometric_mean(scores: List[float]) -> float:
    """Geometric mean — penalises depth and weak links proportionally."""
    if not scores:
        return 0.0
    if any(s <= 0 for s in scores):
        return 0.0
    return math.exp(sum(math.log(s) for s in scores) / len(scores))


def _coherence(scores: List[float]) -> float:
    """
    1.0 − normalised std-dev. 1.0 = all hops equal strength.
    Returns 1.0 for single-hop chains (no variance possible).
    """
    if len(scores) < 2:
        return 1.0
    std = statistics.stdev(scores)
    # Normalise: max possible std-dev for scores in [0,1] is 0.5
    return round(max(0.0, 1.0 - (std / 0.5)), 4)


def _chain_state(chain_trust: float, coherence: float) -> ChainState:
    if chain_trust < _DEGRADED_TRUST_THRESHOLD:
        return ChainState.BROKEN
    if chain_trust < _ACTIVE_TRUST_THRESHOLD or coherence < _ACTIVE_COHERENCE_THRESHOLD:
        return ChainState.DEGRADED
    return ChainState.ACTIVE


# ── Manager ───────────────────────────────────────────────────────────────────

class TrustChainManager:
    """
    Creates, records, verifies, and tracks trust chains.

    Parameters
    ----------
    trust_resolver
        Callable(from_agent_id, to_agent_id) → float in [0.0, 1.0].
        Wires the manager to whatever trust backend is in use.
    drift_threshold
        Minimum absolute change in a hop score before it's flagged as drifted.
        Default: 0.05 (5 percentage points).
    """

    def __init__(
        self,
        trust_resolver: Callable[[str, str], float],
        drift_threshold: float = 0.05,
    ):
        self._resolve  = trust_resolver
        self._drift_th = drift_threshold
        self._chains:  Dict[str, TrustChain] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, agent_ids: List[str]) -> ChainScore:
        """
        Score a chain of agents without recording it.

        agent_ids must have at least 2 elements (one hop).
        """
        self._validate_path(agent_ids)
        hop_scores = [
            self._resolve(agent_ids[i], agent_ids[i + 1])
            for i in range(len(agent_ids) - 1)
        ]
        ct  = _geometric_mean(hop_scores)
        coh = _coherence(hop_scores)
        wi  = hop_scores.index(min(hop_scores))
        return ChainScore(
            agents        = agent_ids,
            hop_scores    = hop_scores,
            chain_trust   = round(ct, 4),
            coherence     = coh,
            state         = _chain_state(ct, coh),
            weakest_index = wi,
            weakest_pair  = (agent_ids[wi], agent_ids[wi + 1]),
        )

    def record(
        self,
        agent_ids: List[str],
        chain_id:  Optional[str] = None,
    ) -> TrustChain:
        """
        Record a trust chain and store it for future verification.

        Returns the recorded TrustChain with its assigned chain_id.
        """
        self._validate_path(agent_ids)
        now        = datetime.now(timezone.utc)
        chain_id   = chain_id or str(uuid.uuid4())
        hop_scores = []
        hops       = []

        for i in range(len(agent_ids) - 1):
            fa, ta = agent_ids[i], agent_ids[i + 1]
            score  = self._resolve(fa, ta)
            hop_scores.append(score)
            hops.append(ChainHop(
                from_agent     = fa,
                to_agent       = ta,
                baseline_score = score,
                current_score  = score,
                recorded_at    = now,
                last_verified  = now,
            ))

        ct  = _geometric_mean(hop_scores)
        coh = _coherence(hop_scores)
        chain = TrustChain(
            chain_id      = chain_id,
            hops          = hops,
            chain_trust   = round(ct, 4),
            coherence     = coh,
            state         = _chain_state(ct, coh),
            created_at    = now,
            last_verified = now,
        )
        self._chains[chain_id] = chain
        return chain

    def verify(self, chain_id: str) -> ChainVerification:
        """
        Re-resolve current trust for every hop and compare to baseline.

        Updates the chain's state and current_score on each hop in-place.
        Returns a ChainVerification with per-hop drift and state delta.
        """
        chain = self._get_chain(chain_id)
        now   = datetime.now(timezone.utc)

        drift_reports: List[HopDrift] = []
        current_scores: List[float]   = []

        for hop in chain.hops:
            current = self._resolve(hop.from_agent, hop.to_agent)
            drift   = round(current - hop.baseline_score, 4)
            drift_reports.append(HopDrift(
                from_agent = hop.from_agent,
                to_agent   = hop.to_agent,
                baseline   = hop.baseline_score,
                current    = current,
                drift      = drift,
                drifted    = abs(drift) >= self._drift_th,
            ))
            hop.current_score  = current
            hop.last_verified  = now
            current_scores.append(current)

        prev_state     = chain.state
        prev_trust     = chain.chain_trust
        prev_coherence = chain.coherence

        new_ct  = _geometric_mean(current_scores)
        new_coh = _coherence(current_scores)
        new_state = _chain_state(new_ct, new_coh)

        chain.chain_trust   = round(new_ct, 4)
        chain.coherence     = new_coh
        chain.state         = new_state
        chain.last_verified = now

        return ChainVerification(
            chain_id         = chain_id,
            previous_state   = prev_state,
            current_state    = new_state,
            state_changed    = prev_state != new_state,
            hop_drift        = drift_reports,
            chain_trust_was  = prev_trust,
            chain_trust_now  = round(new_ct, 4),
            coherence_was    = prev_coherence,
            coherence_now    = new_coh,
            verified_at      = now,
        )

    def complete(self, chain_id: str) -> TrustChain:
        """Mark a chain as completed (explicitly closed)."""
        chain = self._get_chain(chain_id)
        chain.state = ChainState.COMPLETED
        return chain

    def get(self, chain_id: str) -> TrustChain:
        return self._get_chain(chain_id)

    def list_chains(self, agent_id: Optional[str] = None) -> List[TrustChain]:
        """Return all chains, optionally filtered to those involving agent_id."""
        chains = list(self._chains.values())
        if agent_id:
            chains = [c for c in chains if agent_id in c.agents]
        return sorted(chains, key=lambda c: c.created_at, reverse=True)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, db_path: Union[str, Path] = "trust_chains.db") -> None:
        """Persist all chains to SQLite."""
        con = sqlite3.connect(str(db_path))
        with con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS trust_chains (
                    chain_id      TEXT PRIMARY KEY,
                    chain_trust   REAL,
                    coherence     REAL,
                    state         TEXT,
                    created_at    TEXT,
                    last_verified TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS chain_hops (
                    chain_id       TEXT,
                    hop_index      INTEGER,
                    from_agent     TEXT,
                    to_agent       TEXT,
                    baseline_score REAL,
                    current_score  REAL,
                    recorded_at    TEXT,
                    last_verified  TEXT,
                    PRIMARY KEY (chain_id, hop_index)
                )
            """)
            for chain in self._chains.values():
                con.execute("""
                    INSERT OR REPLACE INTO trust_chains
                    VALUES (?,?,?,?,?,?)
                """, (
                    chain.chain_id,
                    chain.chain_trust,
                    chain.coherence,
                    chain.state.value,
                    chain.created_at.isoformat(),
                    chain.last_verified.isoformat() if chain.last_verified else None,
                ))
                con.execute("DELETE FROM chain_hops WHERE chain_id=?", (chain.chain_id,))
                for i, hop in enumerate(chain.hops):
                    con.execute("""
                        INSERT INTO chain_hops VALUES (?,?,?,?,?,?,?,?)
                    """, (
                        chain.chain_id, i,
                        hop.from_agent, hop.to_agent,
                        hop.baseline_score, hop.current_score,
                        hop.recorded_at.isoformat(),
                        hop.last_verified.isoformat() if hop.last_verified else None,
                    ))
        con.close()

    @classmethod
    def load(
        cls,
        db_path: Union[str, Path],
        trust_resolver: Callable[[str, str], float],
        drift_threshold: float = 0.05,
    ) -> "TrustChainManager":
        """Restore a TrustChainManager from SQLite."""
        mgr = cls(trust_resolver=trust_resolver, drift_threshold=drift_threshold)
        path = Path(str(db_path))
        if not path.exists() and str(db_path) != ":memory:":
            return mgr

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        chains_rows = con.execute("SELECT * FROM trust_chains").fetchall()
        for row in chains_rows:
            hop_rows = con.execute(
                "SELECT * FROM chain_hops WHERE chain_id=? ORDER BY hop_index",
                (row["chain_id"],),
            ).fetchall()
            hops = [
                ChainHop(
                    from_agent     = h["from_agent"],
                    to_agent       = h["to_agent"],
                    baseline_score = h["baseline_score"],
                    current_score  = h["current_score"],
                    recorded_at    = datetime.fromisoformat(h["recorded_at"]),
                    last_verified  = datetime.fromisoformat(h["last_verified"]) if h["last_verified"] else None,
                )
                for h in hop_rows
            ]
            chain = TrustChain(
                chain_id      = row["chain_id"],
                hops          = hops,
                chain_trust   = row["chain_trust"],
                coherence     = row["coherence"],
                state         = ChainState(row["state"]),
                created_at    = datetime.fromisoformat(row["created_at"]),
                last_verified = datetime.fromisoformat(row["last_verified"]) if row["last_verified"] else None,
            )
            mgr._chains[chain.chain_id] = chain
        con.close()
        return mgr

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_chain(self, chain_id: str) -> TrustChain:
        chain = self._chains.get(chain_id)
        if chain is None:
            raise KeyError(f"No chain found with id '{chain_id}'")
        return chain

    @staticmethod
    def _validate_path(agent_ids: List[str]) -> None:
        if len(agent_ids) < 2:
            raise ValueError("A trust chain requires at least 2 agents (one hop).")
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Trust chain must not contain duplicate agents (cycles not supported).")
