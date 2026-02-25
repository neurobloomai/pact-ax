"""
PACT-AX Proxy — Session Integrity Layer for MCP

Sits between Cursor (MCP client) and GitHub (MCP server),
maintaining relational state through two layers:

    StoryKeeper  — behavioral coherence, pattern drift detection (AX expression layer)
    RLP-0        — relational primitive gating (rupture detection, gate authority)

Architecture:
    Cursor stdin/stdout
          ↕  MCP JSON-RPC (newline-delimited)
    PACTAXProxy
          ├── StoryKeeper.record_event()   → coherence score, drift alert
          └── RLPBridge.sync()             → rupture_risk, gate open/closed
          ↕  subprocess stdin/stdout
    Docker: ghcr.io/github/github-mcp-server
          ↕
    GitHub API

Environment variables:
    GITHUB_PERSONAL_ACCESS_TOKEN   required — passed to upstream Docker container
    PACT_UPSTREAM_MODE             docker (default) | npx
    PACT_DRIFT_THRESHOLD           float, default 0.3  (StoryKeeper coherence floor)
    PACT_RUPTURE_THRESHOLD         float, default 0.6  (RLP-0 rupture gate trigger)
    PACT_BLOCK_ON_VIOLATION        true | false (default false) — legacy manual block
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import sys
from dataclasses import dataclass, field
from typing import Optional
import logging

from .story_keeper import StoryKeeper, TrustTrajectory
from .rlp_bridge import RLPBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PACT-AX] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("pact-ax")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    upstream_mode: str = field(
        default_factory=lambda: os.environ.get("PACT_UPSTREAM_MODE", "docker")
    )
    drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PACT_DRIFT_THRESHOLD", "0.3"))
    )
    rupture_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PACT_RUPTURE_THRESHOLD", "0.6"))
    )
    trust_violation_threshold: float = 0.2
    block_on_violation: bool = field(
        default_factory=lambda: os.environ.get(
            "PACT_BLOCK_ON_VIOLATION", "false"
        ).lower() == "true"
    )

    @property
    def upstream_command(self) -> str:
        return "npx" if self.upstream_mode == "npx" else "docker"

    @property
    def upstream_args(self) -> list:
        if self.upstream_mode == "npx":
            # NOTE: @modelcontextprotocol/server-github deprecated April 2025.
            # Prefer docker mode.
            return ["-y", "@modelcontextprotocol/server-github"]
        return [
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ]

    @property
    def upstream_env(self) -> dict:
        env = dict(os.environ)
        pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if not pat:
            logger.warning(
                "GITHUB_PERSONAL_ACCESS_TOKEN is not set — "
                "upstream will likely reject requests."
            )
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = pat
        return env


# ---------------------------------------------------------------------------
# MCP message wrapper
# ---------------------------------------------------------------------------

class MCPMessage:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.data = json.loads(raw.decode("utf-8"))

    @property
    def id(self) -> Optional[str]:
        return self.data.get("id")

    @property
    def method(self) -> Optional[str]:
        return self.data.get("method")

    @property
    def params(self) -> dict:
        return self.data.get("params", {})

    @property
    def is_request(self) -> bool:
        return "method" in self.data and "id" in self.data

    @property
    def is_notification(self) -> bool:
        return "method" in self.data and "id" not in self.data

    @property
    def is_response(self) -> bool:
        return "result" in self.data or "error" in self.data

    def to_bytes(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

class PACTAXProxy:
    """
    Bidirectional stdio MCP proxy with two-layer session integrity.

    Layer 1 — StoryKeeper (AX expression layer):
        Tracks resource access patterns, coherence scores, trust trajectory.
        Emits drift alerts when coherence drops below drift_threshold.

    Layer 2 — RLP-0 (relational primitive layer):
        Receives four primitives translated from StoryKeeper state.
        Computes rupture_risk; gates interaction when threshold exceeded.
        Gate authority is final — overrides manual block_on_violation.
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.story_keeper = StoryKeeper()
        self.story_keeper.drift_threshold = config.drift_threshold

        self.rlp_bridge: Optional[RLPBridge] = None
        self.session_id: Optional[str] = None
        self.upstream_process: Optional[asyncio.subprocess.Process] = None

        self.messages_processed = 0
        self.drift_alerts = 0
        self.rupture_events = 0
        self.blocked_requests = 0

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self):
        logger.info(
            f"PACT-AX Proxy starting  "
            f"upstream={self.config.upstream_mode}  "
            f"drift_threshold={self.config.drift_threshold}  "
            f"rupture_threshold={self.config.rupture_threshold}  "
            f"block_on_violation={self.config.block_on_violation}"
        )

        self.session_id = f"pact-{uuid.uuid4().hex[:8]}"
        self.story_keeper.create_session(
            session_id=self.session_id,
            client_identity="cursor:local",
            server_target="github:mcp",
        )
        self.rlp_bridge = RLPBridge(
            rupture_threshold=self.config.rupture_threshold
        )
        logger.info(f"Session: {self.session_id}")

        await self._start_upstream()
        await self._run_proxy()

    async def _start_upstream(self):
        logger.info(
            f"Launching upstream: {self.config.upstream_command} "
            f"{' '.join(self.config.upstream_args)}"
        )
        self.upstream_process = await asyncio.create_subprocess_exec(
            self.config.upstream_command,
            *self.config.upstream_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.config.upstream_env,
        )
        asyncio.create_task(self._drain_upstream_stderr())

    async def _drain_upstream_stderr(self):
        while True:
            line = await self.upstream_process.stderr.readline()
            if not line:
                break
            logger.debug(f"[upstream] {line.decode().strip()}")

    # ------------------------------------------------------------------
    # Proxy loop
    # ------------------------------------------------------------------

    async def _run_proxy(self):
        c2u = asyncio.create_task(self._client_to_upstream())
        u2c = asyncio.create_task(self._upstream_to_client())

        done, pending = await asyncio.wait(
            [c2u, u2c], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()

        if self.upstream_process:
            try:
                self.upstream_process.terminate()
            except ProcessLookupError:
                pass

        self._log_summary()

    async def _client_to_upstream(self):
        """Cursor → PACT-AX evaluation → GitHub MCP."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            try:
                msg = MCPMessage(stripped)
                self.messages_processed += 1
                should_forward, alert = self._evaluate(msg)

                if alert:
                    logger.warning(alert)

                if should_forward:
                    self.upstream_process.stdin.write(stripped + b"\n")
                    await self.upstream_process.stdin.drain()
                else:
                    self.blocked_requests += 1
                    await self._send_blocked(msg)

            except (json.JSONDecodeError, UnicodeDecodeError):
                self.upstream_process.stdin.write(line)
                await self.upstream_process.stdin.drain()

    async def _upstream_to_client(self):
        """GitHub MCP → Cursor (pass-through; annotate in future)."""
        while True:
            line = await self.upstream_process.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = MCPMessage(stripped)
                sys.stdout.buffer.write(msg.to_bytes() + b"\n")
                sys.stdout.buffer.flush()
            except (json.JSONDecodeError, UnicodeDecodeError):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # Two-layer evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, msg: MCPMessage) -> tuple:
        """
        Run the message through both integrity layers.

        Returns:
            (should_forward: bool, alert: Optional[str])
        """
        if not msg.is_request:
            return True, None

        # ── Layer 1: StoryKeeper ──────────────────────────────────
        event, drift_alert = self.story_keeper.record_event(
            session_id=self.session_id,
            method=msg.method,
            params=msg.params,
        )
        session = self.story_keeper.sessions[self.session_id]

        if drift_alert:
            self.drift_alerts += 1

        # ── Layer 2: RLP-0 ───────────────────────────────────────
        rupture_risk, is_gated = self.rlp_bridge.sync(session)
        primitives = self.rlp_bridge.primitive_snapshot()

        logger.info(
            f"[{msg.method}] "
            f"pattern={event.resource_pattern} "
            f"coherence={event.coherence_score:.2f} "
            f"trust={session.trust_level:.2f} "
            f"traj={session.trajectory.value} "
            f"rlp.rupture_risk={rupture_risk:.2f} "
            f"rlp.gated={is_gated}"
        )

        # ── Gate decision ─────────────────────────────────────────
        # RLP-0 gate has final authority
        if is_gated:
            self.rupture_events += 1
            sig = self.rlp_bridge.last_rupture_signal
            alert = (
                f"[RLP-0 RUPTURE GATE CLOSED]\n"
                f"Session         : {self.session_id}\n"
                f"Rupture risk    : {rupture_risk:.2f} "
                f"(threshold: {self.config.rupture_threshold})\n"
                f"Signal          : {sig}\n"
                f"RLP-0 primitives: {primitives}\n"
                f"Gate status     : CLOSED — interaction blocked until repair acknowledged.\n"
                f"---\n"
                f"Policy sees: valid token, permitted scope ✓\n"
                f"RLP-0 sees : relational rupture — trust={primitives['trust']:.2f} "
                f"intent={primitives['intent']:.2f} "
                f"narrative={primitives['narrative']:.2f}"
            )
            return False, alert

        # Legacy manual block (StoryKeeper drift without RLP-0 rupture)
        if drift_alert and self.config.block_on_violation:
            if session.trust_level < self.config.trust_violation_threshold:
                return False, drift_alert

        return True, drift_alert

    async def _send_blocked(self, original: MCPMessage):
        rlp_status = self.rlp_bridge.status() if self.rlp_bridge else {}
        sk_summary = self.story_keeper.get_session_summary(self.session_id)

        response = {
            "jsonrpc": "2.0",
            "id": original.id,
            "error": {
                "code": -32001,
                "message": "PACT-AX: request blocked — relational integrity gate closed",
                "data": {
                    "session_id":     self.session_id,
                    "reason":         "rlp0_rupture_gate",
                    "rupture_risk":   round(self.rlp_bridge.rupture_risk, 3),
                    "rlp0_status":    rlp_status,
                    "session_summary": sk_summary,
                    "repair_hint": (
                        "Call acknowledge_repair() on the RLP-0 bridge "
                        "after the expression layer has addressed the rupture."
                    ),
                },
            },
        }
        sys.stdout.buffer.write(json.dumps(response).encode() + b"\n")
        sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _log_summary(self):
        sk  = self.story_keeper.get_session_summary(self.session_id)
        rlp = self.rlp_bridge.status() if self.rlp_bridge else {}

        logger.info("=" * 62)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Session ID        : {self.session_id}")
        logger.info(f"  Messages processed: {self.messages_processed}")
        logger.info(f"  Drift alerts      : {self.drift_alerts}")
        logger.info(f"  RLP-0 ruptures    : {self.rupture_events}")
        logger.info(f"  Blocked requests  : {self.blocked_requests}")
        logger.info(f"  SK trust level    : {sk.get('trust_level', '?'):.2f}")
        logger.info(f"  SK trajectory     : {sk.get('trajectory', '?')}")
        logger.info(f"  SK drift risk     : {sk.get('drift_risk', '?')}")
        logger.info(f"  RLP-0 rupture_risk: {rlp.get('state', {}).get('rupture_risk', '?')}")
        logger.info(f"  RLP-0 gated now   : {rlp.get('is_gated', '?')}")
        logger.info(f"  Patterns          : {sk.get('established_patterns', [])}")
        logger.info("=" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    config = ProxyConfig()
    proxy = PACTAXProxy(config)
    await proxy.start()


if __name__ == "__main__":
    asyncio.run(main())
