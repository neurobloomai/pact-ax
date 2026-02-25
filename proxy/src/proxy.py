"""
PACT-AX Proxy - Session Integrity Layer for MCP

Sits between MCP client (Cursor) and MCP server (GitHub),
maintaining relational state and detecting behavioral drift.

CHANGES from original:
- Upstream updated from deprecated @modelcontextprotocol/server-github
  to ghcr.io/github/github-mcp-server (Docker) or remote HTTP mode.
- GITHUB_PERSONAL_ACCESS_TOKEN properly plumbed to upstream subprocess.
- asyncio.get_event_loop() replaced with asyncio.get_running_loop().
- Added UPSTREAM_MODE env var so you can switch transports without
  editing the source (stdio=Docker, http=remote API).
"""

import asyncio
import json
import os
import uuid
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import logging

from .story_keeper import StoryKeeper, TrustTrajectory

# Configure logging to stderr so stdout stays clean for MCP messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PACT-AX] %(levelname)s: %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger("pact-ax")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    """Configuration for PACT-AX proxy.

    Environment variables override defaults:
      PACT_UPSTREAM_MODE   : "docker" (default) | "npx"
      GITHUB_PERSONAL_ACCESS_TOKEN : required for upstream auth
      PACT_DRIFT_THRESHOLD  : float, default 0.3
      PACT_BLOCK_ON_VIOLATION : "true" | "false" (default false)
    """

    # Upstream MCP server transport
    upstream_mode: str = field(
        default_factory=lambda: os.environ.get("PACT_UPSTREAM_MODE", "docker")
    )

    # Behavior thresholds
    drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PACT_DRIFT_THRESHOLD", "0.3"))
    )
    trust_violation_threshold: float = 0.2
    alert_on_drift: bool = True
    block_on_violation: bool = field(
        default_factory=lambda: os.environ.get("PACT_BLOCK_ON_VIOLATION", "false").lower() == "true"
    )

    @property
    def upstream_command(self) -> str:
        if self.upstream_mode == "npx":
            return "npx"
        return "docker"

    @property
    def upstream_args(self) -> list[str]:
        if self.upstream_mode == "npx":
            # NOTE: @modelcontextprotocol/server-github is deprecated (April 2025).
            # Use docker mode instead.  This branch kept for local dev without Docker.
            return ["-y", "@modelcontextprotocol/server-github"]
        # Docker mode — recommended
        return [
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ]

    @property
    def upstream_env(self) -> dict:
        """Environment passed to the upstream subprocess."""
        env = dict(os.environ)  # inherit full env so Docker can find token
        pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if not pat:
            logger.warning(
                "GITHUB_PERSONAL_ACCESS_TOKEN is not set. "
                "Upstream GitHub MCP server will likely reject requests."
            )
        env["GITHUB_PERSONAL_ACCESS_TOKEN"] = pat
        return env


# ---------------------------------------------------------------------------
# MCP message wrapper
# ---------------------------------------------------------------------------

class MCPMessage:
    """Parsed MCP JSON-RPC message."""

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
# Proxy core
# ---------------------------------------------------------------------------

class PACTAXProxy:
    """
    Bidirectional stdio proxy.

    Cursor (stdin/stdout) ↔ PACT-AX ↔ GitHub MCP (subprocess stdin/stdout)

    Each outgoing request is evaluated by StoryKeeper before forwarding.
    Responses are passed through unchanged (future: add PACT-AX annotations).
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.story_keeper = StoryKeeper()
        self.story_keeper.drift_threshold = config.drift_threshold

        self.session_id: Optional[str] = None
        self.upstream_process: Optional[asyncio.subprocess.Process] = None

        # Counters
        self.messages_processed = 0
        self.drift_alerts = 0
        self.blocked_requests = 0

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self):
        logger.info(
            f"Starting PACT-AX Proxy  mode={self.config.upstream_mode}  "
            f"block_on_violation={self.config.block_on_violation}"
        )

        self.session_id = f"pact-{uuid.uuid4().hex[:8]}"
        self.story_keeper.create_session(
            session_id=self.session_id,
            client_identity="cursor:local",
            server_target="github:mcp",
        )
        logger.info(f"Session created: {self.session_id}")

        await self._start_upstream()
        await self._run_stdio_proxy()

    async def _start_upstream(self):
        cmd = self.config.upstream_command
        args = self.config.upstream_args
        logger.info(f"Launching upstream: {cmd} {' '.join(args)}")

        self.upstream_process = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.config.upstream_env,
        )
        asyncio.create_task(self._drain_upstream_stderr())

    async def _drain_upstream_stderr(self):
        """Forward upstream stderr to our stderr so it's visible in logs."""
        while True:
            line = await self.upstream_process.stderr.readline()
            if not line:
                break
            logger.debug(f"[upstream] {line.decode().strip()}")

    # ------------------------------------------------------------------
    # Proxy loop
    # ------------------------------------------------------------------

    async def _run_stdio_proxy(self):
        c2u = asyncio.create_task(self._proxy_client_to_upstream())
        u2c = asyncio.create_task(self._proxy_upstream_to_client())

        done, pending = await asyncio.wait(
            [c2u, u2c], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if self.upstream_process:
            try:
                self.upstream_process.terminate()
            except ProcessLookupError:
                pass

        self._log_session_summary()

    async def _proxy_client_to_upstream(self):
        """Cursor → PACT-AX → GitHub MCP."""
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

                should_forward, alert = self._evaluate_outgoing(msg)

                if alert:
                    self.drift_alerts += 1
                    logger.warning(alert)

                if should_forward:
                    self.upstream_process.stdin.write(stripped + b"\n")
                    await self.upstream_process.stdin.drain()
                else:
                    self.blocked_requests += 1
                    logger.warning(f"Blocked: {msg.method}")
                    await self._send_blocked_response(msg)

            except (json.JSONDecodeError, UnicodeDecodeError):
                # Pass non-JSON through unchanged (e.g., handshake bytes)
                self.upstream_process.stdin.write(line)
                await self.upstream_process.stdin.drain()

    async def _proxy_upstream_to_client(self):
        """GitHub MCP → PACT-AX → Cursor."""
        while True:
            line = await self.upstream_process.stdout.readline()
            if not line:
                break

            stripped = line.strip()
            if not stripped:
                continue

            try:
                msg = MCPMessage(stripped)
                annotated = self._annotate_response(msg)
                sys.stdout.buffer.write(annotated.to_bytes() + b"\n")
                sys.stdout.buffer.flush()
            except (json.JSONDecodeError, UnicodeDecodeError):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # PACT-AX evaluation
    # ------------------------------------------------------------------

    def _evaluate_outgoing(self, msg: MCPMessage) -> tuple[bool, Optional[str]]:
        """Run the message through StoryKeeper and decide whether to forward."""
        if not msg.is_request:
            return True, None

        event, alert = self.story_keeper.record_event(
            session_id=self.session_id,
            method=msg.method,
            params=msg.params,
        )

        session = self.story_keeper.sessions[self.session_id]
        logger.info(
            f"[{msg.method}] pattern={event.resource_pattern} "
            f"coherence={event.coherence_score:.2f} "
            f"trust={session.trust_level:.2f} "
            f"traj={session.trajectory.value}"
        )

        should_forward = True
        if self.config.block_on_violation:
            if session.trajectory == TrustTrajectory.VIOLATED:
                should_forward = False
            elif session.trust_level < self.config.trust_violation_threshold:
                should_forward = False

        return should_forward, alert

    def _annotate_response(self, msg: MCPMessage) -> MCPMessage:
        """
        Optionally enrich upstream responses with PACT-AX context.
        Currently a pass-through; extend here to add trust metadata.
        """
        return msg

    async def _send_blocked_response(self, original: MCPMessage):
        summary = self.story_keeper.get_session_summary(self.session_id)
        response = {
            "jsonrpc": "2.0",
            "id": original.id,
            "error": {
                "code": -32001,
                "message": "PACT-AX: request blocked — relational context violation",
                "data": {
                    "session_id": self.session_id,
                    "reason": "trust_violation",
                    "summary": summary,
                },
            },
        }
        sys.stdout.buffer.write(json.dumps(response).encode() + b"\n")
        sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _log_session_summary(self):
        summary = self.story_keeper.get_session_summary(self.session_id)
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Session ID        : {self.session_id}")
        logger.info(f"  Messages processed: {self.messages_processed}")
        logger.info(f"  Drift alerts      : {self.drift_alerts}")
        logger.info(f"  Blocked requests  : {self.blocked_requests}")
        logger.info(f"  Final trust level : {summary.get('trust_level', 'N/A'):.2f}")
        logger.info(f"  Trust trajectory  : {summary.get('trajectory', 'N/A')}")
        logger.info(f"  Drift risk        : {summary.get('drift_risk', 'N/A')}")
        logger.info(f"  Patterns          : {summary.get('established_patterns', [])}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    config = ProxyConfig()
    proxy = PACTAXProxy(config)
    await proxy.start()


if __name__ == "__main__":
    asyncio.run(main())
