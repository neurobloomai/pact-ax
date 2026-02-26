"""
PACT-AX Proxy — Session Integrity Layer for MCP

Sits between Cursor (MCP client) and GitHub (MCP server),
maintaining relational state through two layers:

    StoryKeeper  — behavioral coherence, pattern drift detection (AX expression layer)
    RLP-0        — relational primitive gating (rupture detection, gate authority)

Architecture (stdio mode — Docker/npx):
    Cursor stdin/stdout
          ↕  MCP JSON-RPC (newline-delimited)
    PACTAXProxy
          ├── StoryKeeper.record_event()   → coherence score, drift alert
          └── RLPBridge.sync()             → rupture_risk, gate open/closed
          ↕  subprocess stdin/stdout
    Docker/npx: GitHub MCP server
          ↕
    GitHub API

Architecture (http mode — no Docker needed):
    Cursor stdin/stdout
          ↕  MCP JSON-RPC (newline-delimited)
    PACTAXProxy  (same PACT-AX analysis)
          ↕  HTTPS POST per message
    https://api.githubcopilot.com/mcp/
          ↕
    GitHub API

Environment variables:
    GITHUB_PERSONAL_ACCESS_TOKEN   required
    PACT_UPSTREAM_MODE             http (no Docker) | docker (default) | npx
    PACT_UPSTREAM_URL              override HTTP endpoint (default: api.githubcopilot.com/mcp/)
    PACT_DRIFT_THRESHOLD           float, default 0.3
    PACT_RUPTURE_THRESHOLD         float, default 0.6
    PACT_BLOCK_ON_VIOLATION        true | false (default false)
    PACT_WARMUP_CALLS              int, default 3  — requests forwarded before gating starts
    PACT_BLOCK_INJECTION           true | false (default false) — block on injection detection
    PACT_AUDIT_LOG                 path to append-only JSONL audit log (default: ./pact-ax-audit.jsonl)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid
import sys
from dataclasses import dataclass, field
from typing import Optional
import logging

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

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
    warmup_calls: int = field(
        default_factory=lambda: int(os.environ.get("PACT_WARMUP_CALLS", "3"))
    )
    block_injection: bool = field(
        default_factory=lambda: os.environ.get(
            "PACT_BLOCK_INJECTION", "false"
        ).lower() == "true"
    )
    audit_log_path: str = field(
        default_factory=lambda: os.environ.get("PACT_AUDIT_LOG", "./pact-ax-audit.jsonl")
    )

    @property
    def upstream_url(self) -> str:
        return os.environ.get(
            "PACT_UPSTREAM_URL", "https://api.githubcopilot.com/mcp/"
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
# SSE parser
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> Optional[dict]:
    """
    Extract the first JSON payload from a Server-Sent Events body.

    SSE lines look like:
        data: {"jsonrpc":"2.0","id":1,"result":{...}}

    Returns the parsed dict of the first ``data:`` line that contains
    valid JSON, or None if nothing parseable is found.
    """
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Injection detection
# ---------------------------------------------------------------------------

# Patterns that suggest an attacker has embedded instructions inside
# file content, issue text, or any other upstream response body.
# Matched case-insensitively against every text content item.
_INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "disregard all instructions",
    "forget your instructions",
    "you are now",
    "act as if you are",
    "new instructions:",
    "system prompt:",
    "<system>",
    "[system]",
    "[inst]",
    "###instruction",
    "###system",
    "override your",
    "your new role",
    "your true instructions",
]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only JSONL audit log — one JSON record per line, never truncated.

    Designed for multi-session deployments: every proxy instance (one per
    Cursor connection) writes to the *same* file, distinguished by session_id.
    Survives process restarts.  Compatible with jq, grep, and any log aggregator.

    Minimum fields on every record:
        ts          ISO-8601 UTC timestamp
        event       event type (SESSION_START, REQUEST, GATE_CLOSED, …)
        session_id  pact-<hex> unique to this proxy instance
    """

    # Event type constants
    SESSION_START  = "session_start"
    SESSION_END    = "session_end"
    REQUEST        = "request"
    WARMUP         = "warmup_request"
    GATE_CLOSED    = "gate_closed"
    GATE_REOPENED  = "gate_reopened"
    REPAIR_INVALID = "repair_invalid"
    INJECTION      = "injection_alert"
    BLOCKED        = "blocked_request"

    def __init__(self, path: str):
        self.path = path
        self._file = None

    def open(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # line-buffered so each record hits disk immediately
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")

    def log(self, event: str, session_id: str, **kwargs):
        if self._file is None:
            return
        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "event": event,
            "session_id": session_id,
        }
        record.update(kwargs)
        try:
            self._file.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug(f"Audit log write error: {exc}")

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


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
        self.injection_warnings = 0

        # Cold start: forward first N requests without gating
        self._warmup_count = 0

        # Gate repair auth: token generated when gate first closes;
        # must be presented in a notifications/pact-ax/repair message to reopen.
        self._gate_repair_token: Optional[str] = None

        # Client identity — updated from MCP initialize message params
        self._client_identity: str = "unknown"

        # Persistent audit log — shared across all proxy instances
        self.audit: Optional[AuditLog] = None

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
            client_identity=self._client_identity,
            server_target="github:mcp",
        )
        self.rlp_bridge = RLPBridge(
            rupture_threshold=self.config.rupture_threshold
        )
        logger.info(f"Session: {self.session_id}")

        # Open shared audit log — appends to existing file across sessions
        self.audit = AuditLog(self.config.audit_log_path)
        self.audit.open()
        logger.info(f"Audit log: {self.config.audit_log_path}")
        self.audit.log(AuditLog.SESSION_START, self.session_id,
            upstream_mode=self.config.upstream_mode,
            drift_threshold=self.config.drift_threshold,
            rupture_threshold=self.config.rupture_threshold,
            warmup_calls=self.config.warmup_calls,
            block_injection=self.config.block_injection,
        )

        if self.config.upstream_mode == "http":
            logger.info(f"HTTP upstream: {self.config.upstream_url}")
            if not _HTTPX_AVAILABLE:
                raise RuntimeError("httpx is required for HTTP mode: pip install httpx")
        else:
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
        if self.config.upstream_mode == "http":
            await self._run_http_proxy()
            return

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

        while True:
            # run_in_executor lets the event loop stay live while waiting for stdin.
            # connect_read_pipe returns empty bytes immediately on Python 3.9 macOS
            # when no data is available yet — treating it as EOF. This avoids that.
            line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            try:
                msg = MCPMessage(stripped)
                self.messages_processed += 1

                # Capture real client identity from MCP initialize handshake
                if msg.method == "initialize":
                    ci = msg.params.get("clientInfo", {})
                    self._client_identity = (
                        f"{ci.get('name', 'unknown')}:{ci.get('version', '0')}"
                    )
                    logger.info(f"Client identified: {self._client_identity}")
                    self.audit.log(AuditLog.SESSION_START, self.session_id,
                        client=self._client_identity, event_detail="initialize_received")

                # Repair notifications are consumed here — never forwarded upstream
                if self._handle_repair_attempt(msg):
                    continue

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
        """GitHub MCP → Cursor — with injection inspection before forwarding."""
        while True:
            line = await self.upstream_process.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = MCPMessage(stripped)
                injection_alert = self._inspect_response(msg.data)
                if injection_alert:
                    logger.warning(injection_alert)
                    self.audit.log(AuditLog.INJECTION, self.session_id,
                        alert=injection_alert, blocked=self.config.block_injection)
                    if self.config.block_injection:
                        # Replace response with an error rather than forwarding
                        err = {
                            "jsonrpc": "2.0", "id": msg.id,
                            "error": {
                                "code": -32002,
                                "message": "PACT-AX: response blocked — injection pattern detected",
                                "data": {"alert": injection_alert},
                            },
                        }
                        sys.stdout.buffer.write(json.dumps(err).encode() + b"\n")
                        sys.stdout.buffer.flush()
                        continue
                sys.stdout.buffer.write(msg.to_bytes() + b"\n")
                sys.stdout.buffer.flush()
            except (json.JSONDecodeError, UnicodeDecodeError):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # HTTP upstream mode (no Docker/npx required)
    # ------------------------------------------------------------------

    async def _run_http_proxy(self):
        """
        HTTP upstream — reads MCP messages from stdin, POSTs each to the
        GitHub remote MCP endpoint, writes responses back to stdout.

        Handles:
          - Session establishment via Mcp-Session-Id header
          - 202 Accepted for notifications (no response body)
          - SSE responses (text/event-stream)
          - JSON responses (application/json)
        """
        pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        url = self.config.upstream_url
        mcp_session_id: Optional[str] = None
        loop = asyncio.get_running_loop()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    msg = MCPMessage(stripped)
                    self.messages_processed += 1

                    # Capture real client identity from MCP initialize handshake
                    if msg.method == "initialize":
                        ci = msg.params.get("clientInfo", {})
                        self._client_identity = (
                            f"{ci.get('name', 'unknown')}:{ci.get('version', '0')}"
                        )
                        logger.info(f"Client identified: {self._client_identity}")
                        self.audit.log(AuditLog.SESSION_START, self.session_id,
                            client=self._client_identity, event_detail="initialize_received")

                    # Repair notifications are consumed here — never forwarded upstream
                    if self._handle_repair_attempt(msg):
                        continue

                    # PACT-AX evaluation on requests only
                    should_forward, alert = True, None
                    if msg.is_request:
                        should_forward, alert = self._evaluate(msg)
                        if alert:
                            logger.warning(alert)

                    if not should_forward:
                        self.blocked_requests += 1
                        await self._send_blocked(msg)
                        continue

                    # Build headers
                    headers = {
                        "Authorization": f"Bearer {pat}",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    }
                    if mcp_session_id:
                        headers["Mcp-Session-Id"] = mcp_session_id

                    # POST to GitHub MCP endpoint
                    try:
                        resp = await client.post(url, headers=headers, json=msg.data)
                    except httpx.RequestError as exc:
                        logger.error(f"HTTP request failed: {exc}")
                        if msg.is_request:
                            err = {"jsonrpc":"2.0","id":msg.id,
                                   "error":{"code":-32000,"message":str(exc)}}
                            sys.stdout.buffer.write(json.dumps(err).encode() + b"\n")
                            sys.stdout.buffer.flush()
                        continue

                    # Grab session ID on first response
                    if not mcp_session_id:
                        sid = resp.headers.get("mcp-session-id") or \
                              resp.headers.get("Mcp-Session-Id")
                        if sid:
                            mcp_session_id = sid
                            logger.info(f"HTTP session: {mcp_session_id}")

                    # 202 = notification accepted, no body to forward
                    if resp.status_code == 202:
                        continue

                    # Non-200 error
                    if resp.status_code != 200:
                        logger.warning(f"Upstream HTTP {resp.status_code}: {resp.text[:200]}")
                        err = {"jsonrpc":"2.0","id":msg.id,
                               "error":{"code":resp.status_code,"message":resp.text[:200]}}
                        sys.stdout.buffer.write(json.dumps(err).encode() + b"\n")
                        sys.stdout.buffer.flush()
                        continue

                    # Parse body — JSON or SSE
                    ct = resp.headers.get("content-type", "")
                    if "text/event-stream" in ct:
                        response_data = _parse_sse(resp.text)
                    else:
                        try:
                            response_data = resp.json()
                        except Exception:
                            response_data = None

                    if response_data:
                        injection_alert = self._inspect_response(response_data)
                        if injection_alert:
                            logger.warning(injection_alert)
                            self.audit.log(AuditLog.INJECTION, self.session_id,
                                alert=injection_alert, blocked=self.config.block_injection)
                            if self.config.block_injection:
                                err = {
                                    "jsonrpc": "2.0", "id": msg.id,
                                    "error": {
                                        "code": -32002,
                                        "message": "PACT-AX: response blocked — injection pattern detected",
                                        "data": {"alert": injection_alert},
                                    },
                                }
                                sys.stdout.buffer.write(json.dumps(err).encode() + b"\n")
                                sys.stdout.buffer.flush()
                                continue
                        sys.stdout.buffer.write(
                            json.dumps(response_data).encode() + b"\n"
                        )
                        sys.stdout.buffer.flush()

                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # non-JSON input — skip

        self._log_summary()

    # ------------------------------------------------------------------
    # Response inspection (Layer 3 — downstream guard)
    # ------------------------------------------------------------------

    def _inspect_response(self, data: dict) -> Optional[str]:
        """
        Scan an upstream MCP response for prompt-injection patterns.

        Checks every ``content[].text`` field in ``result``.  Returns a
        warning string if a suspicious pattern is found, otherwise None.

        When PACT_BLOCK_INJECTION=true the caller should treat a non-None
        return as a reason to suppress the response and send an error instead.
        """
        result = data.get("result")
        if not isinstance(result, dict):
            return None
        content_list = result.get("content")
        if not isinstance(content_list, list):
            return None

        for item in content_list:
            text = (item.get("text") or "").lower()
            for pattern in _INJECTION_PATTERNS:
                if pattern in text:
                    self.injection_warnings += 1
                    return (
                        f"[INJECTION ALERT] Upstream response contains suspicious pattern: "
                        f"'{pattern}' — possible prompt injection in content from GitHub."
                    )
        return None

    # ------------------------------------------------------------------
    # Gate repair auth
    # ------------------------------------------------------------------

    def _handle_repair_attempt(self, msg: MCPMessage) -> bool:
        """
        Handle a ``notifications/pact-ax/repair`` notification.

        Returns True if the message was consumed (don't forward upstream).
        The notification params must contain the repair token that was
        logged when the gate closed:

            {"jsonrpc":"2.0","method":"notifications/pact-ax/repair",
             "params":{"token":"<repair_token>"}}
        """
        if msg.method != "notifications/pact-ax/repair":
            return False

        token = (msg.params or {}).get("token", "")

        if not self._gate_repair_token:
            logger.info("[REPAIR] Gate is not closed — no repair needed.")
            return True

        if token != self._gate_repair_token:
            logger.warning(
                f"[REPAIR] Invalid token supplied — gate remains CLOSED. "
                f"Check proxy stderr for the correct token."
            )
            self.audit.log(AuditLog.REPAIR_INVALID, self.session_id,
                token_supplied=token[:4] + "…" if token else "(empty)")
            return True

        # Valid token — acknowledge repair and reopen gate
        self.rlp_bridge.acknowledge_repair()
        self._gate_repair_token = None
        logger.info(
            "[REPAIR] ✓ Repair token accepted — RLP-0 gate REOPENED. "
            "Relational primitives reset. Session continues."
        )
        self.audit.log(AuditLog.GATE_REOPENED, self.session_id)
        return True

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

        # ── Cold start guard ──────────────────────────────────────
        # Forward the first PACT_WARMUP_CALLS requests without gating.
        # StoryKeeper needs a few events to establish a behavioral baseline;
        # firing RLP-0 on call #1 would be a false positive.
        if self._warmup_count < self.config.warmup_calls:
            self._warmup_count += 1
            logger.info(
                f"[{msg.method}] warming up "
                f"({self._warmup_count}/{self.config.warmup_calls}) — "
                f"evaluation deferred, forwarding"
            )
            self.audit.log(AuditLog.WARMUP, self.session_id,
                method=msg.method,
                warmup_count=self._warmup_count,
                warmup_total=self.config.warmup_calls,
            )
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
        self.audit.log(AuditLog.REQUEST, self.session_id,
            method=msg.method,
            pattern=event.resource_pattern,
            coherence=round(event.coherence_score, 3),
            trust=round(session.trust_level, 3),
            trajectory=session.trajectory.value,
            rupture_risk=round(rupture_risk, 3),
            gated=is_gated,
        )

        # ── Gate decision ─────────────────────────────────────────
        # RLP-0 gate has final authority
        if is_gated:
            self.rupture_events += 1
            sig = self.rlp_bridge.last_rupture_signal

            # Generate repair token on first gate closure of this session.
            # Token must be presented in notifications/pact-ax/repair to reopen.
            if not self._gate_repair_token:
                self._gate_repair_token = uuid.uuid4().hex[:12]
                logger.warning(
                    f"\n{'─'*62}\n"
                    f"[GATE CLOSED] To reopen, send this notification to the proxy:\n"
                    f"  method : notifications/pact-ax/repair\n"
                    f"  params : {{\"token\": \"{self._gate_repair_token}\"}}\n"
                    f"{'─'*62}"
                )
                self.audit.log(AuditLog.GATE_CLOSED, self.session_id,
                    rupture_risk=round(rupture_risk, 3),
                    repair_token=self._gate_repair_token,
                    primitives={k: round(v, 3) for k, v in primitives.items()},
                )

            alert = (
                f"[RLP-0 RUPTURE GATE CLOSED]\n"
                f"Session         : {self.session_id}\n"
                f"Rupture risk    : {rupture_risk:.2f} "
                f"(threshold: {self.config.rupture_threshold})\n"
                f"Signal          : {sig}\n"
                f"RLP-0 primitives: {primitives}\n"
                f"Gate status     : CLOSED — send notifications/pact-ax/repair with token to reopen.\n"
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

        # Persist session-end record then close audit file
        if self.audit:
            self.audit.log(AuditLog.SESSION_END, self.session_id,
                client=self._client_identity,
                messages_processed=self.messages_processed,
                drift_alerts=self.drift_alerts,
                rupture_events=self.rupture_events,
                blocked_requests=self.blocked_requests,
                injection_warnings=self.injection_warnings,
                final_trust=round(sk.get("trust_level", 0), 3),
                final_rupture_risk=round(
                    rlp.get("state", {}).get("rupture_risk", 0), 3),
                gate_closed=rlp.get("is_gated", False),
            )
            self.audit.close()

        logger.info("=" * 62)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Session ID        : {self.session_id}")
        logger.info(f"  Messages processed: {self.messages_processed}")
        logger.info(f"  Warmup calls      : {self.config.warmup_calls} (deferred gating)")
        logger.info(f"  Drift alerts      : {self.drift_alerts}")
        logger.info(f"  RLP-0 ruptures    : {self.rupture_events}")
        logger.info(f"  Blocked requests  : {self.blocked_requests}")
        logger.info(f"  Injection warnings: {self.injection_warnings}")
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
