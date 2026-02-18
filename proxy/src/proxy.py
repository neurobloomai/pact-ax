"""
PACT-AX Proxy - Session Integrity Layer for MCP

Sits between MCP client (Cursor) and MCP server (GitHub),
maintaining relational state and detecting behavioral drift.
"""

import asyncio
import json
import uuid
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import logging

from .story_keeper import StoryKeeper, TrustTrajectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PACT-AX] %(levelname)s: %(message)s'
)
logger = logging.getLogger("pact-ax")


@dataclass
class ProxyConfig:
    """Configuration for PACT-AX proxy"""
    listen_host: str = "127.0.0.1"
    listen_port: int = 3000
    
    # Upstream MCP server (GitHub)
    upstream_command: str = "npx"
    upstream_args: list[str] = None
    
    # Behavior thresholds
    drift_threshold: float = 0.3
    trust_violation_threshold: float = 0.2
    alert_on_drift: bool = True
    block_on_violation: bool = False  # Start permissive, log only
    
    def __post_init__(self):
        if self.upstream_args is None:
            # Default: GitHub MCP server
            self.upstream_args = [
                "-y", 
                "@modelcontextprotocol/server-github"
            ]


class MCPMessage:
    """Parsed MCP JSON-RPC message"""
    
    def __init__(self, raw: bytes):
        self.raw = raw
        self.data = json.loads(raw.decode('utf-8'))
        
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
        return json.dumps(self.data).encode('utf-8')


class PACTAXProxy:
    """
    The main proxy that:
    1. Receives MCP messages from client (Cursor)
    2. Records relational context via StoryKeeper
    3. Evaluates coherence
    4. Forwards to upstream server (GitHub MCP)
    5. Returns response with optional annotations
    """
    
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.story_keeper = StoryKeeper()
        self.story_keeper.drift_threshold = config.drift_threshold
        
        self.session_id: Optional[str] = None
        self.upstream_process: Optional[asyncio.subprocess.Process] = None
        
        # Stats
        self.messages_processed = 0
        self.drift_alerts = 0
        self.blocked_requests = 0
    
    async def start(self):
        """Start the proxy and upstream connection"""
        logger.info(f"Starting PACT-AX Proxy on {self.config.listen_host}:{self.config.listen_port}")
        
        # Create session
        self.session_id = f"pact-{uuid.uuid4().hex[:8]}"
        self.story_keeper.create_session(
            session_id=self.session_id,
            client_identity="cursor:local",
            server_target="github:mcp"
        )
        logger.info(f"Created session: {self.session_id}")
        
        # Start upstream MCP server
        await self._start_upstream()
        
        # Run the proxy loop (stdio mode for now)
        await self._run_stdio_proxy()
    
    async def _start_upstream(self):
        """Start the upstream MCP server process"""
        logger.info(f"Starting upstream: {self.config.upstream_command} {' '.join(self.config.upstream_args)}")
        
        self.upstream_process = await asyncio.create_subprocess_exec(
            self.config.upstream_command,
            *self.config.upstream_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start stderr reader (for upstream logs)
        asyncio.create_task(self._read_upstream_stderr())
    
    async def _read_upstream_stderr(self):
        """Read and log upstream server's stderr"""
        while True:
            line = await self.upstream_process.stderr.readline()
            if not line:
                break
            logger.debug(f"[upstream] {line.decode().strip()}")
    
    async def _run_stdio_proxy(self):
        """
        Proxy mode using stdio (how Cursor connects to MCP servers)
        
        Client (stdin) -> PACT-AX -> Upstream
        Upstream -> PACT-AX -> Client (stdout)
        """
        # Create tasks for bidirectional proxying
        client_to_upstream = asyncio.create_task(self._proxy_client_to_upstream())
        upstream_to_client = asyncio.create_task(self._proxy_upstream_to_client())
        
        # Wait for either direction to complete (usually means connection closed)
        done, pending = await asyncio.wait(
            [client_to_upstream, upstream_to_client],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Clean up
        for task in pending:
            task.cancel()
        
        if self.upstream_process:
            self.upstream_process.terminate()
        
        self._log_session_summary()
    
    async def _proxy_client_to_upstream(self):
        """Handle messages from client (Cursor) to upstream (GitHub)"""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            # Read JSON-RPC message (newline-delimited)
            line = await reader.readline()
            if not line:
                break
            
            try:
                msg = MCPMessage(line.strip())
                self.messages_processed += 1
                
                # Process through PACT-AX
                should_forward, alert = self._process_outgoing(msg)
                
                if alert:
                    self.drift_alerts += 1
                    logger.warning(alert)
                
                if should_forward:
                    # Forward to upstream
                    self.upstream_process.stdin.write(line)
                    await self.upstream_process.stdin.drain()
                else:
                    self.blocked_requests += 1
                    logger.warning(f"Blocked request: {msg.method}")
                    # Send error response back to client
                    await self._send_blocked_response(msg)
                    
            except json.JSONDecodeError:
                # Pass through non-JSON content
                self.upstream_process.stdin.write(line)
                await self.upstream_process.stdin.drain()
    
    async def _proxy_upstream_to_client(self):
        """Handle responses from upstream (GitHub) to client (Cursor)"""
        while True:
            line = await self.upstream_process.stdout.readline()
            if not line:
                break
            
            try:
                msg = MCPMessage(line.strip())
                
                # Process response (could add annotations here)
                annotated = self._process_incoming(msg)
                
                # Send to client
                sys.stdout.buffer.write(annotated.to_bytes() + b'\n')
                sys.stdout.buffer.flush()
                
            except json.JSONDecodeError:
                # Pass through non-JSON content
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
    
    def _process_outgoing(self, msg: MCPMessage) -> tuple[bool, Optional[str]]:
        """
        Process outgoing message through PACT-AX.
        
        Returns:
            - should_forward: Whether to send to upstream
            - alert: Optional drift alert message
        """
        if not msg.is_request:
            return True, None
        
        # Record in StoryKeeper
        event, alert = self.story_keeper.record_event(
            session_id=self.session_id,
            method=msg.method,
            params=msg.params
        )
        
        # Log the interaction
        session = self.story_keeper.sessions[self.session_id]
        logger.info(
            f"[{msg.method}] pattern={event.resource_pattern} "
            f"coherence={event.coherence_score:.2f} "
            f"trust={session.trust_level:.2f} "
            f"trajectory={session.trajectory.value}"
        )
        
        # Decide whether to block
        should_forward = True
        if self.config.block_on_violation:
            if session.trajectory == TrustTrajectory.VIOLATED:
                should_forward = False
            if session.trust_level < self.config.trust_violation_threshold:
                should_forward = False
        
        return should_forward, alert
    
    def _process_incoming(self, msg: MCPMessage) -> MCPMessage:
        """
        Process incoming response, potentially adding annotations.
        
        For now, pass through unchanged. Future: add PACT-AX metadata.
        """
        return msg
    
    async def _send_blocked_response(self, original: MCPMessage):
        """Send an error response when we block a request"""
        response = {
            "jsonrpc": "2.0",
            "id": original.id,
            "error": {
                "code": -32001,
                "message": "PACT-AX: Request blocked due to relational context violation",
                "data": {
                    "session_id": self.session_id,
                    "reason": "trust_violation",
                    "summary": self.story_keeper.get_session_summary(self.session_id)
                }
            }
        }
        sys.stdout.buffer.write(json.dumps(response).encode() + b'\n')
        sys.stdout.buffer.flush()
    
    def _log_session_summary(self):
        """Log session summary on exit"""
        summary = self.story_keeper.get_session_summary(self.session_id)
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Messages Processed: {self.messages_processed}")
        logger.info(f"Drift Alerts: {self.drift_alerts}")
        logger.info(f"Blocked Requests: {self.blocked_requests}")
        logger.info(f"Final Trust Level: {summary.get('trust_level', 'N/A')}")
        logger.info(f"Trust Trajectory: {summary.get('trajectory', 'N/A')}")
        logger.info(f"Established Patterns: {summary.get('established_patterns', [])}")
        logger.info(f"Drift Risk: {summary.get('drift_risk', 'N/A')}")
        logger.info("=" * 60)


async def main():
    """Entry point"""
    config = ProxyConfig()
    proxy = PACTAXProxy(config)
    await proxy.start()


if __name__ == "__main__":
    asyncio.run(main())
