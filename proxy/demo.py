#!/usr/bin/env python3
"""
PACT-AX Demo — Real Cursor ↔ GitHub MCP Flow

Unlike the original demo.py (which called StoryKeeper directly with fake data),
this version drives the ACTUAL proxy:

  demo.py  →  proxy.py subprocess  →  Docker: ghcr.io/github/github-mcp-server  →  GitHub API

Three phases reproduce the original narrative but against real GitHub content:

  Phase 1 — read source files        → establishes "repo:read:source" pattern
  Phase 2 — read config files        → slight variation, still coherent
  Phase 3 — org-level access         → behavioral drift; PACT-AX alerts fire

Usage:
  export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx
  python proxy/demo.py [owner] [repo]
  python proxy/demo.py neurobloomai pact-ax      ← default

Requirements:
  - Docker Desktop running  (or: export PACT_UPSTREAM_MODE=npx)
  - GITHUB_PERSONAL_ACCESS_TOKEN set in environment
  - Run from the pact-ax repo root so Python can resolve proxy.src.proxy
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent   # pact-ax/
DEFAULT_OWNER = "neurobloomai"
DEFAULT_REPO = "pact-ax"
PHASE_DELAY = 0.9       # seconds between tool calls (dramatic effect)
PROXY_BOOT_DELAY = 2.0  # seconds to wait for Docker pull on cold start

# ANSI
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RST    = "\033[0m"

# ---------------------------------------------------------------------------
# MCP JSON-RPC helpers
# ---------------------------------------------------------------------------

_id_counter = 0

def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter

def request(method: str, params: dict) -> bytes:
    return json.dumps({
        "jsonrpc": "2.0", "id": _next_id(),
        "method": method, "params": params,
    }).encode() + b"\n"

def notification(method: str, params: Optional[dict] = None) -> bytes:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return json.dumps(msg).encode() + b"\n"

# ---------------------------------------------------------------------------
# Stderr log parser  (reads proxy's structured INFO lines)
# ---------------------------------------------------------------------------

class ProxyLogParser:
    """
    Parses lines like:
      2024-01-01 12:00:00,000 [PACT-AX] INFO: [tools/call] pattern=repo:read:source coherence=0.70 trust=0.52 traj=building
      2024-01-01 12:00:00,000 [PACT-AX] WARNING: [PACT-AX DRIFT ALERT] ...
    """

    def __init__(self):
        self.lines: list[str] = []
        self.drift_alerts: list[str] = []
        # Last known values
        self.trust: Optional[float] = None
        self.coherence: Optional[float] = None
        self.pattern: Optional[str] = None
        self.trajectory: Optional[str] = None

    def feed(self, raw: str):
        self.lines.append(raw)
        if "DRIFT ALERT" in raw:
            self.drift_alerts.append(raw)
        self._parse_metrics(raw)

    def _parse_metrics(self, line: str):
        def extract_float(key: str) -> Optional[float]:
            if key not in line:
                return None
            try:
                return float(line.split(key)[1].split()[0])
            except (IndexError, ValueError):
                return None

        def extract_str(key: str) -> Optional[str]:
            if key not in line:
                return None
            try:
                return line.split(key)[1].split()[0]
            except IndexError:
                return None

        v = extract_float("trust=");      self.trust       = v if v is not None else self.trust
        v = extract_float("coherence=");  self.coherence   = v if v is not None else self.coherence
        s = extract_str("pattern=");      self.pattern     = s if s is not None else self.pattern
        s = extract_str("traj=");         self.trajectory  = s if s is not None else self.trajectory

    @property
    def drift_count(self) -> int:
        return len(self.drift_alerts)

    @property
    def drift_risk(self) -> str:
        n = self.drift_count
        return "HIGH" if n >= 2 else ("MEDIUM" if n == 1 else "LOW")

# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------

class DemoDriver:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.log = ProxyLogParser()
        self._stderr_task: Optional[asyncio.Task] = None
        self._call_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        env = dict(os.environ)
        env.setdefault("PACT_UPSTREAM_MODE", "docker")
        env.setdefault("PACT_DRIFT_THRESHOLD", "0.3")
        env.setdefault("PACT_BLOCK_ON_VIOLATION", "false")

        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "proxy.src.proxy",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self):
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            self.log.feed(line.decode().strip())

    async def stop(self):
        await asyncio.sleep(0.4)   # let last stderr lines land
        if self._stderr_task:
            self._stderr_task.cancel()
        if self.proc:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                self.proc.terminate()

    # ------------------------------------------------------------------
    # MCP message send/receive
    # ------------------------------------------------------------------

    async def _send(self, data: bytes) -> dict:
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()
        line = await self.proc.stdout.readline()
        return json.loads(line.strip())

    async def _notify(self, data: bytes):
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def handshake(self) -> str:
        resp = await self._send(request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pact-ax-demo", "version": "2.0.0"},
        }))
        await self._notify(notification("notifications/initialized"))
        await asyncio.sleep(0.2)
        server_name = (resp.get("result") or {}).get("serverInfo", {}).get("name", "GitHub MCP")
        return server_name

    async def tool_call(self, tool: str, arguments: dict) -> dict:
        self._call_count += 1
        data = request("tools/call", {"name": tool, "arguments": arguments})

        # Snapshot alert count before the call
        alerts_before = self.log.drift_count

        print(f"\n  {DIM}→ tools/call:{RST} {BOLD}{tool}{RST}", flush=True)

        resp = await self._send(data)
        await asyncio.sleep(0.15)   # let stderr catch up

        # ── PACT-AX metrics ──
        coh  = self.log.coherence
        trust = self.log.trust
        pat  = self.log.pattern
        traj = self.log.trajectory

        coh_str   = f"{coh:.2f}"   if coh   is not None else "—"
        trust_str = f"{trust:.2f}" if trust is not None else "—"

        if coh is None:
            indicator = "⚪"
        elif coh > 0.7:
            indicator = f"{GREEN}🟢{RST}"
        elif coh > 0.3:
            indicator = f"{YELLOW}🟡{RST}"
        else:
            indicator = f"{RED}🔴{RST}"

        print(
            f"  {indicator}  coherence={BOLD}{coh_str}{RST}  "
            f"trust={trust_str}  pattern={DIM}{pat or '—'}{RST}  "
            f"traj={traj or '—'}"
        )

        # Alert fired on THIS call?
        if self.log.drift_count > alerts_before:
            print(f"  {RED}{BOLD}⚠  PACT-AX DRIFT ALERT  — "
                  f"behavior outside relational context{RST}")

        # Was the request blocked?
        if "error" in resp:
            err = resp["error"]
            print(f"  {RED}✗  BLOCKED: {err.get('message', '')}{RST}")
        else:
            content = (resp.get("result") or {}).get("content") or []
            if content:
                snippet = content[0].get("text", "")[:90].replace("\n", " ")
                print(f"  {DIM}↩  {snippet!r}…{RST}")
            else:
                print(f"  {DIM}↩  (ok){RST}")

        await asyncio.sleep(PHASE_DELAY)
        return resp

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def banner(self):
        upstream = os.environ.get("PACT_UPSTREAM_MODE", "docker")
        block    = os.environ.get("PACT_BLOCK_ON_VIOLATION", "false")
        print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ██████╗  █████╗  ██████╗████████╗       █████╗ ██╗  ██╗  ║
║  ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝      ██╔══██╗╚██╗██╔╝  ║
║  ██████╔╝███████║██║        ██║   █████╗███████║ ╚███╔╝   ║
║  ██╔═══╝ ██╔══██║██║        ██║   ╚════╝██╔══██║ ██╔██╗   ║
║  ██║     ██║  ██║╚██████╗   ██║         ██║  ██║██╔╝ ██╗  ║
║  ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝         ╚═╝  ╚═╝╚═╝  ╚═╝  ║
║                                                              ║
║  Real Cursor ↔ GitHub MCP Flow  •  Drift Detection Live     ║
║  neurobloom.ai                                              ║
╚══════════════════════════════════════════════════════════════╝{RST}

  Repo     : {BOLD}{self.owner}/{self.repo}{RST}
  Upstream : {BOLD}{upstream}{RST}   (Docker → ghcr.io/github/github-mcp-server)
  Blocking : {BOLD}{block}{RST}      (set PACT_BLOCK_ON_VIOLATION=true to enforce)
""")

    def phase(self, n: int, title: str, description: str):
        print(f"\n{BLUE}{BOLD}{'─' * 62}{RST}")
        print(f"{BLUE}{BOLD}  Phase {n}: {title}{RST}")
        print(f"{DIM}  {description}{RST}")
        print(f"{BLUE}{BOLD}{'─' * 62}{RST}")

    def summary(self):
        log = self.log
        risk_color = (RED if log.drift_risk == "HIGH"
                      else YELLOW if log.drift_risk == "MEDIUM"
                      else GREEN)
        trust = f"{log.trust:.2f}" if log.trust is not None else "—"

        print(f"""
{BOLD}{'═' * 62}
  SESSION SUMMARY
{'═' * 62}{RST}
  Tool calls made  : {self._call_count}
  Drift alerts     : {RED if log.drift_count else GREEN}{log.drift_count}{RST}
  Drift risk       : {risk_color}{BOLD}{log.drift_risk}{RST}
  Final trust      : {trust}
  Trajectory       : {log.trajectory or '—'}

{BOLD}  KEY INSIGHT:{RST}
  Policy engine sees : "Valid token, permitted scope"   {GREEN}✓{RST}
  PACT-AX sees       : "Behavior outside relational context"  {RED}⚠{RST}

  {DIM}The org-level access was policy-permitted but flagged because{RST}
  {DIM}it doesn't cohere with a developer reading /src files.{RST}
  {DIM}This is the gap between compliance and relational integrity.{RST}
{BOLD}{'═' * 62}{RST}
""")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(owner: str, repo: str):
    driver = DemoDriver(owner, repo)
    driver.banner()

    # ── Boot ──────────────────────────────────────────────────────────
    print(f"{BOLD}Starting PACT-AX proxy…{RST}  (Docker pull may take a moment on first run)")
    await driver.start()
    await asyncio.sleep(PROXY_BOOT_DELAY)

    print(f"{BOLD}MCP handshake…{RST}")
    server_name = await driver.handshake()
    print(f"  {GREEN}✓  Connected:{RST} {server_name}\n")

    # ── Phase 1: Establish pattern ────────────────────────────────────
    driver.phase(1,
        "Establishing Normal Pattern",
        "Reading source files — trust builds, repo:read:source becomes established.")

    for path in ["proxy/src/proxy.py", "proxy/src/story_keeper.py",
                 "proxy/demo.py", "README.md"]:
        await driver.tool_call("get_file_contents",
                               {"owner": owner, "repo": repo, "path": path})

    # ── Phase 2: Minor variation ──────────────────────────────────────
    driver.phase(2,
        "Minor Variation (Still Coherent)",
        "Config files — slightly outside source pattern but related.")

    await driver.tool_call("get_file_contents",
                           {"owner": owner, "repo": repo, "path": "pyproject.toml"})
    await driver.tool_call("get_file_contents",
                           {"owner": owner, "repo": repo, "path": "requirements.txt"})

    # ── Phase 3: Drift ────────────────────────────────────────────────
    driver.phase(3,
        "⚠  Behavioral Drift Detected",
        "Org-level and sensitive keyword access — outside established context.")

    await driver.tool_call("list_org_members", {"org": owner})

    await driver.tool_call("search_code",
                           {"q": f"secret OR token OR password repo:{owner}/{repo}"})

    # ── Summary ───────────────────────────────────────────────────────
    driver.summary()
    await driver.stop()


def main():
    owner = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OWNER
    repo  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REPO

    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print(f"\n{RED}✗  GITHUB_PERSONAL_ACCESS_TOKEN is not set.{RST}")
        print("   Export it:  export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx\n")
        sys.exit(1)

    asyncio.run(run(owner, repo))


if __name__ == "__main__":
    main()
