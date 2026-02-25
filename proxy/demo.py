#!/usr/bin/env python3
"""
PACT-AX Demo — Real Cursor ↔ GitHub MCP Flow with RLP-0 Gate

Drives the actual proxy via subprocess with live MCP JSON-RPC messages.

Full stack:
  demo.py
    → proxy.src subprocess
        → StoryKeeper  (behavioral drift detection)
        → RLPBridge    (RLP-0 gate: trust · intent · narrative · commitments)
        → Docker: ghcr.io/github/github-mcp-server
            → GitHub API

Three phases:
  Phase 1 — read source files       → patterns establish, trust builds
  Phase 2 — read config files       → slight coherence dip, still safe
  Phase 3 — org access + sensitive  → trust/narrative/intent collapse
                                       RLP-0 rupture_risk hits threshold
                                       Gate CLOSES → requests BLOCKED

Usage:
  export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx
  python proxy/demo.py [owner] [repo]
  python proxy/demo.py neurobloomai pact-ax      ← default

Optional:
  export PACT_RUPTURE_THRESHOLD=0.5   ← lower = fires earlier (default 0.6)
  export PACT_UPSTREAM_MODE=npx       ← skip Docker (uses deprecated npm pkg)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OWNER = "neurobloomai"
DEFAULT_REPO  = "pact-ax"
PHASE_DELAY   = 0.9
BOOT_DELAY    = 5.0   # allow time for Docker pull on first run

# ANSI
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
B = "\033[94m"; BOLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"

# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

_id = 0

def req(method: str, params: dict) -> bytes:
    global _id; _id += 1
    return json.dumps({"jsonrpc":"2.0","id":_id,"method":method,"params":params}).encode()+b"\n"

def notif(method: str) -> bytes:
    return json.dumps({"jsonrpc":"2.0","method":method}).encode()+b"\n"

# ---------------------------------------------------------------------------
# Proxy log parser — reads stderr, extracts both SK and RLP-0 metrics
# ---------------------------------------------------------------------------

class LogParser:
    def __init__(self):
        self.lines: List[str] = []
        self.drift_alerts: List[str] = []
        self.rupture_events: List[str] = []

        # StoryKeeper metrics
        self.sk_trust: Optional[float] = None
        self.sk_coherence: Optional[float] = None
        self.sk_pattern: Optional[str] = None
        self.sk_traj: Optional[str] = None

        # RLP-0 metrics
        self.rlp_rupture_risk: Optional[float] = None
        self.rlp_gated: bool = False

    def feed(self, line: str):
        self.lines.append(line)
        if "DRIFT ALERT" in line:
            self.drift_alerts.append(line)
        if "RUPTURE GATE" in line or "RUPTURE_DETECTED" in line:
            self.rupture_events.append(line)
        self._parse(line)

    def _parse(self, line: str):
        def f(key: str) -> Optional[float]:
            if key not in line: return None
            try: return float(line.split(key)[1].split()[0])
            except: return None
        def s(key: str) -> Optional[str]:
            if key not in line: return None
            try: return line.split(key)[1].split()[0]
            except: return None

        v = f("coherence=");         self.sk_coherence    = v if v is not None else self.sk_coherence
        v = f("trust=");             self.sk_trust        = v if v is not None else self.sk_trust
        t = s("traj=");              self.sk_traj         = t if t is not None else self.sk_traj
        p = s("pattern=");           self.sk_pattern      = p if p is not None else self.sk_pattern
        v = f("rlp.rupture_risk=");  self.rlp_rupture_risk = v if v is not None else self.rlp_rupture_risk

        if "rlp.gated=True" in line:  self.rlp_gated = True
        if "rlp.gated=False" in line: self.rlp_gated = False

    @property
    def drift_risk(self) -> str:
        n = len(self.drift_alerts)
        return "HIGH" if n >= 2 else "MEDIUM" if n == 1 else "LOW"


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------

class Demo:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo  = repo
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.log   = LogParser()
        self._stderr_task = None
        self._calls = 0

    async def start(self):
        env = dict(os.environ)
        env.setdefault("PACT_UPSTREAM_MODE", "docker")
        env.setdefault("PACT_DRIFT_THRESHOLD", "0.3")
        env.setdefault("PACT_RUPTURE_THRESHOLD", "0.6")

        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "proxy.src.proxy",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain())

    async def health_check(self):
        """Check proxy is still alive after boot delay. Print stderr if crashed."""
        await asyncio.sleep(0.5)
        if self.proc.returncode is not None:
            # Process already exited — collect stderr and show it
            stderr_lines = self.log.lines
            print(f"\n{R}{BOLD}✗  Proxy crashed on startup (exit code {self.proc.returncode}){RST}")
            if stderr_lines:
                print(f"{R}--- proxy stderr ---{RST}")
                for line in stderr_lines[-20:]:
                    print(f"  {line}")
                print(f"{R}--------------------{RST}")
            else:
                print(f"{DIM}  (no stderr captured — check Docker is running){RST}")
            sys.exit(1)

    async def _drain(self):
        while True:
            line = await self.proc.stderr.readline()
            if not line: break
            self.log.feed(line.decode().strip())

    async def _send(self, data: bytes) -> dict:
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()
        line = await self.proc.stdout.readline()
        return json.loads(line.strip())

    async def _notify(self, data: bytes):
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def handshake(self) -> str:
        resp = await self._send(req("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pact-ax-demo", "version": "2.0.0"},
        }))
        await self._notify(notif("notifications/initialized"))
        await asyncio.sleep(0.2)
        return (resp.get("result") or {}).get("serverInfo", {}).get("name", "GitHub MCP")

    async def call(self, tool: str, arguments: dict) -> dict:
        self._calls += 1
        data = req("tools/call", {"name": tool, "arguments": arguments})
        alerts_before  = len(self.log.drift_alerts)
        ruptures_before = len(self.log.rupture_events)

        print(f"\n  {DIM}→ {tool}{RST}", flush=True)

        resp = await self._send(data)
        await asyncio.sleep(0.15)

        coh  = self.log.sk_coherence
        trust = self.log.sk_trust
        rup  = self.log.rlp_rupture_risk
        gated = self.log.rlp_gated

        # Coherence indicator
        if coh is None:  ind = "⚪"
        elif coh > 0.7:  ind = f"{G}🟢{RST}"
        elif coh > 0.3:  ind = f"{Y}🟡{RST}"
        else:             ind = f"{R}🔴{RST}"

        # RLP-0 rupture risk bar
        rup_str = f"{rup:.2f}" if rup is not None else "—"
        rup_color = R if (rup or 0) > 0.5 else Y if (rup or 0) > 0.3 else G
        gate_str = f"{R}{BOLD}CLOSED{RST}" if gated else f"{G}open{RST}"

        print(
            f"  {ind}  coherence={BOLD}{coh:.2f if coh else '—'}{RST}  "
            f"trust={trust:.2f if trust else '—'}  "
            f"traj={self.log.sk_traj or '—'}  │  "
            f"rlp.rupture={rup_color}{rup_str}{RST}  gate={gate_str}"
        )

        # Drift alert fired?
        if len(self.log.drift_alerts) > alerts_before:
            print(f"  {Y}⚠  StoryKeeper drift alert{RST}")

        # RLP-0 rupture fired?
        if len(self.log.rupture_events) > ruptures_before:
            print(f"  {R}{BOLD}🔴 RLP-0 RUPTURE DETECTED — gate is now CLOSED{RST}")

        # Was it blocked?
        if "error" in resp:
            err = resp["error"]
            data_block = err.get("data", {})
            primitives = (data_block.get("rlp0_status") or {}).get("state", {})
            print(f"  {R}{BOLD}✗  BLOCKED by RLP-0 gate{RST}")
            if primitives:
                print(
                    f"  {DIM}    trust={primitives.get('trust','?')}  "
                    f"intent={primitives.get('intent','?')}  "
                    f"narrative={primitives.get('narrative','?')}  "
                    f"commitments={primitives.get('commitments','?')}{RST}"
                )
        else:
            content = (resp.get("result") or {}).get("content") or []
            if content:
                snippet = content[0].get("text","")[:80].replace("\n"," ")
                print(f"  {DIM}↩  {snippet!r}…{RST}")

        await asyncio.sleep(PHASE_DELAY)
        return resp

    async def stop(self):
        await asyncio.sleep(0.4)
        if self._stderr_task: self._stderr_task.cancel()
        if self.proc:
            self.proc.stdin.close()
            try: await asyncio.wait_for(self.proc.wait(), timeout=4.0)
            except asyncio.TimeoutError: self.proc.terminate()

    # ── Display ──────────────────────────────────────────────────────

    def banner(self):
        threshold = os.environ.get("PACT_RUPTURE_THRESHOLD", "0.6")
        upstream  = os.environ.get("PACT_UPSTREAM_MODE", "docker")
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
║  Cursor ↔ GitHub MCP  •  StoryKeeper + RLP-0 live           ║
║  neurobloom.ai                                              ║
╚══════════════════════════════════════════════════════════════╝{RST}

  Repo             : {BOLD}{self.owner}/{self.repo}{RST}
  Upstream         : {BOLD}{upstream}{RST}
  RLP-0 threshold  : {BOLD}{threshold}{RST}  (rupture fires above this)

  {DIM}Each call shows:  coherence · trust · trajectory  │  rlp.rupture_risk · gate{RST}
""")

    def phase(self, n: int, title: str, desc: str):
        print(f"\n{B}{BOLD}{'─'*62}{RST}")
        print(f"{B}{BOLD}  Phase {n}: {title}{RST}")
        print(f"{DIM}  {desc}{RST}")
        print(f"{B}{BOLD}{'─'*62}{RST}")

    def summary(self):
        log = self.log
        risk_c = R if log.drift_risk == "HIGH" else Y if log.drift_risk == "MEDIUM" else G
        trust  = f"{log.sk_trust:.2f}" if log.sk_trust is not None else "—"
        rup    = f"{log.rlp_rupture_risk:.2f}" if log.rlp_rupture_risk is not None else "—"

        print(f"""
{BOLD}{'═'*62}
  SESSION SUMMARY
{'═'*62}{RST}
  Tool calls           : {self._calls}
  StoryKeeper alerts   : {R if log.drift_alerts else G}{len(log.drift_alerts)}{RST}
  RLP-0 rupture events : {R if log.rupture_events else G}{len(log.rupture_events)}{RST}
  SK drift risk        : {risk_c}{BOLD}{log.drift_risk}{RST}
  Final SK trust       : {trust}
  Final RLP-0 rupture  : {rup}
  Gate still closed    : {R+BOLD+"YES"+RST if log.rlp_gated else G+"no"+RST}

{BOLD}  TWO-LAYER INTEGRITY:{RST}
  StoryKeeper  → behavioral drift detection    (AX expression layer)
  RLP-0        → relational primitive gating   (rupture authority)

{BOLD}  THE GAP:{RST}
  Policy engine  : "Valid token, permitted scope"              {G}✓{RST}
  StoryKeeper    : "Coherence dropped — behavior drifted"      {Y}⚠{RST}
  RLP-0          : "trust+intent+narrative collapsed — GATE"   {R}✗{RST}
{BOLD}{'═'*62}{RST}
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(owner: str, repo: str):
    demo = Demo(owner, repo)
    demo.banner()

    print(f"{BOLD}Starting PACT-AX proxy…{RST}")
    await demo.start()
    await asyncio.sleep(BOOT_DELAY)
    await demo.health_check()

    print(f"{BOLD}MCP handshake…{RST}")
    name = await demo.handshake()
    print(f"  {G}✓  Connected:{RST} {name}\n")

    # ── Phase 1 ───────────────────────────────────────────────────
    demo.phase(1, "Establishing Normal Pattern",
        "Reading source files → repo:read:source pattern establishes, trust builds.")

    for path in ["proxy/src/proxy.py", "proxy/src/story_keeper.py",
                 "proxy/src/rlp_bridge.py", "proxy/demo.py"]:
        await demo.call("get_file_contents",
                        {"owner": owner, "repo": repo, "path": path})

    # ── Phase 2 ───────────────────────────────────────────────────
    demo.phase(2, "Minor Variation (Still Safe)",
        "Config files → slight narrative dip, but coherence holds.")

    await demo.call("get_file_contents",
                    {"owner": owner, "repo": repo, "path": "pyproject.toml"})
    await demo.call("get_file_contents",
                    {"owner": owner, "repo": repo, "path": "requirements.txt"})

    # ── Phase 3 ───────────────────────────────────────────────────
    demo.phase(3, "⚠  RLP-0 Rupture Zone",
        "Org access + sensitive search → trust·intent·narrative collapse → gate fires.")

    # This call triggers StoryKeeper drift (org:access:elevated)
    await demo.call("list_org_members", {"org": owner})

    # This call drives RLP-0 rupture_risk over threshold → gate closes
    await demo.call("search_code",
                    {"q": f"secret OR token OR password repo:{owner}/{repo}"})

    # These calls should now be BLOCKED by the RLP-0 gate
    print(f"\n  {DIM}Next two calls should be blocked by the closed gate…{RST}")

    await demo.call("get_file_contents",
                    {"owner": owner, "repo": repo, "path": ".env"})

    await demo.call("list_org_members", {"org": owner})

    # ── Summary ───────────────────────────────────────────────────
    demo.summary()
    await demo.stop()


def main():
    owner = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OWNER
    repo  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REPO

    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print(f"\n{R}✗  GITHUB_PERSONAL_ACCESS_TOKEN is not set.{RST}")
        print("   export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx\n")
        sys.exit(1)

    asyncio.run(run(owner, repo))


if __name__ == "__main__":
    main()
