"""
PACT-AX MCP Server
==================
Exposes PACT-AX trust primitives as MCP tools so any MCP-native client
(Claude Code, Cursor, jkoelker-style servers) can call trust, routing,
capability, memory, and handoff operations directly.

Usage
-----
    python -m pact_ax.mcp.server

Claude Code / Cursor config (~/.claude/settings.json or .cursor/mcp.json):
    {
      "mcpServers": {
        "pact-ax": {
          "command": "python",
          "args": ["-m", "pact_ax.mcp.server"],
          "env": { "PACT_AX_URL": "http://localhost:8000" }
        }
      }
    }

Environment
-----------
    PACT_AX_URL   Base URL of PACT-AX server (default: http://localhost:8000)

Tools exposed (12)
------------------
    Trust         trust_get, trust_update, trust_network, trust_insights, trust_agents
    Routing       route_task, route_any
    Capabilities  capability_register, capability_find
    Memory        memory_record, memory_recall
    Handoff       transfer_prepare
"""

import asyncio
import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

PACT_AX_URL = os.getenv("PACT_AX_URL", "http://localhost:8000")

server = Server("pact-ax")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── Trust ─────────────────────────────────────────────────────────────
        types.Tool(
            name="trust_get",
            description="Get trust score between two agents. Returns overall score and per-context breakdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "The agent querying trust"},
                    "target_id": {"type": "string", "description": "The agent being evaluated"},
                },
                "required": ["agent_id", "target_id"],
            },
        ),
        types.Tool(
            name="trust_update",
            description="Record a collaboration outcome and update trust score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":     {"type": "string", "description": "Agent recording the outcome"},
                    "target_id":    {"type": "string", "description": "Agent being rated"},
                    "outcome":      {"type": "string", "enum": ["positive", "negative", "neutral", "partial"]},
                    "context_type": {"type": "string", "default": "task_knowledge"},
                    "impact":       {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
                },
                "required": ["agent_id", "target_id", "outcome"],
            },
        ),
        types.Tool(
            name="trust_network",
            description="Get transitive network trust for an indirect or unknown agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":  {"type": "string"},
                    "target_id": {"type": "string"},
                },
                "required": ["agent_id", "target_id"],
            },
        ),
        types.Tool(
            name="trust_insights",
            description="Get full trust relationship insights for an agent — all known scores, context breakdown, and history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        ),
        types.Tool(
            name="trust_agents",
            description="List all agents trusted above a minimum threshold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":     {"type": "string"},
                    "min_trust":    {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                    "context_type": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        ),
        # ── Routing ───────────────────────────────────────────────────────────
        types.Tool(
            name="route_task",
            description="Route a task to the best trusted+capable agent for an exact skill name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_agent": {"type": "string"},
                    "skill":      {"type": "string", "description": "Exact capability name to match"},
                    "min_trust":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                    "top_k":      {"type": "integer", "default": 5},
                },
                "required": ["from_agent", "skill"],
            },
        ),
        types.Tool(
            name="route_any",
            description="Route a task by fuzzy keyword search across all registered capabilities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_agent": {"type": "string"},
                    "query":      {"type": "string", "description": "Free-text keyword to match against capabilities"},
                    "min_trust":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                    "top_k":      {"type": "integer", "default": 5},
                },
                "required": ["from_agent", "query"],
            },
        ),
        # ── Capabilities ──────────────────────────────────────────────────────
        types.Tool(
            name="capability_register",
            description="Register a skill/capability for an agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":    {"type": "string"},
                    "skill":       {"type": "string"},
                    "description": {"type": "string", "default": ""},
                    "tags":        {"type": "array", "items": {"type": "string"}, "default": []},
                    "version":     {"type": "string", "default": "1.0"},
                },
                "required": ["agent_id", "skill"],
            },
        ),
        types.Tool(
            name="capability_find",
            description="Find agents registered for a specific skill.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill":     {"type": "string"},
                    "min_trust": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "requester": {"type": "string", "description": "Required if min_trust is set"},
                },
                "required": ["skill"],
            },
        ),
        # ── Episodic Memory ───────────────────────────────────────────────────
        types.Tool(
            name="memory_record",
            description="Record an episodic interaction for an agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":   {"type": "string"},
                    "action":     {"type": "string"},
                    "partner_id": {"type": "string"},
                    "outcome":    {"type": "string", "enum": ["positive", "negative", "neutral", "partial"]},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                    "tags":       {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["agent_id", "action", "outcome"],
            },
        ),
        types.Tool(
            name="memory_recall",
            description="Recall past episodes for an agent with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id":       {"type": "string"},
                    "partner_id":     {"type": "string"},
                    "outcome":        {"type": "string"},
                    "min_importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "limit":          {"type": "integer", "default": 20},
                },
                "required": ["agent_id"],
            },
        ),
        # ── Trust Chain ───────────────────────────────────────────────────────
        types.Tool(
            name="trust_chain_score",
            description=(
                "Score a chain of agents (A→B→C) for relational coherence. "
                "Returns chain_trust (geometric mean across hops), coherence "
                "(variance across hops), state (active/degraded/broken), and "
                "the weakest hop. Does not record the chain."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "description": "Ordered list of agent IDs forming the chain",
                    },
                },
                "required": ["agents"],
            },
        ),
        types.Tool(
            name="trust_chain_verify",
            description=(
                "Re-verify a recorded trust chain against current trust scores. "
                "Returns per-hop drift, whether state changed "
                "(e.g. active→degraded), and updated chain_trust."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chain_id": {
                        "type": "string",
                        "description": "chain_id from /trust-chain/record",
                    },
                },
                "required": ["chain_id"],
            },
        ),
        # ── State Transfer ────────────────────────────────────────────────────
        types.Tool(
            name="transfer_prepare",
            description="Prepare a state handoff packet from one agent to another.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_agent": {"type": "string"},
                    "to_agent":   {"type": "string"},
                    "state_data": {"type": "object", "description": "State payload to transfer"},
                    "reason":     {"type": "string", "default": "continuation"},
                },
                "required": ["from_agent", "to_agent", "state_data"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    async with httpx.AsyncClient(base_url=PACT_AX_URL, timeout=30.0) as client:
        try:
            result = await _dispatch(client, name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except httpx.HTTPStatusError as e:
            return [types.TextContent(
                type="text",
                text=f"PACT-AX error {e.response.status_code}: {e.response.text}",
            )]
        except httpx.ConnectError:
            return [types.TextContent(
                type="text",
                text=f"Cannot connect to PACT-AX at {PACT_AX_URL} — is the server running?",
            )]


async def _dispatch(client: httpx.AsyncClient, name: str, args: dict):
    if name == "trust_get":
        r = await client.get(f"/trust/{args['agent_id']}/{args['target_id']}")
        r.raise_for_status()
        return r.json()

    elif name == "trust_update":
        r = await client.post(f"/trust/{args['agent_id']}/update", json={
            "target_id":    args["target_id"],
            "outcome":      args["outcome"],
            "context_type": args.get("context_type", "task_knowledge"),
            "impact":       args.get("impact", 1.0),
        })
        r.raise_for_status()
        return r.json()

    elif name == "trust_network":
        r = await client.get(f"/trust/{args['agent_id']}/network/{args['target_id']}")
        r.raise_for_status()
        return r.json()

    elif name == "trust_insights":
        r = await client.get(f"/trust/{args['agent_id']}/insights")
        r.raise_for_status()
        return r.json()

    elif name == "trust_agents":
        body = {"min_trust": args.get("min_trust", 0.6)}
        if args.get("context_type"):
            body["context_type"] = args["context_type"]
        r = await client.post(f"/trust/{args['agent_id']}/agents", json=body)
        r.raise_for_status()
        return r.json()

    elif name == "route_task":
        r = await client.post("/route", json={
            "from_agent": args["from_agent"],
            "skill":      args["skill"],
            "min_trust":  args.get("min_trust", 0.0),
            "top_k":      args.get("top_k", 5),
        })
        r.raise_for_status()
        return r.json()

    elif name == "route_any":
        r = await client.post("/route/any", json={
            "from_agent": args["from_agent"],
            "query":      args["query"],
            "min_trust":  args.get("min_trust", 0.0),
            "top_k":      args.get("top_k", 5),
        })
        r.raise_for_status()
        return r.json()

    elif name == "capability_register":
        r = await client.post("/capabilities/register", json={
            "agent_id":    args["agent_id"],
            "skill":       args["skill"],
            "description": args.get("description", ""),
            "tags":        args.get("tags", []),
            "version":     args.get("version", "1.0"),
        })
        r.raise_for_status()
        return r.json()

    elif name == "capability_find":
        body = {"skill": args["skill"]}
        if args.get("min_trust") is not None:
            body["min_trust"] = args["min_trust"]
        if args.get("requester"):
            body["requester"] = args["requester"]
        r = await client.post("/capabilities/find", json=body)
        r.raise_for_status()
        return r.json()

    elif name == "memory_record":
        body = {
            "action":     args["action"],
            "outcome":    args["outcome"],
            "importance": args.get("importance", 0.5),
            "tags":       args.get("tags", []),
        }
        if args.get("partner_id"):
            body["partner_id"] = args["partner_id"]
        r = await client.post(f"/memory/episodes/{args['agent_id']}", json=body)
        r.raise_for_status()
        return r.json()

    elif name == "memory_recall":
        params = {"limit": args.get("limit", 20)}
        if args.get("partner_id"):
            params["partner_id"] = args["partner_id"]
        if args.get("outcome"):
            params["outcome"] = args["outcome"]
        if args.get("min_importance") is not None:
            params["min_importance"] = args["min_importance"]
        r = await client.get(f"/memory/episodes/{args['agent_id']}", params=params)
        r.raise_for_status()
        return r.json()

    elif name == "trust_chain_score":
        r = await client.post("/trust-chain/score", json={"agents": args["agents"]})
        r.raise_for_status()
        return r.json()

    elif name == "trust_chain_verify":
        r = await client.post(f"/trust-chain/{args['chain_id']}/verify")
        r.raise_for_status()
        return r.json()

    elif name == "transfer_prepare":
        r = await client.post("/transfer/prepare", json={
            "from_agent": args["from_agent"],
            "to_agent":   args["to_agent"],
            "state_data": args["state_data"],
            "reason":     args.get("reason", "continuation"),
        })
        r.raise_for_status()
        return r.json()

    else:
        return {"error": f"Unknown tool: {name}"}


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
