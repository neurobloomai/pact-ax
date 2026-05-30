"""
pact_ax/api/routes/orchestrate.py
───────────────────────────────────
REST wrapper around the Orchestrator class.

Three endpoints mirror the three fan-out patterns:

  POST /orchestrate/parallel    — fire all tasks simultaneously, return all results
  POST /orchestrate/conditional — run first task, route_fn decides what to run next
  POST /orchestrate/race        — first task whose response contains stop_keyword wins

LLM execution
─────────────
Tasks carry system_prompt + user_prompt. The endpoint calls Claude
(claude-haiku-4-5 by default, override with PACT_ORCHESTRATE_MODEL).
Set ANTHROPIC_API_KEY to enable real LLM calls. Without it the server
runs in stub mode and echoes back the system prompt prefix + task_id.

In-process transport
────────────────────
All StateTransfer / Trust calls from the Orchestrator go through
httpx.ASGITransport so there is zero network overhead.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pact_ax.observability.event_bus import get_bus
from pact_ax.orchestration.orchestrator import Orchestrator, OrchestratorTask, TaskResult

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])

_MODEL = os.getenv("PACT_ORCHESTRATE_MODEL", "claude-haiku-4-5-20251001")
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call_claude_sync(system: str, user: str) -> str:
    if not _API_KEY:
        return f"[stub] system={system[:60]!r} user={user[:60]!r}"
    import anthropic
    client = anthropic.Anthropic(api_key=_API_KEY)
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


async def _llm_fn(system: str, user: str) -> str:
    return await asyncio.to_thread(_call_claude_sync, system, user)


# ── Request / Response models ─────────────────────────────────────────────────

class TaskRequest(BaseModel):
    to_agent:      str = Field(..., min_length=1)
    payload:       Dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = Field("", description="System prompt for the LLM")
    user_prompt:   str = Field("", description="User prompt for the LLM")
    task_id:       str = Field("", description="Optional task identifier; defaults to to_agent")


class ParallelRequest(BaseModel):
    from_agent: str = Field(..., min_length=1)
    tasks:      List[TaskRequest] = Field(..., min_length=1)


class ConditionalRequest(BaseModel):
    from_agent:  str = Field(..., min_length=1)
    first_task:  TaskRequest
    next_tasks:  List[TaskRequest] = Field(
        default_factory=list,
        description="Tasks to run if first result contains any stop_keyword. "
                    "If empty the route_fn always stops after first_task.",
    )
    route_keyword: str = Field(
        "",
        description="If set, next_tasks fire only when the first result contains this keyword (case-insensitive). "
                    "Leave empty to always fire next_tasks.",
    )


class RaceRequest(BaseModel):
    from_agent:   str = Field(..., min_length=1)
    tasks:        List[TaskRequest] = Field(..., min_length=1)
    stop_keyword: str = Field(
        "",
        description="Cancel remaining tasks when any result contains this keyword (case-insensitive). "
                    "Leave empty to stop on the very first completed result.",
    )


def _to_task(r: TaskRequest) -> OrchestratorTask:
    return OrchestratorTask(
        to_agent=r.to_agent,
        payload=r.payload,
        system_prompt=r.system_prompt,
        user_prompt=r.user_prompt,
        task_id=r.task_id or r.to_agent,
    )


def _result_dict(r: TaskResult) -> Dict[str, Any]:
    return {
        "task_id":      r.task_id,
        "to_agent":     r.to_agent,
        "response":     r.response,
        "packet_id":    r.packet_id,
        "ret_packet_id": r.ret_packet_id,
        "cancelled":    r.cancelled,
    }


# ── shared orchestrator factory ───────────────────────────────────────────────

def _get_app():
    from pact_ax.api.server import app  # lazy — avoids circular import at module level
    return app


def _make_orchestrator(from_agent: str, client: httpx.AsyncClient) -> Orchestrator:
    return Orchestrator(from_agent_id=from_agent, client=client)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/parallel", summary="Fan-out: run all tasks in parallel, return all results")
async def parallel(req: ParallelRequest) -> Dict[str, Any]:
    """
    Fire all tasks simultaneously via asyncio.gather.
    All tasks run in parallel; all results are returned.
    """
    transport = httpx.ASGITransport(app=_get_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        orch = _make_orchestrator(req.from_agent, client)
        tasks = [_to_task(t) for t in req.tasks]
        try:
            results = await orch.fan_out_parallel(tasks, _llm_fn)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    get_bus().emit("orchestrate_parallel",
                   from_agent=req.from_agent, task_count=len(tasks))
    return {
        "pattern":    "parallel",
        "from_agent": req.from_agent,
        "results":    [_result_dict(r) for r in results],
        "count":      len(results),
    }


@router.post("/conditional", summary="Fan-out: run first task, conditionally route to next tasks")
async def conditional(req: ConditionalRequest) -> Dict[str, Any]:
    """
    Run first_task, inspect result, fire next_tasks if route_keyword is found
    (or always fire if route_keyword is empty).
    """
    keyword = req.route_keyword.lower()

    def route_fn(first: TaskResult) -> Optional[List[OrchestratorTask]]:
        if not req.next_tasks:
            return []
        if keyword and keyword not in first.response.lower():
            return []
        return [_to_task(t) for t in req.next_tasks]

    transport = httpx.ASGITransport(app=_get_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        orch = _make_orchestrator(req.from_agent, client)
        try:
            first_result, rest = await orch.fan_out_conditional(
                _to_task(req.first_task), _llm_fn, route_fn
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    get_bus().emit("orchestrate_conditional",
                   from_agent=req.from_agent,
                   routed=len(rest) > 0,
                   next_count=len(rest))
    return {
        "pattern":      "conditional",
        "from_agent":   req.from_agent,
        "first_result": _result_dict(first_result),
        "routed":       len(rest) > 0,
        "next_results": [_result_dict(r) for r in rest],
    }


@router.post("/race", summary="Fan-out: first task matching stop_keyword wins, rest cancelled")
async def race(req: RaceRequest) -> Dict[str, Any]:
    """
    Fire all tasks simultaneously. The first result whose response contains
    stop_keyword (or the first completed result if stop_keyword is empty)
    wins — remaining tasks are cancelled.
    """
    keyword = req.stop_keyword.lower()

    def stop_fn(result: TaskResult) -> bool:
        if not keyword:
            return True
        return keyword in result.response.lower()

    transport = httpx.ASGITransport(app=_get_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        orch = _make_orchestrator(req.from_agent, client)
        tasks = [_to_task(t) for t in req.tasks]
        try:
            winner, completed = await orch.fan_out_race(tasks, _llm_fn, stop_fn)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    get_bus().emit("orchestrate_race",
                   from_agent=req.from_agent,
                   winner=winner.task_id,
                   completed_before_cancel=len(completed))
    return {
        "pattern":                 "race",
        "from_agent":              req.from_agent,
        "winner":                  _result_dict(winner),
        "completed_before_cancel": [_result_dict(r) for r in completed],
    }
