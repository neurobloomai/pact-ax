"""
pact_ax/orchestration/orchestrator.py
───────────────────────────────────────
Async multi-agent fan-out with conditional routing.

Two patterns:
  fan_out_parallel    — fire N tasks simultaneously, wait for all
  fan_out_conditional — fire first task, read result, decide what to fire next

Uses httpx.AsyncClient with ASGI transport so no network hop is needed
when running against the in-process FastAPI app.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx


@dataclass
class OrchestratorTask:
    to_agent:    str
    payload:     Dict[str, Any]
    system_prompt: str = ""
    user_prompt:   str = ""
    task_id:     str = ""

    def __post_init__(self):
        if not self.task_id:
            self.task_id = self.to_agent


@dataclass
class TaskResult:
    task_id:    str
    to_agent:   str
    response:   str
    packet_id:  str = ""
    ret_packet_id: str = ""
    cancelled:  bool = False


class Orchestrator:
    """
    Async orchestrator — wraps StateTransfer + EventBus instrumentation.

    Usage:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            orch = Orchestrator(from_agent_id="agent-a", client=client)
            results = await orch.fan_out_parallel(tasks, llm_fn)
    """

    def __init__(self, from_agent_id: str, client: httpx.AsyncClient):
        self._from  = from_agent_id
        self._client = client

    # ── low-level packet helpers ──────────────────────────────────────────────

    async def _post(self, path: str, **kwargs) -> Dict:
        r = await self._client.post(path, **kwargs)
        r.raise_for_status()
        return r.json()

    async def _get(self, path: str) -> Dict:
        r = await self._client.get(path)
        r.raise_for_status()
        return r.json()

    async def _send_packet(self, to_agent: str, payload: Dict,
                           reason: str = "escalation") -> Tuple[str, Dict]:
        """Prepare → Send → Receive. Returns (packet_id, send_dict)."""
        prep = await self._post("/transfer/prepare", json={
            "from_agent_id": self._from,
            "to_agent_id":   to_agent,
            "reason":        reason,
            "state_data":    payload,
        })
        pid  = prep["packet_id"]
        sent = await self._post("/transfer/send",    json={"agent_id": self._from, "packet_id": pid})
        await self._post("/transfer/receive",        json={"agent_id": to_agent,   "packet": sent})
        return pid, sent

    async def _return_packet(self, from_agent: str, result: Dict) -> str:
        """Sub-agent hands result back to orchestrator."""
        prep = await self._post("/transfer/prepare", json={
            "from_agent_id": from_agent,
            "to_agent_id":   self._from,
            "reason":        "completion",
            "state_data":    result,
        })
        pid  = prep["packet_id"]
        sent = await self._post("/transfer/send",    json={"agent_id": from_agent, "packet_id": pid})
        await self._post("/transfer/receive",        json={"agent_id": self._from, "packet": sent})
        return pid

    # ── core execution ────────────────────────────────────────────────────────

    async def _run_task(
        self,
        task:   OrchestratorTask,
        llm_fn: Callable[[str, str], Awaitable[str]],
    ) -> TaskResult:
        """Fire one task: send packet → await LLM → return packet."""
        pid, _ = await self._send_packet(task.to_agent, task.payload)
        response = await llm_fn(task.system_prompt, task.user_prompt)
        ret_pid  = await self._return_packet(task.to_agent, {"result": response, "agent": task.to_agent})
        return TaskResult(task_id=task.task_id, to_agent=task.to_agent,
                          response=response, packet_id=pid, ret_packet_id=ret_pid)

    # ── public patterns ───────────────────────────────────────────────────────

    async def fan_out_parallel(
        self,
        tasks:  List[OrchestratorTask],
        llm_fn: Callable[[str, str], Awaitable[str]],
    ) -> List[TaskResult]:
        """
        Fire all tasks simultaneously via asyncio.gather.
        All tasks run in parallel and all results are returned.
        """
        return list(await asyncio.gather(
            *[self._run_task(t, llm_fn) for t in tasks]
        ))

    async def fan_out_conditional(
        self,
        first_task: OrchestratorTask,
        llm_fn:     Callable[[str, str], Awaitable[str]],
        route_fn:   Callable[[TaskResult], Optional[List[OrchestratorTask]]],
    ) -> Tuple[TaskResult, List[TaskResult]]:
        """
        Fire first_task, read its result, call route_fn to decide next tasks.

        route_fn(first_result) → list of next tasks, or None/[] to stop.

        Returns (first_result, subsequent_results).
        """
        first_result = await self._run_task(first_task, llm_fn)
        next_tasks   = route_fn(first_result) or []
        if not next_tasks:
            return first_result, []
        rest = await self.fan_out_parallel(next_tasks, llm_fn)
        return first_result, rest

    async def fan_out_race(
        self,
        tasks:    List[OrchestratorTask],
        llm_fn:   Callable[[str, str], Awaitable[str]],
        stop_fn:  Callable[[TaskResult], bool],
    ) -> Tuple[TaskResult, List[TaskResult]]:
        """
        Fire all tasks simultaneously. If stop_fn(result) is True for any
        result, cancel remaining tasks and return immediately.

        Returns (winning_result, completed_before_cancel).
        """
        loop    = asyncio.get_event_loop()
        pending = {loop.create_task(self._run_task(t, llm_fn)): t for t in tasks}
        done_results: List[TaskResult] = []
        winner: Optional[TaskResult]  = None

        remaining = set(pending.keys())
        while remaining:
            finished, remaining = await asyncio.wait(
                remaining, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in finished:
                result = fut.result()
                if stop_fn(result) and winner is None:
                    winner = result
                    for r in remaining:
                        r.cancel()
                    remaining = set()
                    break
                done_results.append(result)

        return winner or done_results[0], done_results
