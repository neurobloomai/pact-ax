"""
pact_ax/api/routes/seam_observer.py
─────────────────────────────────────
SSE endpoint that streams Seam Observer events to the browser.
GET  /seam/events   — SSE stream
GET  /seam/snapshot — current agent/edge state
POST /seam/reset    — clear bus (between demo runs)
"""

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pact_ax.observability.event_bus import get_bus

router = APIRouter(prefix="/seam", tags=["seam-observer"])


@router.get("/events", summary="SSE stream of all seam events")
async def stream_events():
    bus = get_bus()

    async def generate():
        # send current snapshot so the browser can draw existing state
        yield f"data: {json.dumps({'type': 'snapshot', 'payload': bus.snapshot()})}\n\n"
        index = bus.total()
        while True:
            for event in bus.events_since(index):
                yield f"data: {json.dumps({'type': 'event', 'payload': event.to_dict()})}\n\n"
                index += 1
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/snapshot", summary="Current agent graph state")
def get_snapshot() -> Dict[str, Any]:
    return get_bus().snapshot()


@router.post("/reset", summary="Clear the event bus between runs")
def reset_bus() -> Dict[str, Any]:
    get_bus().reset()
    return {"reset": True}
