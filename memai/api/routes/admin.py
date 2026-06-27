"""
Admin routes — /v1/admin/*

GET    /health
GET    /metrics
POST   /sweep
GET    /agents
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from memai.api.auth import get_current_agent
from memai.api.manager import close_all, get_memory, list_agents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_start_time = time.time()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: str
    agents_loaded: int


class MetricsResponse(BaseModel):
    uptime_seconds: float
    agents_loaded: int
    agent_ids: list[str]


class SweepResponse(BaseModel):
    swept_agents: list[str]
    total_deleted: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health check",
    description="Returns server status. No auth required.",
    tags=["admin"],
)
async def health():
    from memai import __version__
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        agents_loaded=len(list_agents()),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Server metrics",
)
async def metrics(agent_id: str = Depends(get_current_agent)):
    agents = list_agents()
    return MetricsResponse(
        uptime_seconds=round(time.time() - _start_time, 2),
        agents_loaded=len(agents),
        agent_ids=agents,
    )


@router.post(
    "/sweep",
    response_model=SweepResponse,
    summary="Run staleness sweep across all agents",
    description="Triggers StalenessDetector sweep for all loaded agents. Deletes stale memories.",
)
async def sweep(agent_id: str = Depends(get_current_agent)):
    agents = list_agents()
    total_deleted = 0
    swept = []

    for aid in agents:
        mem = await get_memory(aid)
        result = await asyncio.to_thread(mem.forget, staleness_threshold=0.1)
        total_deleted += result.deleted_count
        swept.append(aid)

    return SweepResponse(swept_agents=swept, total_deleted=total_deleted)


@router.get(
    "/agents",
    summary="List loaded agents",
)
async def get_agents(agent_id: str = Depends(get_current_agent)):
    return {"agents": list_agents(), "total": len(list_agents())}
