"""
Session routes — /v1/session/*

POST   /session/start
GET    /session/{session_id}/timeline
POST   /session/{session_id}/compress
GET    /session/{session_id}/summary
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from memai.api.auth import get_current_agent
from memai.api.manager import get_memory
from memai.models import EventNode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class StartSessionResponse(BaseModel):
    session_id: str
    agent_id: str


class AddEventRequest(BaseModel):
    text: str
    summary: Optional[str] = None
    entities: Optional[list[str]] = None
    caused_by: Optional[str] = None
    contradicts: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class CompressRequest(BaseModel):
    keep_ratio: float = Field(default=0.3, ge=0.01, le=1.0)


class CompressResponse(BaseModel):
    compressed_count: int
    session_id: str


class TimelineResponse(BaseModel):
    session_id: str
    events: list[EventNode]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/start",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new session",
)
async def start_session(
    request: StartSessionRequest,
    agent_id: str = Depends(get_current_agent),
):
    import uuid
    session_id = request.session_id or str(uuid.uuid4())
    return StartSessionResponse(session_id=session_id, agent_id=agent_id)


@router.post(
    "/{session_id}/events",
    response_model=EventNode,
    status_code=status.HTTP_201_CREATED,
    summary="Add an event to a session",
)
async def add_session_event(
    session_id: str,
    request: AddEventRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    node = await asyncio.to_thread(
        mem.event_memory.add_event,
        text=request.text,
        agent_id=agent_id,
        session_id=session_id,
        summary=request.summary,
        entities=request.entities,
        metadata=request.metadata,
        caused_by=request.caused_by,
        contradicts=request.contradicts,
    )
    return node


@router.get(
    "/{session_id}/timeline",
    response_model=TimelineResponse,
    summary="Get session event timeline",
    description="Returns all events in a session ordered by timestamp (chronological).",
)
async def get_session_timeline(
    session_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    events = await asyncio.to_thread(
        mem.event_memory.get_session_timeline,
        session_id=session_id,
        agent_id=agent_id,
    )
    return TimelineResponse(
        session_id=session_id,
        events=events,
        total=len(events),
    )


@router.post(
    "/{session_id}/compress",
    response_model=CompressResponse,
    summary="Compress old session events",
    description="""
    Remove low-utility events from a session to reduce storage.
    Causal links are preserved by keeping the most recent events.
    `keep_ratio=0.3` means keep the 30% most recent events.
    """,
)
async def compress_session(
    session_id: str,
    request: CompressRequest,
    agent_id: str = Depends(get_current_agent),
):
    from datetime import datetime, timezone
    mem = await get_memory(agent_id)
    cutoff = datetime.now(timezone.utc)
    count = await asyncio.to_thread(
        mem.event_memory.temporal_compress,
        agent_id=agent_id,
        before_timestamp=cutoff,
        keep_ratio=request.keep_ratio,
    )
    return CompressResponse(compressed_count=count, session_id=session_id)


@router.get(
    "/{session_id}/causal/{event_id}",
    summary="Get causal chain from an event",
    description="Traverse CAUSES edges up to depth hops from the given event.",
)
async def get_causal_chain(
    session_id: str,
    event_id: str,
    depth: int = 3,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    chain = await asyncio.to_thread(
        mem.event_memory.get_causal_chain,
        root_id=event_id,
        depth=depth,
    )
    return {
        "root_id": chain.root_id,
        "nodes": [n.model_dump() for n in chain.nodes],
        "depth": chain.depth,
        "total": len(chain.nodes),
    }
