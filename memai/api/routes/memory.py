"""
Memory routes — /v1/memory/*

POST   /memory/add
POST   /memory/search
GET    /memory/{memory_id}
DELETE /memory/{memory_id}
POST   /memory/forget
GET    /memory/list
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from memai.api.auth import get_current_agent
from memai.api.manager import get_memory
from memai.models import (
    AddMemoryRequest,
    AddMemoryResponse,
    ForgetRequest,
    ForgetResponse,
    MemoryItem,
    MemoryType,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

@router.post(
    "/add",
    response_model=AddMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a memory",
    description="""
    Add a new memory for an agent. The memory type can be specified explicitly
    or set to 'auto' to let memai infer it (event vs semantic vs procedural).
    """,
)
async def add_memory(
    request: AddMemoryRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    # Run in thread pool to avoid blocking async loop
    result = await asyncio.to_thread(
        mem.add,
        text=request.text,
        memory_type=request.memory_type,
        session_id=request.session_id,
        metadata=request.metadata,
    )
    return result


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@router.post(
    "/search",
    response_model=SearchResult,
    summary="Search memories with PAMI context injection",
    description="""
    Search memories by semantic similarity, re-ranked by composite utility score.
    Returns a PAMI-formatted context string ready for direct injection into LLM prompts.

    The `pami_context` field in the response can be used directly as your
    system prompt memory section — memories are positioned to avoid the
    "lost in the middle" problem.
    """,
)
async def search_memories(
    request: SearchRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    result = await asyncio.to_thread(
        mem.search,
        query=request.query,
        k=request.k,
        session_id=request.session_id,
        context_budget=request.context_budget,
    )
    return result


# ---------------------------------------------------------------------------
# Get by ID
# ---------------------------------------------------------------------------

@router.get(
    "/{memory_id}",
    response_model=MemoryItem,
    summary="Get a specific memory",
)
async def get_memory_by_id(
    memory_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    item = await asyncio.to_thread(mem.get, memory_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )
    return item


# ---------------------------------------------------------------------------
# Delete by ID
# ---------------------------------------------------------------------------

@router.delete(
    "/{memory_id}",
    summary="Delete a specific memory",
)
async def delete_memory(
    memory_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    deleted = await asyncio.to_thread(mem.delete, memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )
    return {"deleted": True, "memory_id": memory_id}


# ---------------------------------------------------------------------------
# Forget (bulk delete stale memories)
# ---------------------------------------------------------------------------

@router.post(
    "/forget",
    response_model=ForgetResponse,
    summary="Delete stale or old memories",
    description="""
    Run the StalenessDetector sweep and delete memories below the staleness
    threshold. Optionally also compress old event memories.
    """,
)
async def forget_memories(
    request: ForgetRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    result = await asyncio.to_thread(
        mem.forget,
        older_than_days=request.older_than_days,
        staleness_threshold=request.staleness_threshold,
    )
    return result


# ---------------------------------------------------------------------------
# List all memories for agent
# ---------------------------------------------------------------------------

class ListMemoriesResponse(BaseModel):
    memories: list[MemoryItem]
    total: int
    limit: int
    offset: int


@router.get(
    "",
    response_model=ListMemoriesResponse,
    summary="List all memories for the current agent",
)
async def list_memories(
    agent_id: str = Depends(get_current_agent),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    memory_type: Optional[MemoryType] = Query(default=None),
):
    mem = await get_memory(agent_id)
    items = await asyncio.to_thread(
        mem.semantic_memory.list_agent,
        agent_id=agent_id,
        limit=limit,
        offset=offset,
    )
    if memory_type:
        items = [i for i in items if i.memory_type == memory_type]
    total = await asyncio.to_thread(mem.count)
    return ListMemoriesResponse(
        memories=items,
        total=total,
        limit=limit,
        offset=offset,
    )
