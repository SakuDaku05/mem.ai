"""
Memory manager — shared singleton across the API.

Manages one Memory instance per agent_id, cached in a process-level dict.
Thread-safe for async FastAPI (uses asyncio lock).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from memai.api.config import get_settings
from memai.memory import Memory

logger = logging.getLogger(__name__)

_agents: dict[str, Memory] = {}
_lock = asyncio.Lock()


async def get_memory(agent_id: str) -> Memory:
    """Get or create a Memory instance for the given agent_id."""
    if agent_id in _agents:
        return _agents[agent_id]

    async with _lock:
        # Double-check inside lock
        if agent_id in _agents:
            return _agents[agent_id]

        settings = get_settings()
        logger.info("Creating new Memory instance for agent_id=%s", agent_id)
        mem = Memory(
            agent_id=agent_id,
            data_dir=settings.data_dir,
            graph_backend=settings.graph_backend,
            vector_backend=settings.vector_backend,
            embedding_model=settings.embedding_model,
            token_budget=settings.default_token_budget,
            decay_lambda=settings.decay_lambda,
        )
        _agents[agent_id] = mem
        return mem


def get_memory_sync(agent_id: str) -> Memory:
    """Synchronous version — for use outside async context."""
    if agent_id in _agents:
        return _agents[agent_id]
    settings = get_settings()
    mem = Memory(
        agent_id=agent_id,
        data_dir=settings.data_dir,
        graph_backend=settings.graph_backend,
        vector_backend=settings.vector_backend,
        embedding_model=settings.embedding_model,
    )
    _agents[agent_id] = mem
    return mem


async def close_all() -> None:
    """Close all Memory instances (called on shutdown)."""
    for agent_id, mem in _agents.items():
        try:
            mem.close()
            logger.info("Closed memory for agent %s", agent_id)
        except Exception as e:
            logger.warning("Error closing memory for agent %s: %s", agent_id, e)
    _agents.clear()


def list_agents() -> list[str]:
    return list(_agents.keys())
