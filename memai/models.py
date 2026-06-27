"""
Shared data models for memai.
All Pydantic schemas used across modules live here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    EVENT = "event"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    AUTO = "auto"


class StalenessReason(str, Enum):
    TIME_DECAY = "time_decay"
    CONTRADICTION = "contradiction"
    SUPERSEDED = "superseded"
    DORMANT = "dormant"
    NONE = "none"


class EdgeType(str, Enum):
    CAUSES = "CAUSES"
    PRECEDES = "PRECEDES"
    CONTRADICTS = "CONTRADICTS"
    UPDATES = "UPDATES"
    REFERENCES = "REFERENCES"


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class StalenessResult(BaseModel):
    is_stale: bool = False
    reason: StalenessReason = StalenessReason.NONE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    adjusted_score: float = Field(default=1.0, ge=0.0)


# ---------------------------------------------------------------------------
# Core Memory Item
# ---------------------------------------------------------------------------

class MemoryItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: Optional[list[float]] = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    session_id: Optional[str] = None
    agent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    utility_score: float = 0.5
    staleness: StalenessResult = Field(default_factory=StalenessResult)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Graph
# ---------------------------------------------------------------------------

class EventNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    agent_id: str
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalChain(BaseModel):
    root_id: str
    nodes: list[EventNode]
    edges: list[EventEdge]
    depth: int


# ---------------------------------------------------------------------------
# Procedural Memory
# ---------------------------------------------------------------------------

class WorkflowStep(BaseModel):
    step: int
    action: str
    expected_output: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowItem(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    trigger_pattern: str
    steps: list[WorkflowStep]
    agent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None
    success_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search & Retrieval
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    memories: list[MemoryItem]
    pami_context: str          # Ready-to-inject prompt string
    total_tokens_estimated: int
    dropped_count: int         # How many were dropped by PAMI budget


class SearchRequest(BaseModel):
    query: str
    agent_id: str
    k: int = 10
    session_id: Optional[str] = None
    context_budget: int = 2000   # token budget for PAMI
    memory_types: list[MemoryType] = Field(
        default_factory=lambda: [MemoryType.EVENT, MemoryType.SEMANTIC]
    )
    filters: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------

class AddMemoryRequest(BaseModel):
    text: str
    agent_id: str
    session_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.AUTO
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddMemoryResponse(BaseModel):
    memory_id: str
    type_inferred: MemoryType


class ForgetRequest(BaseModel):
    agent_id: str
    older_than_days: Optional[int] = None
    staleness_threshold: float = 0.1


class ForgetResponse(BaseModel):
    deleted_count: int
