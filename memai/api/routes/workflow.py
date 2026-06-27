"""
Workflow routes — /v1/workflow/*

POST   /workflow/save
POST   /workflow/match
GET    /workflow/{workflow_id}
GET    /workflow/list
DELETE /workflow/{workflow_id}
POST   /workflow/{workflow_id}/replay
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from memai.api.auth import get_current_agent
from memai.api.manager import get_memory
from memai.models import WorkflowItem, WorkflowStep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflow", tags=["workflow"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SaveWorkflowRequest(BaseModel):
    name: str
    trigger_pattern: str = Field(
        description="Regex or keyword pattern to match this workflow"
    )
    steps: list[dict] = Field(
        description='List of steps: [{"step": 1, "action": "...", "expected_output": "..."}]'
    )
    metadata: dict = Field(default_factory=dict)


class MatchWorkflowRequest(BaseModel):
    context: Optional[str] = Field(default=None, description="Current agent context to match against workflows")
    query: Optional[str] = Field(default=None, description="Alias for context")
    agent_id: Optional[str] = None
    k: int = 3

    @property
    def resolved_context(self) -> str:
        return self.context or self.query or ""


class MatchWorkflowResponse(BaseModel):
    matched: bool
    matches: list[WorkflowItem] = Field(default_factory=list)
    workflow: Optional[WorkflowItem] = None
    confidence: float = 0.0


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowItem]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/save",
    response_model=WorkflowItem,
    status_code=status.HTTP_201_CREATED,
    summary="Save a workflow",
    description="""
    Store a named workflow with a trigger pattern and ordered steps.
    The trigger_pattern can be a regex or keyword string.
    """,
)
async def save_workflow(
    request: SaveWorkflowRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    workflow = await asyncio.to_thread(
        mem.procedural_memory.save_workflow,
        name=request.name,
        agent_id=agent_id,
        trigger_pattern=request.trigger_pattern,
        steps=request.steps,
        metadata=request.metadata,
    )
    return workflow


@router.post(
    "/match",
    response_model=MatchWorkflowResponse,
    summary="Match context to a workflow",
    description="""
    Find the best matching workflow for the current agent context.
    Uses regex matching first, then keyword overlap as fallback.
    """,
)
async def match_workflow(
    request: MatchWorkflowRequest,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    result = await asyncio.to_thread(
        mem.procedural_memory.match_workflow,
        context=request.resolved_context,
        agent_id=agent_id,
    )
    if result is None:
        return MatchWorkflowResponse(matched=False, matches=[])
    workflow, confidence = result
    return MatchWorkflowResponse(
        matched=True,
        workflow=workflow,
        matches=[workflow],
        confidence=confidence,
    )


@router.get(
    "/list",
    response_model=WorkflowListResponse,
    summary="List all workflows",
)
async def list_workflows(
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    workflows = await asyncio.to_thread(
        mem.procedural_memory.list_workflows,
        agent_id=agent_id,
    )
    return WorkflowListResponse(workflows=workflows, total=len(workflows))


@router.get(
    "/{workflow_id}",
    response_model=WorkflowItem,
    summary="Get a specific workflow",
)
async def get_workflow(
    workflow_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    workflow = await asyncio.to_thread(
        mem.procedural_memory.get_workflow,
        workflow_id=workflow_id,
    )
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )
    return workflow


@router.delete(
    "/{workflow_id}",
    summary="Delete a workflow",
)
async def delete_workflow(
    workflow_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    deleted = await asyncio.to_thread(
        mem.procedural_memory.delete_workflow,
        workflow_id=workflow_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )
    return {"deleted": True, "workflow_id": workflow_id}


@router.post(
    "/{workflow_id}/replay",
    summary="Replay workflow steps",
    description="Execute a workflow step by step and record success.",
)
async def replay_workflow(
    workflow_id: str,
    agent_id: str = Depends(get_current_agent),
):
    mem = await get_memory(agent_id)
    steps = await asyncio.to_thread(
        mem.procedural_memory.replay,
        workflow_id=workflow_id,
    )
    if not steps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )
    return {
        "replayed": True,
        "workflow_id": workflow_id,
        "steps_executed": len(steps),
        "steps": [s.model_dump() for s in steps],
    }
