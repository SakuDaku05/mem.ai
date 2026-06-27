"""
ProceduralMemory — SQLite-backed workflow store.

Stores named workflows as ordered step sequences and matches them
to current context using trigger patterns (regex or embedding similarity).

Answers BEAM abilities:
  - Instruction Following (sustained adherence to constraints across turns)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from memai.models import WorkflowItem, WorkflowStep

logger = logging.getLogger(__name__)


class ProceduralMemory:
    """
    Workflow storage and replay engine.

    Usage:
        pm = ProceduralMemory(db_path="./memai_data/procedural.db")

        pm.save_workflow(
            name="deploy_flow",
            agent_id="a1",
            trigger_pattern=r"deploy|release|push to prod",
            steps=[
                {"step": 1, "action": "Run tests", "expected_output": "All pass"},
                {"step": 2, "action": "Build Docker image"},
                {"step": 3, "action": "Push to registry"},
                {"step": 4, "action": "Apply k8s manifest"},
            ]
        )

        match = pm.match_workflow(current_context="we need to deploy now", agent_id="a1")
        if match:
            for step in match.steps:
                print(f"Step {step.step}: {step.action}")
    """

    def __init__(self, db_path: str = "./memai_data/procedural.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("ProceduralMemory initialized at %s", self.db_path)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id     TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL,
                steps_json      TEXT NOT NULL,
                agent_id        TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                last_used_at    TEXT,
                success_count   INTEGER DEFAULT 0,
                metadata_json   TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_workflows_agent
                ON workflows (agent_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def save_workflow(
        self,
        name: str,
        agent_id: str,
        trigger_pattern: str,
        steps: list[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkflowItem:
        """Save or update a named workflow."""
        step_objects = [
            WorkflowStep(
                step=s.get("step", i + 1),
                action=s["action"],
                expected_output=s.get("expected_output"),
                metadata=s.get("metadata", {}),
            )
            for i, s in enumerate(steps)
        ]
        workflow = WorkflowItem(
            name=name,
            agent_id=agent_id,
            trigger_pattern=trigger_pattern,
            steps=step_objects,
            metadata=metadata or {},
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO workflows
                (workflow_id, name, trigger_pattern, steps_json, agent_id,
                 created_at, last_used_at, success_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow.workflow_id,
                workflow.name,
                workflow.trigger_pattern,
                json.dumps([s.model_dump() for s in workflow.steps]),
                workflow.agent_id,
                workflow.created_at.isoformat(),
                None,
                workflow.success_count,
                json.dumps(workflow.metadata),
            ),
        )
        self._conn.commit()
        logger.debug("Saved workflow '%s' for agent %s", name, agent_id)
        return workflow

    def match_workflow(
        self,
        context: str,
        agent_id: str,
    ) -> Optional[tuple[WorkflowItem, float]]:
        """
        Find the best matching workflow for the current context.

        Matching strategy:
          1. Regex match against trigger_pattern (exact match, score=1.0)
          2. Keyword overlap fallback (score = overlap ratio)

        Returns (WorkflowItem, confidence_score) or None.
        """
        workflows = self.list_workflows(agent_id)
        if not workflows:
            return None

        best: Optional[WorkflowItem] = None
        best_score: float = 0.0

        context_lower = context.lower()
        context_words = set(context_lower.split())

        for wf in workflows:
            # Strategy 1: regex
            try:
                if re.search(wf.trigger_pattern, context, re.IGNORECASE):
                    score = 1.0
                    if score > best_score:
                        best_score = score
                        best = wf
                    continue
            except re.error:
                pass  # Invalid regex — fall through to keyword overlap

            # Strategy 2: keyword overlap
            pattern_words = set(wf.trigger_pattern.lower().split())
            overlap = len(context_words & pattern_words) / max(len(pattern_words), 1)
            if overlap > best_score:
                best_score = overlap
                best = wf

        if best and best_score > 0.0:
            # Update last_used_at
            self._conn.execute(
                "UPDATE workflows SET last_used_at=? WHERE workflow_id=?",
                (datetime.now(timezone.utc).isoformat(), best.workflow_id),
            )
            self._conn.commit()
            return best, best_score

        return None

    def record_success(self, workflow_id: str) -> None:
        """Increment success count for a workflow."""
        self._conn.execute(
            "UPDATE workflows SET success_count = success_count + 1 WHERE workflow_id = ?",
            (workflow_id,),
        )
        self._conn.commit()

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowItem]:
        """Get a workflow by ID."""
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        return self._row_to_workflow(row) if row else None

    def get_by_name(self, name: str, agent_id: str) -> Optional[WorkflowItem]:
        """Get a workflow by name for a given agent."""
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE name = ? AND agent_id = ?",
            (name, agent_id),
        ).fetchone()
        return self._row_to_workflow(row) if row else None

    def list_workflows(self, agent_id: str) -> list[WorkflowItem]:
        """List all workflows for an agent."""
        rows = self._conn.execute(
            "SELECT * FROM workflows WHERE agent_id = ? ORDER BY success_count DESC",
            (agent_id,),
        ).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow by ID."""
        cursor = self._conn.execute(
            "DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def replay(
        self,
        workflow_id: str,
        callback: Optional[Any] = None,
    ) -> list[WorkflowStep]:
        """
        Replay a workflow step by step.
        If callback is provided, calls callback(step) for each step.
        Returns the steps list for inspection.
        """
        wf = self.get_workflow(workflow_id)
        if not wf:
            logger.warning("Workflow %s not found", workflow_id)
            return []
        for step in wf.steps:
            logger.info("Replaying step %d: %s", step.step, step.action)
            if callback:
                callback(step)
        self.record_success(workflow_id)
        return wf.steps

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _row_to_workflow(self, row: sqlite3.Row) -> WorkflowItem:
        steps_raw = json.loads(row["steps_json"])
        steps = [WorkflowStep(**s) for s in steps_raw]
        meta = json.loads(row["metadata_json"] or "{}")
        last_used = row["last_used_at"]
        return WorkflowItem(
            workflow_id=row["workflow_id"],
            name=row["name"],
            trigger_pattern=row["trigger_pattern"],
            steps=steps,
            agent_id=row["agent_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_used_at=datetime.fromisoformat(last_used) if last_used else None,
            success_count=row["success_count"],
            metadata=meta,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
