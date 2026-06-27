"""
memai Python SDK — MemaiClient

Clean, batteries-included Python client for the memai REST API.
Supports both sync and async usage.

Quick start:
    from memai.sdk import MemaiClient

    mem = MemaiClient(api_key="sk-memai-...", base_url="http://localhost:8000")
    mem.add("User prefers concise answers", agent_id="agent-1")
    result = mem.search("user preferences", agent_id="agent-1")
    print(result.pami_context)   # inject directly into LLM prompt

    with mem.session("agent-1") as s:
        e1 = s.add_event("User opened dashboard")
        e2 = s.add_event("Error appeared", caused_by=e1.id)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclasses (no pydantic dep for SDK consumers)
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    id: str
    text: str
    agent_id: str
    memory_type: str
    session_id: Optional[str] = None
    utility_score: float = 0.5
    access_count: int = 0
    created_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            agent_id=d.get("agent_id", ""),
            memory_type=d.get("memory_type", "semantic"),
            session_id=d.get("session_id"),
            utility_score=d.get("utility_score", 0.5),
            access_count=d.get("access_count", 0),
            created_at=d.get("created_at"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SearchResult:
    memories: list[MemoryRecord]
    pami_context: str
    total_tokens_estimated: int
    dropped_count: int

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        return cls(
            memories=[MemoryRecord.from_dict(m) for m in d.get("memories", [])],
            pami_context=d.get("pami_context", ""),
            total_tokens_estimated=d.get("total_tokens_estimated", 0),
            dropped_count=d.get("dropped_count", 0),
        )


@dataclass
class EventRecord:
    id: str
    text: str
    agent_id: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    summary: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "EventRecord":
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            agent_id=d.get("agent_id", ""),
            session_id=d.get("session_id"),
            timestamp=d.get("timestamp"),
            entities=d.get("entities", []),
            summary=d.get("summary"),
        )


@dataclass
class WorkflowRecord:
    workflow_id: str
    name: str
    trigger_pattern: str
    steps: list[dict]
    agent_id: str
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowRecord":
        return cls(
            workflow_id=d.get("workflow_id", ""),
            name=d.get("name", ""),
            trigger_pattern=d.get("trigger_pattern", ""),
            steps=d.get("steps", []),
            agent_id=d.get("agent_id", ""),
            created_at=d.get("created_at"),
        )


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

class SessionContext:
    """Tracks events in a bounded session scope."""

    def __init__(self, client: "MemaiClient", agent_id: str, session_id: str):
        self._client = client
        self.agent_id = agent_id
        self.session_id = session_id
        self._events: list[EventRecord] = []

    def add_event(
        self,
        text: str,
        summary: Optional[str] = None,
        entities: Optional[list[str]] = None,
        caused_by: Optional[str] = None,
        contradicts: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> EventRecord:
        resp = self._client._request(
            "POST",
            f"/session/{self.session_id}/events",
            json={
                "text": text,
                "summary": summary,
                "entities": entities or [],
                "caused_by": caused_by,
                "contradicts": contradicts,
                "metadata": metadata or {},
            },
        )
        event = EventRecord.from_dict(resp)
        self._events.append(event)
        return event

    def timeline(self) -> list[EventRecord]:
        resp = self._client._request("GET", f"/session/{self.session_id}/timeline")
        return [EventRecord.from_dict(e) for e in resp.get("events", [])]

    def causal_chain(self, event_id: str, depth: int = 3) -> dict:
        return self._client._request(
            "GET",
            f"/session/{self.session_id}/causal/{event_id}",
            params={"depth": depth},
        )

    def compress(self, keep_ratio: float = 0.3) -> dict:
        return self._client._request(
            "POST",
            f"/session/{self.session_id}/compress",
            json={"keep_ratio": keep_ratio},
        )

    @property
    def local_events(self) -> list[EventRecord]:
        return self._events


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MemaiError(Exception):
    """Base exception for memai SDK errors."""


class MemaiAPIError(MemaiError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"memai API error {status_code}: {detail}")


class MemaiConnectionError(MemaiError):
    """Cannot connect to the memai server."""


# ---------------------------------------------------------------------------
# Main Client
# ---------------------------------------------------------------------------

class MemaiClient:
    """
    memai SDK client — wraps the REST API cleanly.

    Args:
        api_key:   Bearer API key (sk-memai-...)
        base_url:  memai server URL (default: http://localhost:8000)
        agent_id:  Default agent ID (can override per-call)
        timeout:   Request timeout seconds
    """

    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        agent_id: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required: pip install httpx")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/v1"
        self.default_agent_id = agent_id
        self.timeout = timeout

        import httpx as _httpx
        self._httpx = _httpx
        self._client = _httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            verify=verify_ssl,
        )

    # context manager support
    def __enter__(self) -> "MemaiClient":
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = self._client.request(method, path, **kwargs)
            r.raise_for_status()
            return r.json()
        except self._httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            raise MemaiAPIError(e.response.status_code, detail or str(e)) from e
        except self._httpx.RequestError as e:
            raise MemaiConnectionError(
                f"Cannot connect to {self.base_url}. Is memai running? {e}"
            ) from e

    def _agent(self, agent_id: Optional[str]) -> str:
        resolved = agent_id or self.default_agent_id
        if not resolved:
            raise ValueError(
                "agent_id required. Pass agent_id per-call or set MemaiClient(agent_id=...)"
            )
        return resolved

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/admin/health")

    def ping(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: str = "auto",
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a memory. Returns memory_id."""
        resp = self._request(
            "POST", "/memory/add",
            json={
                "text": text,
                "agent_id": self._agent(agent_id),
                "session_id": session_id,
                "memory_type": memory_type,
                "metadata": metadata or {},
            },
        )
        return resp["memory_id"]

    def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        k: int = 10,
        session_id: Optional[str] = None,
        context_budget: int = 2000,
    ) -> SearchResult:
        """Search memories. Returns SearchResult with .pami_context ready for LLM injection."""
        resp = self._request(
            "POST", "/memory/search",
            json={
                "query": query,
                "agent_id": self._agent(agent_id),
                "k": k,
                "session_id": session_id,
                "context_budget": context_budget,
            },
        )
        return SearchResult.from_dict(resp)

    def inject(
        self,
        query: str,
        agent_id: Optional[str] = None,
        k: int = 10,
        context_budget: int = 2000,
    ) -> str:
        """One-liner: search + return PAMI context string for direct LLM injection."""
        return self.search(query, agent_id=agent_id, k=k, context_budget=context_budget).pami_context

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        try:
            return MemoryRecord.from_dict(self._request("GET", f"/memory/{memory_id}"))
        except MemaiAPIError as e:
            if e.status_code == 404:
                return None
            raise

    def delete(self, memory_id: str) -> bool:
        try:
            self._request("DELETE", f"/memory/{memory_id}")
            return True
        except MemaiAPIError as e:
            if e.status_code == 404:
                return False
            raise

    def list(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        params: dict = {"limit": limit, "offset": offset}
        if memory_type:
            params["memory_type"] = memory_type
        resp = self._request("GET", "/memory", params=params)
        return [MemoryRecord.from_dict(m) for m in resp.get("memories", [])]

    def forget(
        self,
        agent_id: Optional[str] = None,
        older_than_days: Optional[int] = None,
        staleness_threshold: float = 0.1,
    ) -> int:
        """Delete stale memories. Returns count deleted."""
        resp = self._request(
            "POST", "/memory/forget",
            json={
                "agent_id": self._agent(agent_id),
                "older_than_days": older_than_days,
                "staleness_threshold": staleness_threshold,
            },
        )
        return resp.get("deleted_count", 0)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(self, agent_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        resp = self._request("POST", "/session/start", json={"session_id": session_id})
        return resp["session_id"]

    @contextmanager
    def session(self, agent_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Context manager for session-scoped event tracking.

            with mem.session("my-agent") as s:
                e1 = s.add_event("User clicked button")
                e2 = s.add_event("Modal opened", caused_by=e1.id)
        """
        sid = self.start_session(agent_id, session_id)
        yield SessionContext(self, self._agent(agent_id), sid)

    def get_timeline(self, session_id: str) -> list[EventRecord]:
        resp = self._request("GET", f"/session/{session_id}/timeline")
        return [EventRecord.from_dict(e) for e in resp.get("events", [])]

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def save_workflow(
        self,
        name: str,
        trigger_pattern: str,
        steps: list[dict],
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> WorkflowRecord:
        resp = self._request(
            "POST", "/workflow/save",
            json={
                "name": name,
                "trigger_pattern": trigger_pattern,
                "steps": steps,
                "metadata": metadata or {},
            },
        )
        return WorkflowRecord.from_dict(resp)

    def match_workflow(self, query: str, agent_id: Optional[str] = None) -> Optional[WorkflowRecord]:
        resp = self._request(
            "POST", "/workflow/match",
            json={"query": query, "agent_id": self._agent(agent_id)},
        )
        if resp.get("matched") and resp.get("workflow"):
            return WorkflowRecord.from_dict(resp["workflow"])
        return None

    def list_workflows(self, agent_id: Optional[str] = None) -> list[WorkflowRecord]:
        resp = self._request("GET", "/workflow/list")
        return [WorkflowRecord.from_dict(w) for w in resp.get("workflows", [])]

    def replay_workflow(self, workflow_id: str) -> list[dict]:
        return self._request("POST", f"/workflow/{workflow_id}/replay").get("steps", [])

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        return self._request("GET", "/admin/metrics")

    def sweep(self) -> dict:
        return self._request("POST", "/admin/sweep")

    # ------------------------------------------------------------------
    # Async wrappers (run sync methods in thread pool)
    # ------------------------------------------------------------------

    async def async_add(self, text: str, agent_id: Optional[str] = None, **kwargs) -> str:
        return await asyncio.to_thread(self.add, text, agent_id, **kwargs)

    async def async_search(self, query: str, agent_id: Optional[str] = None, **kwargs) -> SearchResult:
        return await asyncio.to_thread(self.search, query, agent_id, **kwargs)

    async def async_inject(self, query: str, agent_id: Optional[str] = None, **kwargs) -> str:
        return await asyncio.to_thread(self.inject, query, agent_id, **kwargs)

    async def async_forget(self, agent_id: Optional[str] = None, **kwargs) -> int:
        return await asyncio.to_thread(self.forget, agent_id, **kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs) -> "MemaiClient":
        """Create client from environment variables (MEMAI_API_KEY, MEMAI_BASE_URL, MEMAI_AGENT_ID)."""
        api_key = os.environ.get("MEMAI_API_KEY", "")
        if not api_key:
            raise ValueError("MEMAI_API_KEY environment variable is not set.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("MEMAI_BASE_URL", cls.DEFAULT_BASE_URL),
            agent_id=os.environ.get("MEMAI_AGENT_ID"),
            **kwargs,
        )
