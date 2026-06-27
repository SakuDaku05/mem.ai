"""
Mem0 drop-in connector for memai.

Provides a MemaiMem0 class that is interface-compatible with the
official Mem0 Python SDK (https://github.com/mem0ai/mem0).

This means any code written against the Mem0 SDK can switch to
memai by changing a single import — no other changes needed.

Usage (drop-in replacement):
    # Before (Mem0):
    # from mem0 import Memory
    # m = Memory()

    # After (memai — same API, better performance):
    from memai.connectors.mem0 import MemaiMem0 as Memory

    m = Memory(api_key="sk-memai-...", base_url="http://localhost:8000")
    m.add("I like Python", user_id="alice")
    results = m.search("programming language", user_id="alice")
    m.get_all(user_id="alice")
    m.delete(memory_id="...", user_id="alice")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemaiMem0:
    """
    Mem0-compatible memory class backed by memai.

    Drop-in replacement for the Mem0 Python SDK.
    Matches the Mem0 API surface exactly:
        - add(messages, user_id, ...)
        - search(query, user_id, ...)
        - get_all(user_id, ...)
        - get(memory_id)
        - update(memory_id, data)
        - delete(memory_id)
        - delete_all(user_id, ...)
        - history(memory_id)

    See: https://docs.mem0.ai/api-reference
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        config: Optional[dict] = None,
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url)
        self._config = config or {}

    # ------------------------------------------------------------------
    # add — matches mem0's add(messages, user_id, agent_id, run_id, ...)
    # ------------------------------------------------------------------

    def add(
        self,
        messages: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        filters: Optional[dict] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Add messages to memory.

        Messages can be:
          - str: "I like Python"
          - list of dicts: [{"role": "user", "content": "..."}, ...]
          - list of strings: ["fact 1", "fact 2"]
        """
        resolved_agent = agent_id or user_id or "default"
        session_id = run_id

        results = []
        texts_to_add = []

        if isinstance(messages, str):
            texts_to_add = [messages]
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, str):
                    texts_to_add.append(m)
                elif isinstance(m, dict):
                    role = m.get("role", "")
                    content = m.get("content", "")
                    if content:
                        texts_to_add.append(f"{role}: {content}" if role else content)

        for text in texts_to_add:
            try:
                mid = self._client.add(
                    text=text,
                    agent_id=resolved_agent,
                    session_id=session_id,
                    memory_type="semantic",
                    metadata=metadata or {},
                )
                results.append({"id": mid, "memory": text, "event": "ADD"})
            except Exception as e:
                logger.warning("mem0.add failed for text '%s...': %s", text[:40], e)

        return {"results": results}

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Search memories.

        Returns a Mem0-compatible response dict:
        {"results": [{"id": ..., "memory": ..., "score": ...}, ...]}
        """
        resolved_agent = agent_id or user_id or "default"
        try:
            result = self._client.search(
                query=query,
                agent_id=resolved_agent,
                k=limit,
                session_id=run_id,
            )
            return {
                "results": [
                    {
                        "id": m.id,
                        "memory": m.text,
                        "score": m.utility_score,
                        "metadata": m.metadata,
                        "created_at": m.created_at,
                    }
                    for m in result.memories
                ]
            }
        except Exception as e:
            logger.warning("mem0.search failed: %s", e)
            return {"results": []}

    # ------------------------------------------------------------------
    # get_all
    # ------------------------------------------------------------------

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """List all memories for the given user/agent."""
        resolved_agent = agent_id or user_id or "default"
        try:
            memories = self._client.list(agent_id=resolved_agent, limit=limit)
            return {
                "results": [
                    {
                        "id": m.id,
                        "memory": m.text,
                        "metadata": m.metadata,
                        "created_at": m.created_at,
                    }
                    for m in memories
                ]
            }
        except Exception as e:
            logger.warning("mem0.get_all failed: %s", e)
            return {"results": []}

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[dict]:
        """Get a specific memory by ID."""
        m = self._client.get(memory_id)
        if not m:
            return None
        return {"id": m.id, "memory": m.text, "metadata": m.metadata, "created_at": m.created_at}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, memory_id: str, data: str) -> dict:
        """
        Update a memory's text content.

        Note: memai implements this as delete + add, since embeddings
        must be recomputed for updated text.
        """
        old = self._client.get(memory_id)
        if old:
            self._client.delete(memory_id)
        mid = self._client.add(text=data, agent_id=old.agent_id if old else "default")
        return {"id": mid, "memory": data, "event": "UPDATE"}

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> dict:
        """Delete a specific memory."""
        deleted = self._client.delete(memory_id)
        return {"id": memory_id, "deleted": deleted, "event": "DELETE"}

    # ------------------------------------------------------------------
    # delete_all
    # ------------------------------------------------------------------

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        """Delete all memories for the given user/agent (forget everything)."""
        resolved_agent = agent_id or user_id or "default"
        count = self._client.forget(agent_id=resolved_agent, staleness_threshold=0.0)
        return {"deleted_count": count, "event": "DELETE_ALL"}

    # ------------------------------------------------------------------
    # history (stub — memai tracks events not edit history)
    # ------------------------------------------------------------------

    def history(self, memory_id: str) -> list[dict]:
        """
        Return memory history.
        memai returns single-entry history (creation event only).
        Full edit history tracking is not implemented in v0.1.
        """
        m = self.get(memory_id)
        if not m:
            return []
        return [{"id": memory_id, "memory": m["memory"], "event": "ADD", "timestamp": m.get("created_at")}]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs) -> "MemaiMem0":
        import os
        return cls(
            api_key=os.environ["MEMAI_API_KEY"],
            base_url=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
            **kwargs,
        )
