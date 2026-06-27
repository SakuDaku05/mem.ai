"""
LlamaIndex connector for memai.

Exposes memai as a LlamaIndex BaseMemory subclass and as a
custom retriever node, so agents using LlamaIndex can plug in
memai memory without any code changes.

Usage (as memory):
    from memai.connectors.llamaindex import MemaiChatMemoryBuffer

    from llama_index.core.chat_engine import SimpleChatEngine
    memory = MemaiChatMemoryBuffer(api_key="sk-memai-...", agent_id="li-agent")
    engine = SimpleChatEngine.from_defaults(memory=memory)

Usage (as retriever):
    from memai.connectors.llamaindex import MemaiRetriever
    retriever = MemaiRetriever(api_key="sk-memai-...", agent_id="li-agent")
    nodes = retriever.retrieve("what does the user prefer?")
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)


class MemaiRetriever:
    """
    LlamaIndex-compatible retriever backed by memai.

    Returns TextNode objects containing memories, compatible with
    LlamaIndex's retriever interface.

    Usage:
        retriever = MemaiRetriever(api_key="sk-memai-...", agent_id="my-agent")
        nodes = retriever.retrieve("what does the user prefer?")
        for node in nodes:
            print(node.text)
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        k: int = 10,
        context_budget: int = 2000,
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self.agent_id = agent_id
        self.k = k
        self.context_budget = context_budget

    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant memories for the given query.
        Returns a list of LlamaIndex NodeWithScore objects (if available)
        or plain dicts as fallback.
        """
        result = self._client.search(
            query=query,
            agent_id=self.agent_id,
            k=self.k,
            context_budget=self.context_budget,
        )

        nodes = []
        try:
            from llama_index.core.schema import TextNode, NodeWithScore
            for mem in result.memories:
                node = TextNode(
                    text=mem.text,
                    id_=mem.id,
                    metadata={
                        "agent_id": mem.agent_id,
                        "memory_type": mem.memory_type,
                        "session_id": mem.session_id or "",
                        "utility_score": mem.utility_score,
                    },
                )
                nodes.append(NodeWithScore(node=node, score=mem.utility_score))
        except ImportError:
            # Fallback: return plain dicts
            for mem in result.memories:
                nodes.append({"text": mem.text, "id": mem.id, "score": mem.utility_score})

        return nodes

    async def aretrieve(self, query: str) -> list:
        """Async version of retrieve()."""
        import asyncio
        return await asyncio.to_thread(self.retrieve, query)

    def get_pami_context(self, query: str) -> str:
        """Return the PAMI-formatted context string directly."""
        return self._client.inject(query=query, agent_id=self.agent_id, k=self.k)

    @classmethod
    def from_env(cls, agent_id: str, **kwargs) -> "MemaiRetriever":
        import os
        return cls(
            api_key=os.environ["MEMAI_API_KEY"],
            agent_id=agent_id,
            base_url=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
            **kwargs,
        )


class MemaiChatMemoryBuffer:
    """
    LlamaIndex ChatMemoryBuffer-compatible class backed by memai.

    Stores conversation turns as semantic memories and retrieves
    relevant context using PAMI on each new query.

    Usage:
        from llama_index.core.chat_engine import SimpleChatEngine
        memory = MemaiChatMemoryBuffer(api_key="sk-memai-...", agent_id="my-agent")
        engine = SimpleChatEngine.from_defaults(memory=memory)
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
        token_limit: int = 2000,
        k: int = 10,
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self.agent_id = agent_id
        self.session_id = session_id
        self.token_limit = token_limit
        self.k = k
        self._chat_history: list = []

    def put(self, message) -> None:
        """Store a chat message in memai. Accepts ChatMessage objects or dicts."""
        self._chat_history.append(message)
        try:
            role = getattr(message, "role", message.get("role", "unknown"))
            content = getattr(message, "content", message.get("content", ""))
            self._client.add(
                text=f"{role}: {content}",
                agent_id=self.agent_id,
                session_id=self.session_id,
                memory_type="semantic",
            )
        except Exception as e:
            logger.warning("memai put failed: %s", e)

    def get(self, input: Optional[str] = None, initial_token_count: int = 0) -> list:
        """
        LlamaIndex interface: get relevant messages for the current query.
        Returns list of ChatMessage objects (or plain dicts as fallback).
        """
        query = input or ""
        context = self._client.inject(
            query=query,
            agent_id=self.agent_id,
            k=self.k,
            context_budget=self.token_limit,
        )

        try:
            from llama_index.core.llms import ChatMessage, MessageRole
            if context:
                return [ChatMessage(role=MessageRole.SYSTEM, content=context)]
            return []
        except ImportError:
            if context:
                return [{"role": "system", "content": context}]
            return []

    def get_all(self) -> list:
        """Return all messages in local cache."""
        return self._chat_history

    def set(self, messages: list) -> None:
        """Bulk set messages (stores each to memai)."""
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """Clear local cache and run forget on memai."""
        self._chat_history.clear()
        try:
            self._client.forget(agent_id=self.agent_id, staleness_threshold=0.0)
        except Exception as e:
            logger.warning("memai reset failed: %s", e)

    @property
    def token_count(self) -> int:
        """Approximate token count for compatibility."""
        total_chars = sum(
            len(getattr(m, "content", m.get("content", "")))
            for m in self._chat_history
        )
        return total_chars // 4  # rough approximation

    @classmethod
    def from_env(cls, agent_id: str, **kwargs) -> "MemaiChatMemoryBuffer":
        import os
        return cls(
            api_key=os.environ["MEMAI_API_KEY"],
            agent_id=agent_id,
            base_url=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
            **kwargs,
        )
