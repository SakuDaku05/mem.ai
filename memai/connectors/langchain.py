"""
LangChain connector for memai.

Exposes memai as a LangChain BaseMemory subclass so it drops in
to any LangChain chain without changes.

Usage:
    from memai.connectors.langchain import MemaiMemory
    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI

    memory = MemaiMemory(
        api_key="sk-memai-...",
        agent_id="langchain-agent",
        base_url="http://localhost:8000",
    )

    chain = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory,
    )
    response = chain.run("What is my name?")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemaiMemory:
    """
    memai memory backend for LangChain.

    Compatible with LangChain's BaseMemory interface — provides
    load_memory_variables() and save_context() as required.

    Installs as a drop-in replacement for ConversationBufferMemory,
    ConversationSummaryMemory, or any other LangChain memory class.
    """

    memory_key: str = "history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    return_messages: bool = False

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        k: int = 10,
        context_budget: int = 2000,
        return_messages: bool = False,
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self.agent_id = agent_id
        self.session_id = session_id
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.k = k
        self.context_budget = context_budget
        self.return_messages = return_messages

        # Try to import LangChain for type hints (optional)
        try:
            from langchain.schema import BaseMemory  # noqa: F401
            self._lc_available = True
        except ImportError:
            self._lc_available = False
            logger.warning(
                "langchain not installed. MemaiMemory will work as a standalone memory object."
            )

    @property
    def memory_variables(self) -> list[str]:
        """LangChain interface: which keys this memory injects."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        LangChain interface: load relevant memories for the current input.
        Returns {memory_key: pami_context_string}
        """
        query = inputs.get(self.input_key or "input", "") or str(inputs)
        try:
            context = self._client.inject(
                query=query,
                agent_id=self.agent_id,
                k=self.k,
                context_budget=self.context_budget,
            )
        except Exception as e:
            logger.warning("memai load_memory_variables failed: %s", e)
            context = ""

        if self.return_messages:
            # Return as a list of HumanMessage-style dicts for chat models
            try:
                from langchain.schema import SystemMessage
                return {self.memory_key: [SystemMessage(content=context)] if context else []}
            except ImportError:
                pass
        return {self.memory_key: context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """
        LangChain interface: save the current interaction to memory.
        Stores both the user input and AI output as semantic memories.
        """
        user_text = inputs.get(self.input_key or "input", "")
        ai_text = outputs.get(self.output_key or "output", "")

        try:
            if user_text:
                self._client.add(
                    text=f"User: {user_text}",
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    memory_type="semantic",
                )
            if ai_text:
                self._client.add(
                    text=f"Assistant: {ai_text}",
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    memory_type="semantic",
                )
        except Exception as e:
            logger.warning("memai save_context failed: %s", e)

    def clear(self) -> None:
        """LangChain interface: clear all memories (runs staleness sweep)."""
        try:
            self._client.forget(agent_id=self.agent_id, staleness_threshold=0.0)
        except Exception as e:
            logger.warning("memai clear failed: %s", e)

    # Allow using as context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    # ------------------------------------------------------------------
    # Convenience: attach to existing chain
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, agent_id: str, **kwargs) -> "MemaiMemory":
        """Create from MEMAI_API_KEY and MEMAI_BASE_URL env vars."""
        import os
        return cls(
            api_key=os.environ["MEMAI_API_KEY"],
            agent_id=agent_id,
            base_url=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
            **kwargs,
        )
