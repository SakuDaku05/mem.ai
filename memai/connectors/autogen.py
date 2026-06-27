"""
AutoGen connector for memai.

Hooks into Microsoft AutoGen's agent memory system so that
any AutoGen agent can use memai as its persistent memory backend.

Two integration patterns:
  1. MemaiConversableAgent — subclass of ConversableAgent with built-in memory
  2. MemaiMiddleware — function middleware to inject memories into any agent's messages

Usage (pattern 1 — agent subclass):
    from memai.connectors.autogen import MemaiConversableAgent

    agent = MemaiConversableAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        api_key="sk-memai-...",
        agent_id="autogen-assistant",
    )

Usage (pattern 2 — middleware):
    from memai.connectors.autogen import MemaiMiddleware, inject_memory

    @inject_memory(api_key="sk-memai-...", agent_id="autogen-agent")
    def reply_func(messages, sender, **kwargs):
        # messages already have memai context prepended
        ...
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemaiConversableAgent:
    """
    AutoGen ConversableAgent subclass with memai memory.

    Automatically:
    - Retrieves relevant memories before each reply
    - Stores each conversation turn in memai
    - Injects PAMI context into the system message

    Usage:
        agent = MemaiConversableAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            api_key="sk-memai-...",
            agent_id="my-agent",
            llm_config={"model": "gpt-4o"},
        )
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        system_message: str = "You are a helpful assistant.",
        k: int = 10,
        context_budget: int = 1500,
        **autogen_kwargs,
    ):
        from memai.sdk import MemaiClient
        self._mem_client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self._agent_id = agent_id
        self._k = k
        self._context_budget = context_budget
        self._base_system_message = system_message
        self.name = name

        # Try to wrap actual AutoGen ConversableAgent
        try:
            import autogen
            self._agent = autogen.ConversableAgent(
                name=name,
                system_message=system_message,
                **autogen_kwargs,
            )
            self._autogen_available = True
            # Hook memory injection into the agent's reply function
            self._agent.register_reply(
                trigger=autogen.ConversableAgent,
                reply_func=self._memory_enhanced_reply,
                position=0,  # Run first — before default GPT reply
            )
            logger.info("MemaiConversableAgent initialized with AutoGen backend")
        except ImportError:
            self._agent = None
            self._autogen_available = False
            logger.warning(
                "autogen not installed. MemaiConversableAgent works as a memory-only wrapper. "
                "Install with: pip install pyautogen"
            )

    def _memory_enhanced_reply(self, messages, sender, config, **kwargs):
        """
        AutoGen reply hook: inject memai memory before generating a reply.
        Prepends PAMI context to the system message for the current turn.
        """
        # Extract the latest user message as search query
        latest_msg = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                latest_msg = last.get("content", "")
            else:
                latest_msg = str(last)

        # Retrieve memories
        context = ""
        if latest_msg:
            try:
                context = self._mem_client.inject(
                    query=latest_msg,
                    agent_id=self._agent_id,
                    k=self._k,
                    context_budget=self._context_budget,
                )
            except Exception as e:
                logger.warning("memai memory retrieval failed: %s", e)

        # Update system message with memory context
        if context and self._agent:
            enhanced_system = f"{self._base_system_message}\n\n{context}"
            self._agent.update_system_message(enhanced_system)

        # Store the user message
        if latest_msg:
            try:
                self._mem_client.add(
                    text=latest_msg,
                    agent_id=self._agent_id,
                    memory_type="semantic",
                )
            except Exception as e:
                logger.warning("memai memory store failed: %s", e)

        # Return False to allow the default reply function to continue
        return False, None

    def store_reply(self, reply_text: str) -> None:
        """Store an agent reply as a memory (call after generation)."""
        try:
            self._mem_client.add(
                text=f"Assistant replied: {reply_text}",
                agent_id=self._agent_id,
                memory_type="semantic",
            )
        except Exception as e:
            logger.warning("memai store_reply failed: %s", e)

    def __getattr__(self, name: str):
        """Proxy all other attribute access to the underlying AutoGen agent."""
        if self._agent is not None:
            return getattr(self._agent, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Middleware / decorator approach
# ---------------------------------------------------------------------------

def inject_memory(
    api_key: str,
    agent_id: str,
    base_url: str = "http://localhost:8000",
    k: int = 10,
    context_budget: int = 1500,
):
    """
    Decorator to inject memai memories into an AutoGen reply function.

    Usage:
        @inject_memory(api_key="sk-memai-...", agent_id="my-agent")
        def reply_func(messages, sender, **kwargs):
            # messages[0] now contains a system message with PAMI context
            return True, generate_reply(messages)
    """
    from memai.sdk import MemaiClient
    client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)

    def decorator(func):
        @wraps(func)
        def wrapper(messages, sender, **kwargs):
            # Get query from latest message
            latest_msg = ""
            if messages:
                last = messages[-1]
                latest_msg = last.get("content", "") if isinstance(last, dict) else str(last)

            # Inject memories as first system message
            if latest_msg:
                try:
                    context = client.inject(
                        query=latest_msg,
                        agent_id=agent_id,
                        k=k,
                        context_budget=context_budget,
                    )
                    if context:
                        memory_msg = {"role": "system", "content": context}
                        messages = [memory_msg] + list(messages)
                except Exception as e:
                    logger.warning("inject_memory decorator failed: %s", e)

            return func(messages, sender, **kwargs)
        return wrapper
    return decorator


class MemaiGroupChatManager:
    """
    Wraps AutoGen GroupChatManager to store and retrieve shared memories
    across all agents in a group chat.

    Usage:
        from memai.connectors.autogen import MemaiGroupChatManager
        manager = MemaiGroupChatManager(
            groupchat=groupchat,
            api_key="sk-memai-...",
            agent_id="group-chat",
        )
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        k: int = 10,
        **autogen_kwargs,
    ):
        from memai.sdk import MemaiClient
        self._mem = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self._agent_id = agent_id
        self._k = k

        try:
            import autogen
            self._manager = autogen.GroupChatManager(**autogen_kwargs)
            self._autogen_available = True
        except ImportError:
            self._manager = None
            self._autogen_available = False

    def store_message(self, sender_name: str, content: str) -> None:
        """Store a group chat message in memai."""
        try:
            self._mem.add(
                text=f"[{sender_name}]: {content}",
                agent_id=self._agent_id,
                memory_type="semantic",
            )
        except Exception as e:
            logger.warning("MemaiGroupChatManager.store_message failed: %s", e)

    def get_context(self, query: str) -> str:
        """Retrieve PAMI context for the group chat."""
        return self._mem.inject(query=query, agent_id=self._agent_id, k=self._k)

    def __getattr__(self, name: str):
        if self._manager is not None:
            return getattr(self._manager, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
