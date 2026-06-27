"""
Generic LLM middleware for memai.

Framework-agnostic memory injection that works with any LLM SDK.
Provides two patterns:

1. Functional (wrap any completion function):
    from memai.connectors.generic import with_memory

    @with_memory(api_key="sk-memai-...", agent_id="my-agent")
    def generate(messages):
        return call_any_llm(messages)

2. Class-based (MemaiMiddleware):
    from memai.connectors.generic import MemaiMiddleware

    middleware = MemaiMiddleware(api_key="sk-memai-...", agent_id="my-agent")
    enhanced_messages = middleware.before(messages, query="user query")
    response = call_any_llm(enhanced_messages)
    middleware.after(query="user query", response="assistant reply")
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MemaiMiddleware:
    """
    Generic middleware for injecting memai into any LLM pipeline.

    Use before() to inject memories, after() to store replies.
    Works with any message format: list[dict], str, or custom objects.

    Usage:
        mid = MemaiMiddleware(api_key="sk-memai-...", agent_id="my-agent")

        # Before LLM call
        messages = mid.before(messages, query=user_input)

        # After LLM call
        response_text = llm(messages)
        mid.after(query=user_input, response=response_text)
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        k: int = 10,
        context_budget: int = 2000,
        auto_store: bool = True,
        system_role: str = "system",
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)
        self.agent_id = agent_id
        self.k = k
        self.context_budget = context_budget
        self.auto_store = auto_store
        self.system_role = system_role

    def before(
        self,
        messages: list[dict],
        query: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Inject relevant memories into messages before the LLM call.

        Prepends a system message containing the PAMI context.
        If a system message already exists, memory is appended to it.
        """
        search_query = query or self._extract_query(messages)
        if not search_query:
            return messages

        try:
            context = self._client.inject(
                query=search_query,
                agent_id=self.agent_id,
                k=self.k,
                context_budget=self.context_budget,
            )
        except Exception as e:
            logger.warning("MemaiMiddleware.before failed: %s", e)
            return messages

        if not context:
            return messages

        enhanced = list(messages)
        if enhanced and enhanced[0].get("role") == self.system_role:
            enhanced[0] = {
                **enhanced[0],
                "content": enhanced[0]["content"] + f"\n\n{context}",
            }
        else:
            enhanced = [{"role": self.system_role, "content": context}] + enhanced

        return enhanced

    def after(
        self,
        query: str,
        response: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Store the interaction in memai after the LLM call."""
        if not self.auto_store:
            return
        try:
            if query:
                self._client.add(f"User: {query}", agent_id=self.agent_id, session_id=session_id)
            if response:
                self._client.add(f"Assistant: {response}", agent_id=self.agent_id, session_id=session_id)
        except Exception as e:
            logger.warning("MemaiMiddleware.after failed: %s", e)

    def _extract_query(self, messages: list[dict]) -> str:
        """Extract the last user message as the search query."""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def get_context(self, query: str) -> str:
        """Get PAMI context for a query without modifying messages."""
        return self._client.inject(query=query, agent_id=self.agent_id, k=self.k)

    def store(self, text: str, memory_type: str = "auto", session_id: Optional[str] = None) -> str:
        """Directly store a memory. Returns memory_id."""
        return self._client.add(text=text, agent_id=self.agent_id, memory_type=memory_type, session_id=session_id)


# ---------------------------------------------------------------------------
# Functional decorator
# ---------------------------------------------------------------------------

def with_memory(
    api_key: str,
    agent_id: str,
    base_url: str = "http://localhost:8000",
    k: int = 10,
    context_budget: int = 2000,
    auto_store: bool = True,
    messages_arg: str = "messages",
    query_extractor: Optional[Callable] = None,
    response_extractor: Optional[Callable] = None,
):
    """
    Decorator — wraps any LLM function to inject memai memories.

    The decorated function must accept a 'messages' argument (list[dict])
    and return a response string or dict.

    Usage:
        @with_memory(api_key="sk-memai-...", agent_id="my-agent")
        def call_llm(messages):
            return openai_client.chat.completions.create(messages=messages)

    Custom extractors:
        @with_memory(
            api_key="...",
            agent_id="...",
            query_extractor=lambda msgs: msgs[-1]["content"],
            response_extractor=lambda r: r.choices[0].message.content,
        )
        def call_llm(messages): ...
    """
    middleware = MemaiMiddleware(
        api_key=api_key,
        agent_id=agent_id,
        base_url=base_url,
        k=k,
        context_budget=context_budget,
        auto_store=auto_store,
    )

    def _extract_query(messages: list) -> str:
        if query_extractor:
            return query_extractor(messages)
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _extract_response(result: Any) -> str:
        if response_extractor:
            return response_extractor(result)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("content", "") or result.get("text", "") or str(result)
        # OpenAI-style
        try:
            return result.choices[0].message.content
        except Exception:
            return str(result)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract messages
            messages = kwargs.get(messages_arg)
            if messages is None and args:
                messages = args[0]

            query = _extract_query(messages or [])

            # Inject memories
            if messages is not None:
                enhanced = middleware.before(messages, query=query)
                if messages_arg in kwargs:
                    kwargs[messages_arg] = enhanced
                elif args:
                    args = (enhanced,) + args[1:]

            # Call original function
            result = func(*args, **kwargs)

            # Store response
            if auto_store:
                response_text = _extract_response(result)
                middleware.after(query=query, response=response_text)

            return result
        return wrapper
    return decorator
