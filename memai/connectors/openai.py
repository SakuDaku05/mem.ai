"""
OpenAI / Codex connector for memai.

Wraps the OpenAI Python SDK to inject memai memories as a system
message before each completion call.

Usage:
    from memai.connectors.openai import MemaiOpenAI

    client = MemaiOpenAI(
        openai_api_key="sk-openai-...",
        memai_api_key="sk-memai-...",
        agent_id="openai-agent",
    )

    # Works exactly like openai.OpenAI() but with automatic memory injection
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What do I prefer?"}],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemaiOpenAI:
    """
    OpenAI client wrapper with automatic memai memory injection.

    All chat completion calls are intercepted to:
    1. Search memai for relevant memories using the user's last message
    2. Prepend the PAMI context as a system message
    3. Store the response back into memai

    Args:
        openai_api_key:  Your OpenAI API key
        memai_api_key:   Your memai API key
        agent_id:        Agent identifier for memory scoping
        base_url:        memai server URL
        k:               Max memories to retrieve per call
        context_budget:  Max tokens for PAMI context
        auto_store:      Whether to automatically store AI replies
    """

    def __init__(
        self,
        openai_api_key: str,
        memai_api_key: str,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        k: int = 10,
        context_budget: int = 1500,
        auto_store: bool = True,
        **openai_kwargs,
    ):
        try:
            import openai
            self._openai = openai.OpenAI(api_key=openai_api_key, **openai_kwargs)
        except ImportError:
            raise ImportError("openai is required: pip install openai")

        from memai.sdk import MemaiClient
        self._mem = MemaiClient(api_key=memai_api_key, base_url=base_url, agent_id=agent_id)
        self._agent_id = agent_id
        self._k = k
        self._context_budget = context_budget
        self._auto_store = auto_store
        self.chat = _MemaiChatCompletions(self)

    def __getattr__(self, name: str):
        """Proxy non-chat attributes to the underlying OpenAI client."""
        return getattr(self._openai, name)


class _MemaiChatCompletions:
    """Inner class mimicking openai.chat.completions with memory injection."""

    def __init__(self, parent: "MemaiOpenAI"):
        self._parent = parent

    def create(self, messages: list[dict], **kwargs) -> Any:
        """
        Intercept chat completion — inject memai context, call OpenAI, store reply.
        """
        p = self._parent

        # 1. Extract latest user message as search query
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # 2. Retrieve and inject PAMI context
        enhanced_messages = list(messages)
        if user_content:
            try:
                context = p._mem.inject(
                    query=user_content,
                    agent_id=p._agent_id,
                    k=p._k,
                    context_budget=p._context_budget,
                )
                if context:
                    # Insert memory context as first system message
                    memory_msg = {"role": "system", "content": context}
                    # Find existing system message and append, or prepend
                    if enhanced_messages and enhanced_messages[0].get("role") == "system":
                        existing = enhanced_messages[0]["content"]
                        enhanced_messages[0] = {
                            "role": "system",
                            "content": f"{existing}\n\n{context}",
                        }
                    else:
                        enhanced_messages = [memory_msg] + enhanced_messages
            except Exception as e:
                logger.warning("memai memory injection failed: %s", e)

        # 3. Call OpenAI
        response = p._openai.chat.completions.create(
            messages=enhanced_messages, **kwargs
        )

        # 4. Store the user turn and AI reply
        if p._auto_store and user_content:
            try:
                p._mem.add(text=f"User: {user_content}", agent_id=p._agent_id)
                reply_text = response.choices[0].message.content
                if reply_text:
                    p._mem.add(text=f"Assistant: {reply_text}", agent_id=p._agent_id)
            except Exception as e:
                logger.warning("memai auto-store failed: %s", e)

        return response

    async def acreate(self, messages: list[dict], **kwargs) -> Any:
        """Async version."""
        import asyncio
        return await asyncio.to_thread(self.create, messages, **kwargs)


# ---------------------------------------------------------------------------
# Standalone function wrappers (for users who don't want a class)
# ---------------------------------------------------------------------------

def patch_openai_client(
    client,
    memai_api_key: str,
    agent_id: str,
    base_url: str = "http://localhost:8000",
    k: int = 10,
    context_budget: int = 1500,
    auto_store: bool = True,
) -> None:
    """
    Monkey-patch an existing OpenAI client to inject memai memories.

    Usage:
        import openai
        from memai.connectors.openai import patch_openai_client

        client = openai.OpenAI(api_key="sk-openai-...")
        patch_openai_client(client, memai_api_key="sk-memai-...", agent_id="my-agent")

        # Now all client.chat.completions.create() calls have memai memory
    """
    from memai.sdk import MemaiClient
    mem = MemaiClient(api_key=memai_api_key, base_url=base_url, agent_id=agent_id)
    original_create = client.chat.completions.create

    def patched_create(messages, **kwargs):
        user_content = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        enhanced = list(messages)
        if user_content:
            try:
                ctx = mem.inject(query=user_content, agent_id=agent_id, k=k, context_budget=context_budget)
                if ctx:
                    if enhanced and enhanced[0].get("role") == "system":
                        enhanced[0]["content"] += f"\n\n{ctx}"
                    else:
                        enhanced = [{"role": "system", "content": ctx}] + enhanced
            except Exception as e:
                logger.warning("patch_openai_client inject failed: %s", e)

        response = original_create(enhanced, **kwargs)
        if auto_store and user_content:
            try:
                mem.add(f"User: {user_content}", agent_id=agent_id)
                reply = response.choices[0].message.content
                if reply:
                    mem.add(f"Assistant: {reply}", agent_id=agent_id)
            except Exception:
                pass
        return response

    client.chat.completions.create = patched_create
    logger.info("memai memory injected into OpenAI client for agent_id=%s", agent_id)
