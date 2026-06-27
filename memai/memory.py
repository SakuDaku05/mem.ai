"""
Memory — Main orchestrator for memai.

Unifies EventMemory, SemanticMemory, ProceduralMemory,
StalenessDetector, UtilityScorer, and PAMI into a single
clean API.

Usage:
    from memai import Memory

    # Embedded (local) mode
    mem = Memory(agent_id="my-agent")

    # Add a memory (auto-routes to semantic or event memory)
    mem.add("User prefers dark mode and uses Python 3.11")

    # Search — returns PAMI-ready context string
    result = mem.search("what does the user prefer?", context_budget=2000)
    print(result.pami_context)  # inject directly into LLM prompt

    # Session mode
    with mem.session("session-123") as s:
        s.add_event("User asked about indexing")
        s.add_event("Agent recommended B-tree")
        timeline = s.timeline()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

from memai.core.event_memory import EventMemory
from memai.core.pami import PAMI
from memai.core.procedural_memory import ProceduralMemory
from memai.core.semantic_memory import SemanticMemory
from memai.core.staleness_detector import StalenessDetector
from memai.core.utility_scorer import UtilityScorer
from memai.models import (
    AddMemoryResponse,
    EventNode,
    ForgetResponse,
    MemoryItem,
    MemoryType,
    SearchResult,
    WorkflowItem,
)

logger = logging.getLogger(__name__)


class SessionContext:
    """
    Context manager for session-scoped memory operations.

    Usage:
        with mem.session("session-123") as s:
            s.add_event("...")
            timeline = s.timeline()
    """

    def __init__(self, memory: "Memory", session_id: str) -> None:
        self._memory = memory
        self.session_id = session_id

    def add_event(
        self,
        text: str,
        summary: Optional[str] = None,
        entities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        causes: Optional[str] = None,
        precedes: Optional[str] = None,
    ) -> EventNode:
        return self._memory.event_memory.add_event(
            text=text,
            agent_id=self._memory.agent_id,
            session_id=self.session_id,
            summary=summary,
            entities=entities,
            metadata=metadata,
            causes=causes,
            precedes=precedes,
        )

    def timeline(self) -> list[EventNode]:
        return self._memory.event_memory.get_session_timeline(
            self.session_id, self._memory.agent_id
        )

    def add(self, text: str, **kwargs) -> MemoryItem:
        return self._memory.add(text, session_id=self.session_id, **kwargs)

    def search(self, query: str, **kwargs) -> SearchResult:
        return self._memory.search(query, session_id=self.session_id, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class WorkflowProxy:
    """Proxy for procedural memory operations, accessible via mem.workflows.*"""

    def __init__(self, memory: "Memory") -> None:
        self._memory = memory

    def save(
        self,
        name: str,
        steps: list[dict[str, Any]],
        trigger_pattern: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkflowItem:
        return self._memory.procedural_memory.save_workflow(
            name=name,
            agent_id=self._memory.agent_id,
            trigger_pattern=trigger_pattern,
            steps=steps,
            metadata=metadata,
        )

    def match(self, context: str):
        return self._memory.procedural_memory.match_workflow(
            context=context,
            agent_id=self._memory.agent_id,
        )

    def list(self) -> list[WorkflowItem]:
        return self._memory.procedural_memory.list_workflows(self._memory.agent_id)

    def get(self, workflow_id: str) -> Optional[WorkflowItem]:
        return self._memory.procedural_memory.get_workflow(workflow_id)

    def delete(self, workflow_id: str) -> bool:
        return self._memory.procedural_memory.delete_workflow(workflow_id)


class Memory:
    """
    memai — Unified Agentic Memory.

    The single entrypoint for all memory operations.
    Composes EventMemory, SemanticMemory, ProceduralMemory,
    StalenessDetector, UtilityScorer, and PAMI.

    Args:
        agent_id: Unique identifier for this agent/user
        data_dir: Base directory for persistent storage
        embedding_model: SentenceTransformer model name
        graph_backend: "kuzu" | "networkx" | "auto"
        vector_backend: "chromadb" | "dict" | "auto"
        token_budget: Default PAMI token budget
        decay_lambda: StalenessDetector R1 decay rate
        utility_weights: dict with keys semantic/recency/frequency/causal
    """

    def __init__(
        self,
        agent_id: str,
        data_dir: str = "./memai_data",
        embedding_model: str = "all-MiniLM-L6-v2",
        graph_backend: str = "auto",
        vector_backend: str = "auto",
        token_budget: int = 2000,
        decay_lambda: float = 0.05,
        utility_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.agent_id = agent_id
        self.data_dir = data_dir
        self.token_budget = token_budget

        # Initialize all subsystems
        self.event_memory = EventMemory(
            db_path=f"{data_dir}/events",
            backend=graph_backend,
        )
        self.semantic_memory = SemanticMemory(
            db_path=f"{data_dir}/semantic",
            embedding_model=embedding_model,
            backend=vector_backend,
        )
        self.procedural_memory = ProceduralMemory(
            db_path=f"{data_dir}/procedural.db"
        )
        self.staleness_detector = StalenessDetector(
            decay_lambda=decay_lambda,
        )

        weights = utility_weights or {}
        self.utility_scorer = UtilityScorer(
            w_semantic=weights.get("semantic", 0.40),
            w_recency=weights.get("recency", 0.25),
            w_frequency=weights.get("frequency", 0.20),
            w_causal=weights.get("causal", 0.15),
        )

        self.pami = PAMI(token_budget=token_budget)

        # Convenience proxy
        self.workflows = WorkflowProxy(self)

        logger.info("memai Memory initialized for agent_id=%s", agent_id)

    # ------------------------------------------------------------------
    # CORE OPERATIONS
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.AUTO,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AddMemoryResponse:
        """
        Add a memory. Automatically routes to EventMemory or SemanticMemory
        based on detected content type.
        """
        inferred_type = self._infer_type(text) if memory_type == MemoryType.AUTO else memory_type

        if inferred_type == MemoryType.EVENT:
            node = self.event_memory.add_event(
                text=text,
                agent_id=self.agent_id,
                session_id=session_id,
                metadata=metadata,
            )
            return AddMemoryResponse(memory_id=node.id, type_inferred=MemoryType.EVENT)
        else:
            item = self.semantic_memory.add(
                text=text,
                agent_id=self.agent_id,
                session_id=session_id,
                memory_type=inferred_type,
                metadata=metadata,
            )
            return AddMemoryResponse(memory_id=item.id, type_inferred=inferred_type)

    def search(
        self,
        query: str,
        k: int = 10,
        session_id: Optional[str] = None,
        context_budget: Optional[int] = None,
        include_events: bool = True,
        include_semantic: bool = True,
    ) -> SearchResult:
        """
        Search memories, re-rank by utility, apply PAMI, return context.

        Returns a SearchResult with:
          - .memories: the ranked list of MemoryItem
          - .pami_context: ready-to-inject prompt string
        """
        budget = context_budget or self.token_budget
        candidates: list[MemoryItem] = []

        # Gather semantic memories
        if include_semantic:
            semantic_results = self.semantic_memory.search(
                query=query,
                agent_id=self.agent_id,
                k=k,
                session_id=session_id,
            )
            candidates.extend(semantic_results)

        # Gather event memories (converted to MemoryItem for unified ranking)
        if include_events:
            events = self.event_memory.search_events(
                agent_id=self.agent_id,
                session_id=session_id,
                limit=k,
            )
            candidates.extend(self._events_to_memory_items(events))

        if not candidates:
            return SearchResult(
                memories=[],
                pami_context="",
                total_tokens_estimated=0,
                dropped_count=0,
            )

        # Staleness filtering
        staleness_scores: dict[str, float] = {}
        for mem in candidates:
            result = self.staleness_detector.check(mem)
            staleness_scores[mem.id] = result.adjusted_score

        # Get query embedding for utility scoring
        query_embedding = None
        embedder = self.semantic_memory._get_embedder()
        if embedder:
            try:
                embs = embedder.encode([query], normalize_embeddings=True)
                query_embedding = embs[0].tolist()
            except Exception:
                pass

        # Re-rank by utility
        ranked = self.utility_scorer.rerank(
            memories=candidates,
            query_embedding=query_embedding,
            query_text=query,
            staleness_adjusted_scores=staleness_scores,
        )

        # Update usage stats for top results
        for mem, score in ranked[:5]:
            self.utility_scorer.update_usage(mem)

        # Apply PAMI for position-aware injection (budget configured at search time)
        pami_engine = PAMI(token_budget=budget)
        pami_result = pami_engine.inject(ranked_memories=ranked)

        return SearchResult(
            memories=[m for m, _ in ranked[:k]],
            pami_context=pami_result.context,
            total_tokens_estimated=pami_result.total_tokens,
            dropped_count=len(pami_result.dropped),
        )

    def forget(
        self,
        older_than_days: Optional[int] = None,
        staleness_threshold: float = 0.1,
    ) -> ForgetResponse:
        """
        Delete stale or old memories for this agent.
        """
        all_memories = self.semantic_memory.list_agent(self.agent_id, limit=10000)
        stale_pairs = self.staleness_detector.sweep(all_memories)

        deleted_count = 0
        for mem, result in stale_pairs:
            if result.adjusted_score < staleness_threshold:
                if self.semantic_memory.delete(mem.id):
                    deleted_count += 1

        if older_than_days is not None:
            from datetime import timezone
            cutoff = datetime.now(timezone.utc)
            n = self.event_memory.temporal_compress(
                agent_id=self.agent_id,
                before_timestamp=cutoff,
                keep_ratio=0.5,
            )
            deleted_count += n

        return ForgetResponse(deleted_count=deleted_count)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID."""
        return self.semantic_memory.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        return self.semantic_memory.delete(memory_id)

    def count(self) -> int:
        """Count all semantic memories for this agent."""
        return self.semantic_memory.count(self.agent_id)

    # ------------------------------------------------------------------
    # SESSION CONTEXT MANAGER
    # ------------------------------------------------------------------

    def session(self, session_id: str) -> SessionContext:
        """
        Start a session context for grouped event tracking.

        Usage:
            with mem.session("sess-123") as s:
                s.add_event("User asked X")
                s.add_event("Agent responded Y")
                timeline = s.timeline()
        """
        return SessionContext(self, session_id)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _infer_type(self, text: str) -> MemoryType:
        """
        Auto-detect memory type from text content.
        Events: temporal words, actions, agent/user interaction markers.
        Procedural: step-like, workflow-like content.
        Semantic: default (facts, preferences, knowledge).
        """
        text_lower = text.lower()
        event_keywords = {
            "happened", "occurred", "asked", "responded", "said",
            "called", "created", "deleted", "updated", "clicked",
            "session", "turn", "step", "then", "after", "before",
        }
        procedural_keywords = {
            "step", "first", "second", "then", "finally", "workflow",
            "procedure", "process", "run", "execute", "do this",
        }
        words = set(text_lower.split())
        if len(words & event_keywords) >= 2:
            return MemoryType.EVENT
        if len(words & procedural_keywords) >= 2:
            return MemoryType.PROCEDURAL
        return MemoryType.SEMANTIC

    def _events_to_memory_items(self, events: list[EventNode]) -> list[MemoryItem]:
        """Convert EventNode objects to MemoryItem for unified ranking."""
        return [
            MemoryItem(
                id=e.id,
                text=e.text,
                agent_id=e.agent_id,
                session_id=e.session_id,
                memory_type=MemoryType.EVENT,
                created_at=e.timestamp,
                metadata=e.metadata,
            )
            for e in events
        ]

    def close(self) -> None:
        """Clean up resources."""
        self.event_memory.close()
        self.procedural_memory.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"Memory(agent_id={self.agent_id!r}, data_dir={self.data_dir!r})"
