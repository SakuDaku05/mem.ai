"""
UtilityScorer — Composite Q-style utility scoring for memory re-ranking.

Formula:
    U(m, q) = w1 * semantic_sim(m, q)
            + w2 * recency_score(m)
            + w3 * usage_frequency(m)
            + w4 * causal_relevance(m, context)

This is NOT reinforcement learning — it is a deterministic composite score
that mimics Q-value behavior (reward-weighted re-ranking) without requiring
an explicit RL training loop. In v2, weights can be learned via bandit updates.

Answers BEAM abilities:
  - Information Extraction (surfaces most relevant facts)
  - Multi-Session Reasoning (causal relevance term)
  - Preference Following (usage frequency tracks what worked before)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional, Sequence

from memai.models import MemoryItem

logger = logging.getLogger(__name__)


class UtilityScorer:
    """
    Composite utility scorer for memory re-ranking.

    Weights (default):
        w_semantic  = 0.40  — cosine similarity to query
        w_recency   = 0.25  — exponential recency (newer = higher)
        w_frequency = 0.20  — normalized access count
        w_causal    = 0.15  — graph hop distance to current context

    Usage:
        scorer = UtilityScorer()
        ranked = scorer.rerank(memories, query="what does user prefer?")
        top_memory = ranked[0]
    """

    def __init__(
        self,
        w_semantic: float = 0.40,
        w_recency: float = 0.25,
        w_frequency: float = 0.20,
        w_causal: float = 0.15,
        recency_halflife_days: float = 7.0,
        frequency_cap: int = 100,
    ) -> None:
        total = w_semantic + w_recency + w_frequency + w_causal
        if abs(total - 1.0) > 0.01:
            # Normalize to sum to 1
            w_semantic /= total
            w_recency /= total
            w_frequency /= total
            w_causal /= total

        self.w_semantic = w_semantic
        self.w_recency = w_recency
        self.w_frequency = w_frequency
        self.w_causal = w_causal
        self.recency_halflife_days = recency_halflife_days
        self.frequency_cap = frequency_cap

    # ------------------------------------------------------------------
    # PRIMARY API
    # ------------------------------------------------------------------

    def score(
        self,
        memory: MemoryItem,
        query_embedding: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        causal_hop_distance: Optional[int] = None,
    ) -> float:
        """
        Compute composite utility score for a single memory.

        Args:
            memory: The memory item to score
            query_embedding: Query vector for semantic similarity
            query_text: Fallback text for keyword similarity
            causal_hop_distance: Graph hop distance from current event (None = unknown)

        Returns:
            Utility score in [0, 1]
        """
        s_sem = self._semantic_score(memory, query_embedding, query_text)
        s_rec = self._recency_score(memory)
        s_freq = self._frequency_score(memory)
        s_caus = self._causal_score(causal_hop_distance)

        utility = (
            self.w_semantic * s_sem
            + self.w_recency * s_rec
            + self.w_frequency * s_freq
            + self.w_causal * s_caus
        )
        return max(0.0, min(1.0, utility))

    def rerank(
        self,
        memories: Sequence[MemoryItem],
        query_embedding: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        causal_distances: Optional[dict[str, int]] = None,
        staleness_adjusted_scores: Optional[dict[str, float]] = None,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Re-rank a list of memories by utility score.

        Args:
            memories: Candidate memory items
            query_embedding: Query vector
            query_text: Query text (fallback)
            causal_distances: {memory_id -> hop_distance} from graph traversal
            staleness_adjusted_scores: Override base score for stale memories

        Returns:
            Sorted list of (MemoryItem, score) tuples, highest score first.
        """
        scored = []
        for mem in memories:
            # Apply staleness adjustment if provided
            if staleness_adjusted_scores and mem.id in staleness_adjusted_scores:
                stale_adj = staleness_adjusted_scores[mem.id]
                if stale_adj < 0.01:
                    # Effectively stale — skip entirely
                    continue

            hop = causal_distances.get(mem.id) if causal_distances else None
            s = self.score(
                mem,
                query_embedding=query_embedding,
                query_text=query_text,
                causal_hop_distance=hop,
            )

            # Apply staleness dampening
            if staleness_adjusted_scores and mem.id in staleness_adjusted_scores:
                s *= staleness_adjusted_scores[mem.id]

            scored.append((mem, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def update_usage(self, memory: MemoryItem) -> MemoryItem:
        """
        Update access statistics after a memory is used.
        Call this when a retrieved memory is actually injected into a prompt.
        """
        memory.access_count += 1
        memory.last_accessed_at = datetime.now(timezone.utc)
        # Incrementally update utility_score toward usage signal
        # Simple EMA: score = 0.9 * old_score + 0.1 * 1.0
        memory.utility_score = min(1.0, 0.9 * memory.utility_score + 0.1)
        return memory

    def penalize_unused(self, memory: MemoryItem, penalty: float = 0.02) -> MemoryItem:
        """
        Slightly reduce utility for memories that were retrieved but not used.
        This mimics negative reward in RL.
        """
        memory.utility_score = max(0.0, memory.utility_score - penalty)
        return memory

    # ------------------------------------------------------------------
    # COMPONENT SCORES
    # ------------------------------------------------------------------

    def _semantic_score(
        self,
        memory: MemoryItem,
        query_embedding: Optional[list[float]],
        query_text: Optional[str],
    ) -> float:
        """Cosine similarity between memory and query embeddings."""
        if query_embedding and memory.embedding:
            return self._cosine(query_embedding, memory.embedding)
        if query_text:
            return self._keyword_overlap(query_text, memory.text)
        return 0.5  # neutral if no query info

    def _recency_score(self, memory: MemoryItem) -> float:
        """
        Exponential recency: newer memories score higher.
        Score = exp(-ln(2) * age_days / halflife_days)
        At age=0: score=1.0, at age=halflife: score=0.5
        """
        now = datetime.now(timezone.utc)
        created = memory.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - created).total_seconds() / 86400)
        halflife = self.recency_halflife_days
        return math.exp(-math.log(2) * age_days / halflife)

    def _frequency_score(self, memory: MemoryItem) -> float:
        """
        Normalized access count.
        Uses log scale to prevent super-frequently accessed memories
        from dominating.
        """
        count = min(memory.access_count, self.frequency_cap)
        if count == 0:
            return 0.0
        # log(1+count) / log(1+cap) → [0, 1]
        return math.log1p(count) / math.log1p(self.frequency_cap)

    def _causal_score(self, hop_distance: Optional[int]) -> float:
        """
        Inverse hop distance score.
        hop=0 (same event): 1.0
        hop=1: 0.5
        hop=2: 0.33
        hop=None (unknown): 0.5 (neutral)
        """
        if hop_distance is None:
            return 0.5
        if hop_distance == 0:
            return 1.0
        return 1.0 / (1.0 + hop_distance)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def _keyword_overlap(query: str, text: str) -> float:
        """Simple keyword overlap as fallback similarity."""
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        if not q_words:
            return 0.0
        return len(q_words & t_words) / len(q_words)

    def describe(self, memory: MemoryItem, query_text: str = "") -> str:
        """Debug: show all component scores for a memory."""
        s_sem = self._semantic_score(memory, None, query_text)
        s_rec = self._recency_score(memory)
        s_freq = self._frequency_score(memory)
        s_caus = self._causal_score(None)
        total = (
            self.w_semantic * s_sem
            + self.w_recency * s_rec
            + self.w_frequency * s_freq
            + self.w_causal * s_caus
        )
        return (
            f"Semantic: {s_sem:.3f} (w={self.w_semantic}) | "
            f"Recency: {s_rec:.3f} (w={self.w_recency}) | "
            f"Frequency: {s_freq:.3f} (w={self.w_frequency}) | "
            f"Causal: {s_caus:.3f} (w={self.w_causal}) | "
            f"Total: {total:.3f}"
        )
