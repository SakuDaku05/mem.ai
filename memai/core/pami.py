"""
PAMI — Position-Aware Memory Injection.

Solves the "lost in the middle" problem by placing high-utility memories
at the START and END of the context window, not in the middle.

Algorithm:
  1. Score all candidate memories via UtilityScorer
  2. Sort by utility score descending
  3. Assign positions:
       TOP 20% utility    -> inject at START of context (before task)
       BOTTOM 20% utility -> inject at END of context (after task)
       MIDDLE utility     -> compress and inject
       BELOW threshold    -> drop entirely
  4. Budget: respect token count limit
  5. Output: a formatted prompt block ready for LLM injection

References:
  - Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts"
  - Our research contribution: utility-score-driven position assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from memai.models import MemoryItem

logger = logging.getLogger(__name__)

# Approximate tokens-per-character ratio (conservative)
_CHARS_PER_TOKEN = 4


@dataclass
class PAMIResult:
    """Result of PAMI context construction."""
    context: str                  # Full formatted context string
    start_memories: list[MemoryItem]   # Placed at start
    end_memories: list[MemoryItem]     # Placed at end
    middle_memories: list[MemoryItem]  # Compressed/summarized
    dropped: list[MemoryItem]          # Dropped due to budget/threshold
    total_tokens: int
    token_budget: int


class PAMI:
    """
    Position-Aware Memory Injection engine.

    Usage:
        pami = PAMI(token_budget=2000)
        ranked = [(mem1, 0.9), (mem2, 0.6), (mem3, 0.2)]
        result = pami.inject(ranked)
        # Use result.context as the memory section in your LLM prompt
    """

    def __init__(
        self,
        token_budget: int = 2000,
        top_fraction: float = 0.20,
        bottom_fraction: float = 0.20,
        drop_threshold: float = 0.05,
        compress_middle: bool = True,
    ) -> None:
        """
        Args:
            token_budget: Max tokens to use for injected memory context
            top_fraction: Fraction of memories to place at START (high utility)
            bottom_fraction: Fraction to place at END
            drop_threshold: Utility score below which memory is dropped entirely
            compress_middle: Whether to summarize middle-tier memories
        """
        self.token_budget = token_budget
        self.top_fraction = top_fraction
        self.bottom_fraction = bottom_fraction
        self.drop_threshold = drop_threshold
        self.compress_middle = compress_middle

    # ------------------------------------------------------------------
    # PRIMARY API
    # ------------------------------------------------------------------

    def inject(
        self,
        ranked_memories: Sequence[tuple[MemoryItem, float]],
        task_text: Optional[str] = None,
        header: str = "## Relevant Memory Context",
    ) -> PAMIResult:
        """
        Build a positioned memory context string.

        Args:
            ranked_memories: List of (MemoryItem, utility_score) sorted desc
            task_text: Current task/query (placed in the middle, between memories)
            header: Section header for the context block

        Returns:
            PAMIResult with the formatted context and categorization info
        """
        if not ranked_memories:
            return PAMIResult(
                context="",
                start_memories=[],
                end_memories=[],
                middle_memories=[],
                dropped=[],
                total_tokens=0,
                token_budget=self.token_budget,
            )

        # Filter below drop threshold
        viable = [(m, s) for m, s in ranked_memories if s >= self.drop_threshold]
        dropped = [m for m, s in ranked_memories if s < self.drop_threshold]

        if not viable:
            return PAMIResult(
                context="",
                start_memories=[],
                end_memories=[],
                middle_memories=[],
                dropped=dropped,
                total_tokens=0,
                token_budget=self.token_budget,
            )

        n = len(viable)
        n_top = max(1, int(n * self.top_fraction))
        n_bottom = max(1, int(n * self.bottom_fraction))

        # Prevent overlap
        if n_top + n_bottom >= n:
            n_top = max(1, n // 2)
            n_bottom = n - n_top

        top_group = viable[:n_top]
        bottom_group = viable[n - n_bottom:]
        middle_group = viable[n_top: n - n_bottom]

        top_items = [m for m, _ in top_group]
        bottom_items = [m for m, _ in bottom_group]
        middle_items = [m for m, _ in middle_group]

        # Build context respecting token budget
        context, total_tokens = self._build_context(
            top_items=top_items,
            middle_items=middle_items,
            bottom_items=bottom_items,
            task_text=task_text,
            header=header,
        )

        return PAMIResult(
            context=context,
            start_memories=top_items,
            end_memories=bottom_items,
            middle_memories=middle_items,
            dropped=dropped,
            total_tokens=total_tokens,
            token_budget=self.token_budget,
        )

    def inject_into_prompt(
        self,
        system_prompt: str,
        user_message: str,
        ranked_memories: Sequence[tuple[MemoryItem, float]],
    ) -> tuple[str, str]:
        """
        Convenience: inject memory context into an existing prompt pair.

        Returns (enriched_system_prompt, user_message) with memories placed
        according to PAMI positioning.
        """
        result = self.inject(ranked_memories, task_text=user_message)
        if not result.context:
            return system_prompt, user_message

        enriched_system = f"{system_prompt}\n\n{result.context}"
        return enriched_system, user_message

    # ------------------------------------------------------------------
    # CONTEXT BUILDER
    # ------------------------------------------------------------------

    def _build_context(
        self,
        top_items: list[MemoryItem],
        middle_items: list[MemoryItem],
        bottom_items: list[MemoryItem],
        task_text: Optional[str],
        header: str,
    ) -> tuple[str, int]:
        """
        Build the final context string with budget enforcement.

        Structure:
            [header]
            --- High-relevance memories (START) ---
            [top memories]
            --- Task ---
            [task_text if provided]
            --- Supporting context ---
            [middle memories, compressed]
            --- Additional context (END) ---
            [bottom memories]
        """
        budget_chars = self.token_budget * _CHARS_PER_TOKEN
        parts = []
        used_chars = 0

        def _add(text: str) -> bool:
            nonlocal used_chars
            if used_chars + len(text) > budget_chars:
                return False
            parts.append(text)
            used_chars += len(text)
            return True

        _add(f"{header}\n")

        # START section — high-utility memories
        if top_items:
            _add("\n### Key Context (Critical)\n")
            for i, mem in enumerate(top_items):
                text = self._format_memory(mem, index=i + 1)
                if not _add(text):
                    break

        # MIDDLE section — task placeholder + supporting memories
        if task_text:
            _add(f"\n### Current Task\n{task_text}\n")

        if middle_items:
            _add("\n### Supporting Context\n")
            for i, mem in enumerate(middle_items):
                if self.compress_middle:
                    text = self._format_memory_compressed(mem, index=i + 1)
                else:
                    text = self._format_memory(mem, index=i + 1)
                if not _add(text):
                    break

        # END section — bottom-tier memories
        if bottom_items:
            _add("\n### Additional Background\n")
            for i, mem in enumerate(bottom_items):
                text = self._format_memory(mem, index=i + 1)
                if not _add(text):
                    break

        context = "".join(parts)
        total_tokens = used_chars // _CHARS_PER_TOKEN
        return context, total_tokens

    # ------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------

    def _format_memory(self, mem: MemoryItem, index: int) -> str:
        """Full memory format."""
        ts = mem.created_at.strftime("%Y-%m-%d %H:%M UTC") if mem.created_at else ""
        session_info = f" [session: {mem.session_id}]" if mem.session_id else ""
        return (
            f"[{index}] ({ts}{session_info})\n"
            f"{mem.text}\n\n"
        )

    def _format_memory_compressed(self, mem: MemoryItem, index: int) -> str:
        """Compressed memory format — truncate to first 100 chars."""
        preview = mem.text[:100].rstrip()
        if len(mem.text) > 100:
            preview += "…"
        ts = mem.created_at.strftime("%Y-%m-%d") if mem.created_at else ""
        return f"[{index}] ({ts}) {preview}\n"

    def budget_summary(self, result: PAMIResult) -> str:
        """Human-readable summary of PAMI decisions."""
        return (
            f"PAMI Summary: "
            f"{len(result.start_memories)} critical | "
            f"{len(result.middle_memories)} supporting | "
            f"{len(result.end_memories)} background | "
            f"{len(result.dropped)} dropped | "
            f"{result.total_tokens}/{result.token_budget} tokens used"
        )
