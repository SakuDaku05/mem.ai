"""
StalenessDetector — Rule-based staleness detection for agent memories.

Implements 4 rules:
  R1: TIME DECAY    — exponential score decay based on memory age
  R2: CONTRADICTION — semantic/keyword detection of conflicting facts
  R3: SUPERSEDE     — pattern-based detection of explicit updates
  R4: DORMANT       — frequency-floor for rarely accessed memories

Answers BEAM abilities:
  - Knowledge Update (R3: detects "X was updated to Y")
  - Contradiction Resolution (R2: detects conflicting facts)
  - Abstention (refuse to surface stale/poisoned facts)
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Optional, Sequence

from memai.models import MemoryItem, StalenessReason, StalenessResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supersede patterns — R3
# ---------------------------------------------------------------------------
_SUPERSEDE_PATTERNS = [
    r"\bis now\b",
    r"\bhas been updated to\b",
    r"\bchanged (from .+? )?to\b",
    r"\bno longer\b",
    r"\bwas replaced by\b",
    r"\binstead of\b",
    r"\bpreviously .+? now\b",
    r"\bused to be\b",
    r"\bswitched (from .+? )?to\b",
    r"\brevised to\b",
    r"\bcorrected to\b",
]
_SUPERSEDE_RE = re.compile("|".join(_SUPERSEDE_PATTERNS), re.IGNORECASE)

# Negation keywords used for contradiction detection (R2)
_NEGATION_WORDS = {
    "not", "never", "no", "none", "neither", "nor",
    "cannot", "can't", "won't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't",
}


class StalenessDetector:
    """
    Rule-based staleness checker for MemoryItems.

    Configuration:
        decay_lambda     — R1 time decay rate (default 0.05/day)
        dormant_days     — R4 days without access before DORMANT
        dormant_utility  — R4 utility floor for dormant classification
        staleness_floor  — adjusted score below which memory is STALE
        contradiction_sim_threshold — keyword overlap to trigger R2 check
    """

    def __init__(
        self,
        decay_lambda: float = 0.05,
        dormant_days: int = 30,
        dormant_utility_threshold: float = 0.15,
        staleness_floor: float = 0.05,
        contradiction_overlap_threshold: float = 0.6,
    ) -> None:
        self.decay_lambda = decay_lambda
        self.dormant_days = dormant_days
        self.dormant_utility_threshold = dormant_utility_threshold
        self.staleness_floor = staleness_floor
        self.contradiction_overlap_threshold = contradiction_overlap_threshold

    # ------------------------------------------------------------------
    # PRIMARY API
    # ------------------------------------------------------------------

    def check(self, memory: MemoryItem) -> StalenessResult:
        """
        Run all staleness rules on a single memory.
        Returns the most severe staleness result found.
        """
        now = datetime.now(timezone.utc)
        age_days = self._age_days(memory, now)

        # R1: Time decay
        r1 = self._rule_time_decay(age_days, memory.utility_score)

        # R3: Supersede pattern (text-based, no external reference needed)
        r3 = self._rule_supersede(memory.text)

        # R4: Dormant check
        r4 = self._rule_dormant(memory, now)

        # Return most severe
        results = [r1, r3, r4]
        stale_results = [r for r in results if r.is_stale]
        if stale_results:
            # Most confident stale result wins
            return max(stale_results, key=lambda r: r.confidence)

        # Not stale — return R1 result (has adjusted_score)
        return r1

    def check_contradiction(
        self,
        new_memory: MemoryItem,
        existing_memories: Sequence[MemoryItem],
    ) -> list[tuple[MemoryItem, StalenessResult]]:
        """
        R2: Check if new_memory contradicts any existing memories.
        Returns list of (existing_memory, StalenessResult) for contradicted items.
        """
        contradicted = []
        new_words = set(new_memory.text.lower().split())
        new_has_negation = bool(new_words & _NEGATION_WORDS)

        for existing in existing_memories:
            if existing.id == new_memory.id:
                continue
            result = self._rule_contradiction(new_memory, existing, new_has_negation)
            if result.is_stale:
                contradicted.append((existing, result))

        return contradicted

    def sweep(
        self,
        memories: Sequence[MemoryItem],
    ) -> list[tuple[MemoryItem, StalenessResult]]:
        """
        Batch sweep: run check() on all memories.
        Returns only the stale ones with their results.
        """
        stale = []
        for mem in memories:
            result = self.check(mem)
            if result.is_stale:
                stale.append((mem, result))
        return stale

    def adjusted_score(self, memory: MemoryItem) -> float:
        """Return the time-decay-adjusted utility score for a memory."""
        now = datetime.now(timezone.utc)
        age_days = self._age_days(memory, now)
        return self._decay(memory.utility_score, age_days)

    # ------------------------------------------------------------------
    # RULE IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def _rule_time_decay(self, age_days: float, base_score: float) -> StalenessResult:
        """R1: Exponential time decay."""
        adjusted = self._decay(base_score, age_days)
        is_stale = adjusted < self.staleness_floor
        return StalenessResult(
            is_stale=is_stale,
            reason=StalenessReason.TIME_DECAY if is_stale else StalenessReason.NONE,
            confidence=1.0 - adjusted if is_stale else 1.0,
            adjusted_score=adjusted,
        )

    def _rule_contradiction(
        self,
        new_memory: MemoryItem,
        existing: MemoryItem,
        new_has_negation: bool,
    ) -> StalenessResult:
        """R2: Detect if new_memory contradicts existing."""
        new_words = set(new_memory.text.lower().split())
        old_words = set(existing.text.lower().split())
        old_has_negation = bool(old_words & _NEGATION_WORDS)

        # High overlap + negation asymmetry = likely contradiction
        content_words_new = new_words - _NEGATION_WORDS
        content_words_old = old_words - _NEGATION_WORDS
        overlap = len(content_words_new & content_words_old) / max(len(content_words_new), 1)

        is_contradiction = (
            overlap >= self.contradiction_overlap_threshold
            and new_has_negation != old_has_negation
        )

        if is_contradiction:
            confidence = min(1.0, overlap)
            return StalenessResult(
                is_stale=True,
                reason=StalenessReason.CONTRADICTION,
                confidence=confidence,
                adjusted_score=max(0.0, existing.utility_score * (1.0 - confidence)),
            )
        return StalenessResult()

    def _rule_supersede(self, text: str) -> StalenessResult:
        """R3: Detect explicit supersede language in the memory text."""
        if _SUPERSEDE_RE.search(text):
            return StalenessResult(
                is_stale=True,
                reason=StalenessReason.SUPERSEDED,
                confidence=0.9,
                adjusted_score=0.0,
            )
        return StalenessResult()

    def _rule_dormant(self, memory: MemoryItem, now: datetime) -> StalenessResult:
        """R4: Mark as dormant if not accessed recently and low utility."""
        age_days = self._age_days(memory, now)
        if memory.access_count == 0 and age_days > self.dormant_days:
            return StalenessResult(
                is_stale=True,
                reason=StalenessReason.DORMANT,
                confidence=min(1.0, age_days / (self.dormant_days * 2)),
                adjusted_score=memory.utility_score * 0.1,
            )
        last_access = memory.last_accessed_at or memory.created_at
        days_since_access = (now - last_access).total_seconds() / 86400
        if (
            days_since_access > self.dormant_days
            and memory.utility_score < self.dormant_utility_threshold
        ):
            return StalenessResult(
                is_stale=True,
                reason=StalenessReason.DORMANT,
                confidence=0.7,
                adjusted_score=memory.utility_score * 0.1,
            )
        return StalenessResult()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _decay(self, base_score: float, age_days: float) -> float:
        """Exponential decay: score * e^(-lambda * age_days)."""
        return base_score * math.exp(-self.decay_lambda * age_days)

    def _age_days(self, memory: MemoryItem, now: datetime) -> float:
        created = memory.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (now - created).total_seconds() / 86400)

    def explain(self, memory: MemoryItem) -> str:
        """Human-readable staleness explanation for a memory."""
        result = self.check(memory)
        age_days = self._age_days(memory, datetime.now(timezone.utc))
        adjusted = self._decay(memory.utility_score, age_days)
        lines = [
            f"Memory ID: {memory.id}",
            f"Age: {age_days:.1f} days",
            f"Base utility: {memory.utility_score:.3f}",
            f"Adjusted score (R1 decay): {adjusted:.3f}",
            f"Stale: {result.is_stale}",
            f"Reason: {result.reason.value}",
            f"Confidence: {result.confidence:.2f}",
        ]
        return "\n".join(lines)
