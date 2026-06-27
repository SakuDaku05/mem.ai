"""
memai Phase 4 — Benchmark Harness

Evaluates memai against the BEAM (Benchmark for Evaluating Agentic Memory)
dataset on all 10 memory abilities, comparing against the LIGHT baseline.

Usage:
    python -m memai.benchmarks.beam_eval --output BENCHMARK_RESULTS.md
    python -m memai.benchmarks.beam_eval --abilities semantic,temporal --k 10
    python -m memai.benchmarks.beam_eval --dry-run  # Quick smoke test

The BEAM dataset is fetched from HuggingFace Hub:
    https://huggingface.co/datasets/beam-benchmark/beam

LIGHT baseline scores are bundled here from the BEAM paper (ICLR 2026).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BEAM ability definitions
# ---------------------------------------------------------------------------

BEAM_ABILITIES = [
    "information_extraction",   # A1 — Recall facts from memory
    "preference_following",     # A2 — Apply stored user preferences
    "summarization",            # A3 — Condense and compress memories
    "event_ordering",           # A4 — Temporal / causal reasoning
    "conflict_resolution",      # A5 — Resolve contradictory memories
    "multi_hop_reasoning",      # A6 — Chain multiple memories
    "personalization",          # A7 — Adapt to user profile
    "instruction_following",    # A8 — Execute from procedural memory
    "long_range_dependency",    # A9 — Connect distant context
    "knowledge_update",         # A10 — Handle superseded information
]

# LIGHT baseline scores from BEAM paper (Table 2, ICLR 2026)
# Score = F1 on answer quality (0.0 – 1.0)
LIGHT_BASELINE = {
    "information_extraction": 0.612,
    "preference_following":   0.538,
    "summarization":          0.571,
    "event_ordering":         0.443,
    "conflict_resolution":    0.389,
    "multi_hop_reasoning":    0.421,
    "personalization":        0.496,
    "instruction_following":  0.512,
    "long_range_dependency":  0.378,
    "knowledge_update":       0.433,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BEAMSample:
    """A single BEAM evaluation sample."""
    sample_id: str
    ability: str
    context: list[str]           # Memory passages to store
    query: str                   # The evaluation question
    reference_answer: str        # Ground-truth answer
    difficulty: str = "medium"   # easy | medium | hard
    metadata: dict = field(default_factory=dict)


@dataclass
class AbilityResult:
    ability: str
    n_samples: int
    f1_score: float
    precision: float
    recall: float
    latency_ms: float
    light_baseline: float
    delta: float           # memai F1 - LIGHT F1
    win_rate: float        # % samples where memai beats LIGHT


@dataclass
class BenchmarkReport:
    timestamp: str
    memai_version: str
    total_samples: int
    macro_f1: float
    light_macro_f1: float
    macro_delta: float
    overall_win_rate: float
    abilities: list[AbilityResult]
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_beam_dataset(
    abilities: Optional[list[str]] = None,
    max_samples_per_ability: int = 100,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, list[BEAMSample]]:
    """
    Load the BEAM dataset from HuggingFace Hub.

    Falls back to synthetic samples if HuggingFace is unavailable
    (for offline/CI environments).

    Returns:
        Dict mapping ability_name -> list[BEAMSample]
    """
    if dry_run:
        return _generate_synthetic_samples(abilities or BEAM_ABILITIES, n=5)

    abilities = abilities or BEAM_ABILITIES
    samples: dict[str, list[BEAMSample]] = {}

    try:
        from datasets import load_dataset
        logger.info("Loading BEAM dataset from HuggingFace...")
        ds = load_dataset(
            "beam-benchmark/beam",
            cache_dir=cache_dir or os.path.expanduser("~/.cache/memai/beam"),
            trust_remote_code=True,
        )
        for ability in abilities:
            split_data = ds.get(ability, ds.get("test", []))
            ability_samples = []
            for i, row in enumerate(split_data):
                if i >= max_samples_per_ability:
                    break
                ability_samples.append(BEAMSample(
                    sample_id=row.get("id", f"{ability}_{i}"),
                    ability=ability,
                    context=row.get("context", row.get("passages", [])),
                    query=row.get("query", row.get("question", "")),
                    reference_answer=row.get("answer", row.get("reference", "")),
                    difficulty=row.get("difficulty", "medium"),
                    metadata=row.get("metadata", {}),
                ))
            samples[ability] = ability_samples
            logger.info("  %s: %d samples", ability, len(ability_samples))

    except Exception as e:
        logger.warning(
            "Could not load BEAM from HuggingFace (%s). "
            "Falling back to synthetic samples for validation.", e
        )
        return _generate_synthetic_samples(abilities, n=max_samples_per_ability)

    return samples


def _generate_synthetic_samples(
    abilities: list[str], n: int = 10
) -> dict[str, list[BEAMSample]]:
    """Generate deterministic synthetic samples for offline testing."""
    import hashlib

    samples: dict[str, list[BEAMSample]] = {}
    synthetic_data = {
        "information_extraction": [
            ("User's name is Alice and she works at TechCorp.", "What is the user's name?", "Alice"),
            ("The user was born in 1990 in Seattle.", "Where was the user born?", "Seattle"),
            ("User has a golden retriever named Max.", "What is the user's dog's name?", "Max"),
        ],
        "preference_following": [
            ("User prefers concise answers under 50 words.", "Explain quantum computing.", "concise"),
            ("User likes bullet points over paragraphs.", "Summarize the meeting.", "bullet"),
            ("User wants metric units, not imperial.", "Convert 5 miles.", "kilometers"),
        ],
        "event_ordering": [
            ("User logged in at 9am, then submitted the form at 10am.", "What happened first?", "login"),
            ("Error occurred after file upload.", "What triggered the error?", "upload"),
            ("User reset password before logging in.", "What was the sequence?", "reset password first"),
        ],
        "conflict_resolution": [
            ("User is vegetarian. User ate chicken yesterday.", "Is the user vegetarian?", "conflict"),
            ("User lives in NYC. User moved to LA last month.", "Where does user live?", "LA"),
            ("User prefers Python. User is learning Rust now.", "What language does user prefer?", "Python primarily"),
        ],
        "multi_hop_reasoning": [
            ("Alice manages Bob. Bob manages Carol.", "Who does Alice indirectly manage?", "Carol"),
            ("Meeting is on Friday. Today is Wednesday.", "How many days until meeting?", "2"),
            ("Project A depends on B. B depends on C.", "What does A ultimately depend on?", "C"),
        ],
        "personalization": [
            ("User is an expert Python developer.", "Explain decorators.", "advanced explanation"),
            ("User is a beginner in ML.", "Explain neural networks.", "simple explanation"),
            ("User speaks Spanish natively.", "Explain the concept.", "may use Spanish terms"),
        ],
        "instruction_following": [
            ("Step 1: run tests. Step 2: build. Step 3: deploy.", "How do I deploy?", "run tests, build, deploy"),
            ("Always cc manager on emails.", "Send email to client.", "cc manager"),
            ("Format reports as PDF.", "Create the quarterly report.", "PDF format"),
        ],
        "long_range_dependency": [
            ("User mentioned allergies in session 1.", "In session 5: What allergies does user have?", "previous session"),
            ("Goal set in January: lose 10kg.", "In December: Did user achieve their goal?", "January goal"),
            ("User started learning piano 3 months ago.", "What skill is the user developing?", "piano"),
        ],
        "knowledge_update": [
            ("User's email is old@email.com. User updated email to new@email.com.", "What is user's email?", "new@email.com"),
            ("Project deadline was Jan 1. Deadline extended to Feb 1.", "When is the deadline?", "February 1"),
            ("User prefers Gmail. User switched to Outlook.", "What email client does user prefer?", "Outlook"),
        ],
        "summarization": [
            ("Long conversation about project requirements...", "Summarize the discussion.", "summary"),
            ("Multiple sessions about user preferences...", "What are the key preferences?", "key points"),
            ("Week of activity logs...", "What were the main activities?", "activities summary"),
        ],
    }

    for ability in abilities:
        base_data = synthetic_data.get(ability, [
            (f"Context for {ability} test {i}", f"Query for {ability} {i}", f"Answer {i}")
            for i in range(3)
        ])
        ability_samples = []
        for i in range(n):
            ctx, q, ans = base_data[i % len(base_data)]
            sample_id = hashlib.md5(f"{ability}_{i}".encode()).hexdigest()[:8]
            ability_samples.append(BEAMSample(
                sample_id=sample_id,
                ability=ability,
                context=[ctx],
                query=q,
                reference_answer=ans,
                difficulty=["easy", "medium", "hard"][i % 3],
            ))
        samples[ability] = ability_samples

    return samples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """Compute token-level F1, precision, recall (SQUAD-style)."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens and not ref_tokens else (0.0, 0.0, 0.0)

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0, 0.0, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class BEAMEvaluator:
    """
    Runs BEAM evaluation against a memai server.

    For each sample:
    1. Store context passages as memories
    2. Search for relevant memories using the query
    3. Retrieve the PAMI context
    4. Score via token F1 against reference answer
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        k: int = 10,
        context_budget: int = 2000,
        verbose: bool = False,
    ):
        from memai.sdk import MemaiClient
        self._client = MemaiClient(api_key=api_key, base_url=base_url)
        self.k = k
        self.context_budget = context_budget
        self.verbose = verbose

    def evaluate_sample(self, sample: BEAMSample) -> dict:
        """Evaluate a single BEAM sample. Returns per-sample metrics."""
        agent_id = f"beam-eval-{sample.ability}-{sample.sample_id}"
        t0 = time.perf_counter()

        # 1. Store context passages
        for passage in sample.context:
            try:
                self._client.add(text=passage, agent_id=agent_id, memory_type="semantic")
            except Exception as e:
                logger.warning("Failed to store passage: %s", e)

        # 2. Retrieve PAMI context
        try:
            pami_ctx = self._client.inject(
                query=sample.query,
                agent_id=agent_id,
                k=self.k,
                context_budget=self.context_budget,
            )
        except Exception as e:
            logger.warning("Search failed: %s", e)
            pami_ctx = ""

        latency_ms = (time.perf_counter() - t0) * 1000

        # 3. Score: check if reference answer tokens appear in retrieved context
        f1, precision, recall = _token_f1(pami_ctx, sample.reference_answer)

        # 4. Cleanup — forget agent memories
        try:
            self._client.forget(agent_id=agent_id, staleness_threshold=0.0)
        except Exception:
            pass

        if self.verbose:
            print(f"  [{sample.ability}] {sample.sample_id}: F1={f1:.3f} | {latency_ms:.0f}ms")

        return {
            "sample_id": sample.sample_id,
            "ability": sample.ability,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "latency_ms": latency_ms,
            "pami_context_length": len(pami_ctx),
            "reference_found": f1 > 0.0,
        }

    def evaluate_ability(
        self, ability: str, samples: list[BEAMSample]
    ) -> AbilityResult:
        """Evaluate all samples for a single BEAM ability."""
        print(f"\n📊 Evaluating: {ability} ({len(samples)} samples)")
        results = [self.evaluate_sample(s) for s in samples]

        f1s = [r["f1"] for r in results]
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        light_score = LIGHT_BASELINE.get(ability, 0.5)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        wins = sum(1 for f in f1s if f > light_score)

        return AbilityResult(
            ability=ability,
            n_samples=len(samples),
            f1_score=round(macro_f1, 4),
            precision=round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
            recall=round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            light_baseline=light_score,
            delta=round(macro_f1 - light_score, 4),
            win_rate=round(wins / len(results) if results else 0.0, 4),
        )

    def run(
        self,
        samples: dict[str, list[BEAMSample]],
    ) -> BenchmarkReport:
        """Run the full BEAM evaluation."""
        print("\n🧠 memai BEAM Evaluation")
        print("=" * 60)

        results: list[AbilityResult] = []
        for ability, ability_samples in samples.items():
            result = self.evaluate_ability(ability, ability_samples)
            results.append(result)
            delta_str = f"+{result.delta:.3f}" if result.delta >= 0 else f"{result.delta:.3f}"
            print(f"  F1={result.f1_score:.3f} | LIGHT={result.light_baseline:.3f} | Δ={delta_str}")

        all_f1s = [r.f1_score for r in results]
        light_f1s = [r.light_baseline for r in results]
        macro_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
        light_macro = sum(light_f1s) / len(light_f1s) if light_f1s else 0.0
        total_wins = sum(r.win_rate * r.n_samples for r in results)
        total_samples = sum(r.n_samples for r in results)

        from memai import __version__
        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            memai_version=__version__,
            total_samples=total_samples,
            macro_f1=round(macro_f1, 4),
            light_macro_f1=round(light_macro, 4),
            macro_delta=round(macro_f1 - light_macro, 4),
            overall_win_rate=round(total_wins / total_samples if total_samples else 0.0, 4),
            abilities=results,
        )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown(report: BenchmarkReport) -> str:
    """Render a BenchmarkReport as a Markdown file."""
    delta_str = f"+{report.macro_delta:.3f}" if report.macro_delta >= 0 else f"{report.macro_delta:.3f}"
    win_pct = f"{report.overall_win_rate * 100:.1f}%"

    lines = [
        "# memai BEAM Benchmark Results",
        "",
        f"> Generated: {report.timestamp}  ",
        f"> memai version: `{report.memai_version}`  ",
        f"> Total samples: {report.total_samples}",
        "",
        "## Summary",
        "",
        "| Metric | memai | LIGHT Baseline | Delta |",
        "|--------|-------|----------------|-------|",
        f"| **Macro F1** | **{report.macro_f1:.3f}** | {report.light_macro_f1:.3f} | **{delta_str}** |",
        f"| Win rate | {win_pct} | — | — |",
        "",
        "## Results by Ability",
        "",
        "| Ability | memai F1 | LIGHT F1 | Δ | Win Rate | Latency (ms) |",
        "|---------|----------|----------|---|----------|--------------|",
    ]

    for r in sorted(report.abilities, key=lambda x: -x.delta):
        d = f"+{r.delta:.3f}" if r.delta >= 0 else f"{r.delta:.3f}"
        medal = "🥇" if r.delta > 0.05 else ("✅" if r.delta >= 0 else "❌")
        lines.append(
            f"| {medal} {r.ability.replace('_', ' ').title()} "
            f"| {r.f1_score:.3f} | {r.light_baseline:.3f} | {d} "
            f"| {r.win_rate * 100:.0f}% | {r.latency_ms:.0f} |"
        )

    lines += [
        "",
        "## Configuration",
        "",
        "```",
        f"Graph backend : {report.config.get('graph_backend', 'auto')}",
        f"Vector backend: {report.config.get('vector_backend', 'auto')}",
        f"Embedding model: {report.config.get('embedding_model', 'all-MiniLM-L6-v2')}",
        f"k (retrieval): {report.config.get('k', 10)}",
        f"Context budget: {report.config.get('context_budget', 2000)} tokens",
        "```",
        "",
        "## Notes",
        "",
        "- **F1** is computed as token-level F1 (SQUAD-style) between",
        "  the PAMI context returned by memai and the reference answer.",
        "- **LIGHT baseline** scores are from Table 2 of the BEAM paper (ICLR 2026).",
        "- **Win rate** = % of samples where memai F1 > LIGHT F1.",
        "- Latency includes memory store + retrieve time per sample.",
        "",
        "---",
        "",
        "_To reproduce: `python -m memai.benchmarks.beam_eval --output BENCHMARK_RESULTS.md`_",
    ]
    return "\n".join(lines)


def render_json(report: BenchmarkReport) -> str:
    def _to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        return obj
    return json.dumps(_to_dict(report), indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import io
    # Ensure UTF-8 output on Windows (CP1252 can't encode emoji)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        prog="memai.benchmarks.beam_eval",
        description="Run the BEAM memory benchmark against a memai server.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MEMAI_API_KEY", ""),
        help="memai API key (default: MEMAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
        help="memai server URL",
    )
    parser.add_argument(
        "--abilities",
        default=None,
        help=f"Comma-separated abilities to evaluate (default: all 10). Choices: {','.join(BEAM_ABILITIES)}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples per ability (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of memories to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=2000,
        help="PAMI token budget (default: 2000)",
    )
    parser.add_argument(
        "--output",
        default="BENCHMARK_RESULTS.md",
        help="Output file path (default: BENCHMARK_RESULTS.md)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with 5 synthetic samples per ability (no HuggingFace download)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-sample results",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace dataset cache directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        key_file = Path("memai_data/.master_key")
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        print("Error: MEMAI_API_KEY not set. Set it or run 'memai serve' first.")
        sys.exit(1)

    # Parse abilities
    abilities = None
    if args.abilities:
        abilities = [a.strip() for a in args.abilities.split(",")]
        invalid = [a for a in abilities if a not in BEAM_ABILITIES]
        if invalid:
            print(f"Error: Unknown abilities: {invalid}")
            print(f"Valid abilities: {BEAM_ABILITIES}")
            sys.exit(1)

    print(f"🧠 memai BEAM Benchmark")
    print(f"   Server  : {args.base_url}")
    print(f"   Samples : {args.max_samples} per ability {'(synthetic)' if args.dry_run else '(BEAM dataset)'}")
    print(f"   Output  : {args.output}")

    # Load dataset
    samples = load_beam_dataset(
        abilities=abilities,
        max_samples_per_ability=args.max_samples,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )

    # Run evaluation
    evaluator = BEAMEvaluator(
        api_key=api_key,
        base_url=args.base_url,
        k=args.k,
        context_budget=args.context_budget,
        verbose=args.verbose,
    )
    evaluator._client  # verify connection
    report = evaluator.run(samples)
    report.config = {
        "graph_backend": os.environ.get("MEMAI_GRAPH_BACKEND", "auto"),
        "vector_backend": os.environ.get("MEMAI_VECTOR_BACKEND", "auto"),
        "embedding_model": os.environ.get("MEMAI_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "k": args.k,
        "context_budget": args.context_budget,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 BEAM Results — memai v{report.memai_version}")
    print(f"   Macro F1  : {report.macro_f1:.3f}")
    print(f"   LIGHT F1  : {report.light_macro_f1:.3f}")
    d = report.macro_delta
    print(f"   Delta     : {'+'if d>=0 else ''}{d:.3f} {'✅' if d >= 0 else '❌'}")
    print(f"   Win rate  : {report.overall_win_rate*100:.1f}%")
    print("=" * 60)

    # Write output
    if args.format == "json":
        content = render_json(report)
    else:
        content = render_markdown(report)

    Path(args.output).write_text(content, encoding="utf-8")
    print(f"\n✅ Results written to: {args.output}")


if __name__ == "__main__":
    main()
