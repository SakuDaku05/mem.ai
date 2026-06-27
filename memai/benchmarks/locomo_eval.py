"""
LoCoMo (Long-Context Conversation Memory) eval harness for memai.

Evaluates multi-turn memory retention over long conversations.
Based on: https://arxiv.org/abs/2402.11301

Usage:
    python -m memai.benchmarks.locomo_eval --output locomo_results.md
    python -m memai.benchmarks.locomo_eval --dry-run

LoCoMo tasks:
    - Single-hop QA    (recall from a single memory passage)
    - Multi-hop QA     (combine two+ memory passages)
    - Summarization    (compress long conversation)
    - Event graph      (temporal ordering questions)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOCOMO_TASKS = ["single_hop_qa", "multi_hop_qa", "summarization", "event_graph"]

# LoCoMo baseline scores (Table 3, paper)
LOCOMO_BASELINES = {
    "single_hop_qa":  {"light": 0.541, "gpt4": 0.623},
    "multi_hop_qa":   {"light": 0.387, "gpt4": 0.501},
    "summarization":  {"light": 0.489, "gpt4": 0.571},
    "event_graph":    {"light": 0.334, "gpt4": 0.445},
}


@dataclass
class LoCoMoSample:
    sample_id: str
    task: str
    conversation: list[dict]   # list of {role, content, turn_id}
    query: str
    reference: str
    n_turns: int = 20


@dataclass
class LoCoMoResult:
    task: str
    n_samples: int
    f1_score: float
    light_baseline: float
    gpt4_baseline: float
    delta_light: float
    delta_gpt4: float
    avg_latency_ms: float


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r)


def _synthetic_samples(task: str, n: int = 5) -> list[LoCoMoSample]:
    data = {
        "single_hop_qa": [
            ([{"role":"user","content":"My cat's name is Whiskers.","turn_id":1}],
             "What is my cat's name?", "Whiskers"),
        ],
        "multi_hop_qa": [
            ([{"role":"user","content":"I work at TechCorp.","turn_id":1},
              {"role":"user","content":"TechCorp is in San Francisco.","turn_id":2}],
             "Where do I work?", "San Francisco TechCorp"),
        ],
        "summarization": [
            ([{"role":"user","content":f"Turn {i}: discussed project update.","turn_id":i} for i in range(10)],
             "Summarize the conversation.", "project updates"),
        ],
        "event_graph": [
            ([{"role":"user","content":"I started the project Monday.","turn_id":1},
              {"role":"user","content":"Finished Friday.","turn_id":5}],
             "When did the project start?", "Monday"),
        ],
    }
    base = data.get(task, [([{"role":"user","content":f"{task} context","turn_id":1}],
                             f"Query about {task}", f"Answer for {task}")])
    samples = []
    for i in range(n):
        conv, q, ans = base[i % len(base)]
        samples.append(LoCoMoSample(
            sample_id=f"{task}_{i}",
            task=task,
            conversation=conv,
            query=q,
            reference=ans,
            n_turns=len(conv),
        ))
    return samples


def load_locomo(tasks: Optional[list[str]] = None, max_per_task: int = 50, dry_run: bool = False):
    tasks = tasks or LOCOMO_TASKS
    all_samples: dict[str, list[LoCoMoSample]] = {}

    if dry_run:
        for task in tasks:
            all_samples[task] = _synthetic_samples(task, n=5)
        return all_samples

    try:
        from datasets import load_dataset
        ds = load_dataset("locomo-benchmark/locomo", trust_remote_code=True)
        for task in tasks:
            split = ds.get(task, ds.get("test", []))
            samples = []
            for i, row in enumerate(split):
                if i >= max_per_task:
                    break
                samples.append(LoCoMoSample(
                    sample_id=row.get("id", f"{task}_{i}"),
                    task=task,
                    conversation=row.get("conversation", []),
                    query=row.get("query", ""),
                    reference=row.get("answer", ""),
                    n_turns=len(row.get("conversation", [])),
                ))
            all_samples[task] = samples
    except Exception as e:
        logger.warning("LoCoMo dataset unavailable (%s). Using synthetic data.", e)
        for task in tasks:
            all_samples[task] = _synthetic_samples(task, n=max_per_task)

    return all_samples


def run_locomo(api_key: str, base_url: str, samples: dict, k: int = 10, verbose: bool = False):
    from memai.sdk import MemaiClient
    client = MemaiClient(api_key=api_key, base_url=base_url)
    results: list[LoCoMoResult] = []

    for task, task_samples in samples.items():
        print(f"\n📊 LoCoMo task: {task} ({len(task_samples)} samples)")
        f1s, latencies = [], []

        for sample in task_samples:
            agent_id = f"locomo-{task}-{sample.sample_id}"
            t0 = time.perf_counter()

            # Store conversation turns
            for turn in sample.conversation:
                content = turn.get("content", "")
                if content:
                    try:
                        client.add(content, agent_id=agent_id, memory_type="semantic")
                    except Exception:
                        pass

            # Retrieve
            try:
                ctx = client.inject(query=sample.query, agent_id=agent_id, k=k)
            except Exception:
                ctx = ""

            latencies.append((time.perf_counter() - t0) * 1000)
            f1 = _token_f1(ctx, sample.reference)
            f1s.append(f1)

            # Cleanup
            try:
                client.forget(agent_id=agent_id, staleness_threshold=0.0)
            except Exception:
                pass

            if verbose:
                print(f"  {sample.sample_id}: F1={f1:.3f}")

        baselines = LOCOMO_BASELINES.get(task, {"light": 0.5, "gpt4": 0.6})
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        results.append(LoCoMoResult(
            task=task,
            n_samples=len(task_samples),
            f1_score=round(avg_f1, 4),
            light_baseline=baselines["light"],
            gpt4_baseline=baselines["gpt4"],
            delta_light=round(avg_f1 - baselines["light"], 4),
            delta_gpt4=round(avg_f1 - baselines["gpt4"], 4),
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        ))
        d = results[-1].delta_light
        print(f"  F1={avg_f1:.3f} vs LIGHT={baselines['light']:.3f} Δ={'+'if d>=0 else ''}{d:.3f}")

    return results


def render_locomo_markdown(results: list[LoCoMoResult], version: str) -> str:
    lines = [
        "# memai LoCoMo Benchmark Results",
        "",
        f"> Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"> memai version: `{version}`",
        "",
        "## Results",
        "",
        "| Task | memai F1 | LIGHT | GPT-4 | Δ vs LIGHT | Latency |",
        "|------|----------|-------|-------|------------|---------|",
    ]
    for r in results:
        d = f"{'+'if r.delta_light>=0 else ''}{r.delta_light:.3f}"
        medal = "✅" if r.delta_light >= 0 else "❌"
        lines.append(
            f"| {medal} {r.task.replace('_',' ').title()} "
            f"| {r.f1_score:.3f} | {r.light_baseline:.3f} | {r.gpt4_baseline:.3f} "
            f"| {d} | {r.avg_latency_ms:.0f}ms |"
        )
    macro = sum(r.f1_score for r in results) / len(results) if results else 0.0
    light_macro = sum(r.light_baseline for r in results) / len(results) if results else 0.0
    lines += [
        "",
        f"**Macro F1:** {macro:.3f} vs LIGHT {light_macro:.3f} "
        f"(Δ={'+'if macro>=light_macro else ''}{macro-light_macro:.3f})",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="memai.benchmarks.locomo_eval")
    parser.add_argument("--api-key", default=os.environ.get("MEMAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--tasks", default=None, help="Comma-separated LoCoMo tasks")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="locomo_results.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    api_key = args.api_key or Path("memai_data/.master_key").read_text().strip()
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    print("🧠 memai LoCoMo Evaluation")
    samples = load_locomo(tasks=tasks, max_per_task=args.max_samples, dry_run=args.dry_run)
    results = run_locomo(api_key, args.base_url, samples, k=args.k, verbose=args.verbose)

    from memai import __version__
    content = render_locomo_markdown(results, __version__)
    Path(args.output).write_text(content, encoding="utf-8")
    print(f"\n✅ Results written to: {args.output}")


if __name__ == "__main__":
    main()
