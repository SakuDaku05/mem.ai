# memai BEAM Benchmark Results

> Generated: 2026-06-27 (dry-run with synthetic samples — run full eval after server boot)  
> memai version: `0.1.0`  
> Total samples: 50 (5 per ability, synthetic)

## Summary

| Metric | memai | LIGHT Baseline | Delta |
|--------|-------|----------------|-------|
| **Macro F1** | **TBD** | 0.479 | **TBD** |
| Win rate | TBD | — | — |

> **To run the full benchmark:**
> ```bash
> memai serve &
> python -m memai.benchmarks.beam_eval --output BENCHMARK_RESULTS.md
> ```

## Results by Ability

| Ability | LIGHT Baseline | Notes |
|---------|---------------|-------|
| 🔵 Information Extraction | 0.612 | Dense retrieval via SemanticMemory |
| 🔵 Preference Following | 0.538 | Metadata-tagged preference memories |
| 🔵 Summarization | 0.571 | Compressed summaries in SemanticMemory |
| 🔵 Event Ordering | 0.443 | Causal graph (EventMemory + Kuzu) |
| 🔵 Conflict Resolution | 0.389 | StalenessDetector R3 (contradiction) |
| 🔵 Multi-Hop Reasoning | 0.421 | Causal chain traversal |
| 🔵 Personalization | 0.496 | Agent-scoped memory partitioning |
| 🔵 Instruction Following | 0.512 | ProceduralMemory workflow replay |
| 🔵 Long-Range Dependency | 0.378 | PAMI boundary injection |
| 🔵 Knowledge Update | 0.433 | StalenessDetector R2 (superseded) |

## Architecture Advantages vs LIGHT

| LIGHT limitation | memai solution |
|-----------------|----------------|
| Flat vector store — no causal links | EventMemory causal graph (Kuzu) |
| No staleness handling | StalenessDetector R1–R4 rules |
| Lost-in-the-middle retrieval | PAMI position-aware injection |
| Single memory type | Semantic + Event + Procedural |
| No re-ranking | UtilityScorer composite Q-scoring |

## How to Reproduce

```bash
# 1. Install
pip install -e ".[all]"

# 2. Start server
memai serve

# 3. Run BEAM (full dataset — requires HuggingFace access)
python -m memai.benchmarks.beam_eval \
    --max-samples 200 \
    --output BENCHMARK_RESULTS.md

# 4. Run LoCoMo
python -m memai.benchmarks.locomo_eval \
    --output locomo_results.md

# 5. Run LongMemEval
python -m memai.benchmarks.longmemeval \
    --output longmem_results.md

# Dry run (no internet required):
python -m memai.benchmarks.beam_eval --dry-run
```

---

_Results will be updated with full dataset scores as infrastructure is provisioned._
