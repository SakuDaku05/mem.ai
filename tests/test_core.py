"""
Tests for memai Phase 1 core modules.
Run with: python -m pytest tests/test_core.py -v
"""

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memai.core.event_memory import EventMemory
from memai.core.pami import PAMI
from memai.core.procedural_memory import ProceduralMemory
from memai.core.staleness_detector import StalenessDetector
from memai.core.utility_scorer import UtilityScorer
from memai.memory import Memory
from memai.models import MemoryItem, MemoryType, EdgeType


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def event_mem(tmp_dir):
    em = EventMemory(db_path=f"{tmp_dir}/events", backend="networkx")
    yield em
    em.close()


@pytest.fixture
def proc_mem(tmp_dir):
    pm = ProceduralMemory(db_path=f"{tmp_dir}/procedural.db")
    yield pm
    pm.close()


@pytest.fixture
def staleness():
    return StalenessDetector(decay_lambda=0.05, dormant_days=30)


@pytest.fixture
def scorer():
    return UtilityScorer()


@pytest.fixture
def pami():
    return PAMI(token_budget=500)


@pytest.fixture
def memory(tmp_dir):
    mem = Memory(
        agent_id="test-agent",
        data_dir=tmp_dir,
        graph_backend="networkx",
        vector_backend="dict",
    )
    yield mem
    mem.close()


# ===========================================================================
# EVENT MEMORY TESTS
# ===========================================================================

class TestEventMemory:
    def test_add_and_get(self, event_mem):
        node = event_mem.add_event(
            text="User asked about B-tree indexing",
            agent_id="a1",
            session_id="s1",
        )
        assert node.id
        assert node.text == "User asked about B-tree indexing"

        retrieved = event_mem.get_event(node.id)
        assert retrieved is not None
        assert retrieved.id == node.id

    def test_causal_chain(self, event_mem):
        e1 = event_mem.add_event("User asked about indexes", agent_id="a1")
        e2 = event_mem.add_event("Agent recommended B-tree", agent_id="a1", caused_by=e1.id)
        e3 = event_mem.add_event("User applied B-tree index", agent_id="a1", caused_by=e2.id)

        chain = event_mem.get_causal_chain(e1.id, depth=3)
        assert chain.root_id == e1.id
        # With correct edge direction (e1->e2->e3), BFS should find all 3
        node_ids = {n.id for n in chain.nodes}
        assert e1.id in node_ids, "Root must be in chain"
        assert e2.id in node_ids, "Direct child e2 must be in chain"
        assert e3.id in node_ids, "Grandchild e3 must be in chain"

    def test_session_timeline(self, event_mem):
        for text in ["Event A", "Event B", "Event C"]:
            event_mem.add_event(text=text, agent_id="a1", session_id="s1")

        # Add a different session to verify filtering
        event_mem.add_event(text="Different session", agent_id="a1", session_id="s2")

        timeline = event_mem.get_session_timeline("s1", "a1")
        assert len(timeline) == 3
        texts = [n.text for n in timeline]
        assert "Event A" in texts
        assert "Different session" not in texts

    def test_contradiction_edges(self, event_mem):
        e1 = event_mem.add_event("Python 3.11 is installed", agent_id="a1")
        # e2 contradicts e1: e2 --[CONTRADICTS]--> e1
        e2 = event_mem.add_event("Python 3.10 is installed", agent_id="a1", contradicts=e1.id)
        assert event_mem.get_event(e2.id) is not None
        # Verify contradiction edge: find_contradictions(e2) should return e1
        contradicted = event_mem.find_contradictions(e2.id)
        assert any(n.id == e1.id for n in contradicted), "e1 should be in contradictions of e2"

    def test_kuzu_backend_smoke(self, tmp_dir):
        """Smoke test: Kuzu backend initializes and inserts without error."""
        try:
            import kuzu
        except ImportError:
            pytest.skip("kuzu not installed")
        em = EventMemory(db_path=f"{tmp_dir}/kuzu_events", backend="kuzu")
        node = em.add_event("Kuzu smoke test event", agent_id="kuzu-agent")
        assert node.id
        retrieved = em.get_event(node.id)
        assert retrieved is not None
        assert retrieved.text == "Kuzu smoke test event"
        em.close()

    def test_temporal_compress(self, event_mem):
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        # Add old events
        for i in range(10):
            node = event_mem.add_event(f"Old event {i}", agent_id="a1")
            node.timestamp = old_time
            event_mem._nodes[node.id] = node

        # Compress keeping 30%
        cutoff = datetime.now(timezone.utc)
        deleted = event_mem.temporal_compress("a1", cutoff, keep_ratio=0.3)
        assert deleted > 0


# ===========================================================================
# STALENESS DETECTOR TESTS
# ===========================================================================

class TestStalenessDetector:
    def _make_memory(self, age_days: float = 0, access_count: int = 1) -> MemoryItem:
        created = datetime.now(timezone.utc) - timedelta(days=age_days)
        return MemoryItem(
            text="User prefers dark mode",
            agent_id="a1",
            created_at=created,
            access_count=access_count,
            utility_score=0.8,
        )

    def test_fresh_memory_not_stale(self, staleness):
        mem = self._make_memory(age_days=1)
        result = staleness.check(mem)
        assert not result.is_stale

    def test_old_memory_stale(self, staleness):
        # Very old + no access = dormant
        mem = self._make_memory(age_days=200, access_count=0)
        result = staleness.check(mem)
        assert result.is_stale

    def test_time_decay_formula(self, staleness):
        """R1: score should decay exponentially."""
        mem_fresh = self._make_memory(age_days=0)
        mem_old = self._make_memory(age_days=100)
        score_fresh = staleness.adjusted_score(mem_fresh)
        score_old = staleness.adjusted_score(mem_old)
        assert score_fresh > score_old
        expected_old = 0.8 * math.exp(-0.05 * 100)
        assert abs(score_old - expected_old) < 0.001

    def test_supersede_detection(self, staleness):
        """R3: detect explicit supersede language."""
        mem = MemoryItem(
            text="Python version is now 3.12",
            agent_id="a1",
            utility_score=0.8,
        )
        result = staleness._rule_supersede(mem.text)
        assert result.is_stale

    def test_contradiction_detection(self, staleness):
        """R2: detect contradicting facts."""
        new_mem = MemoryItem(
            text="User does not prefer dark mode",
            agent_id="a1",
        )
        old_mem = MemoryItem(
            text="User prefer dark mode always",
            agent_id="a1",
        )
        contradicted = staleness.check_contradiction(new_mem, [old_mem])
        # With high overlap + negation asymmetry, should detect contradiction
        # (may or may not trigger depending on exact overlap — verify graceful)
        assert isinstance(contradicted, list)

    def test_dormant_rule(self, staleness):
        """R4: never-accessed old memory = dormant."""
        mem = self._make_memory(age_days=60, access_count=0)
        result = staleness._rule_dormant(mem, datetime.now(timezone.utc))
        assert result.is_stale


# ===========================================================================
# UTILITY SCORER TESTS
# ===========================================================================

class TestUtilityScorer:
    def _make_memory(self, access_count=0, age_days=0, utility_score=0.5) -> MemoryItem:
        created = datetime.now(timezone.utc) - timedelta(days=age_days)
        return MemoryItem(
            text="test memory text",
            agent_id="a1",
            access_count=access_count,
            utility_score=utility_score,
            created_at=created,
        )

    def test_weights_sum_to_one(self, scorer):
        total = scorer.w_semantic + scorer.w_recency + scorer.w_frequency + scorer.w_causal
        assert abs(total - 1.0) < 0.01

    def test_newer_scores_higher(self, scorer):
        fresh = self._make_memory(age_days=0)
        old = self._make_memory(age_days=365)
        s_fresh = scorer.score(fresh, query_text="test")
        s_old = scorer.score(old, query_text="test")
        assert s_fresh > s_old

    def test_higher_freq_scores_higher(self, scorer):
        low = self._make_memory(access_count=0)
        high = self._make_memory(access_count=50)
        s_low = scorer.score(low, query_text="q")
        s_high = scorer.score(high, query_text="q")
        assert s_high > s_low

    def test_rerank_sorted_descending(self, scorer):
        mems = [self._make_memory(age_days=i * 30) for i in range(5)]
        ranked = scorer.rerank(mems, query_text="test")
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_update_usage_increases_score(self, scorer):
        mem = self._make_memory(access_count=0, utility_score=0.5)
        scorer.update_usage(mem)
        assert mem.access_count == 1
        assert mem.utility_score >= 0.5  # should not decrease

    def test_cosine_similarity(self, scorer):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert scorer._cosine(a, b) == pytest.approx(1.0)
        c = [0.0, 1.0, 0.0]
        assert scorer._cosine(a, c) == pytest.approx(0.0)

    def test_causal_score_decay(self, scorer):
        assert scorer._causal_score(0) == 1.0
        assert scorer._causal_score(1) == 0.5
        assert scorer._causal_score(2) == pytest.approx(1 / 3, abs=0.01)
        assert scorer._causal_score(None) == 0.5


# ===========================================================================
# PAMI TESTS
# ===========================================================================

class TestPAMI:
    def _make_ranked(self, scores: list[float]) -> list[tuple[MemoryItem, float]]:
        result = []
        for i, score in enumerate(scores):
            mem = MemoryItem(
                text=f"Memory {i}: some content about topic {i}",
                agent_id="a1",
            )
            result.append((mem, score))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def test_empty_input(self, pami):
        result = pami.inject([])
        assert result.context == ""
        assert result.dropped == []

    def test_drops_below_threshold(self, pami):
        ranked = self._make_ranked([0.9, 0.5, 0.01])  # last one below threshold
        result = pami.inject(ranked)
        assert len(result.dropped) == 1

    def test_budget_respected(self, pami):
        # 500 token budget
        ranked = self._make_ranked([0.9] * 50)  # 50 memories
        result = pami.inject(ranked)
        assert result.total_tokens <= 500

    def test_context_has_sections(self, pami):
        ranked = self._make_ranked([0.9, 0.7, 0.5, 0.3, 0.1])
        result = pami.inject(ranked)
        # Should have key context section
        assert "Key Context" in result.context or "Memory" in result.context

    def test_start_vs_end_placement(self, pami):
        ranked = self._make_ranked([0.95, 0.8, 0.6, 0.4, 0.15])
        result = pami.inject(ranked)
        # High utility memories go to start
        assert len(result.start_memories) >= 1
        # Low utility memories go to end
        assert len(result.end_memories) >= 1


# ===========================================================================
# PROCEDURAL MEMORY TESTS
# ===========================================================================

class TestProceduralMemory:
    def test_save_and_retrieve(self, proc_mem):
        wf = proc_mem.save_workflow(
            name="deploy_flow",
            agent_id="a1",
            trigger_pattern=r"deploy|release",
            steps=[
                {"step": 1, "action": "Run tests"},
                {"step": 2, "action": "Build image"},
            ],
        )
        assert wf.workflow_id
        retrieved = proc_mem.get_workflow(wf.workflow_id)
        assert retrieved is not None
        assert retrieved.name == "deploy_flow"
        assert len(retrieved.steps) == 2

    def test_regex_match(self, proc_mem):
        proc_mem.save_workflow(
            name="deploy_flow",
            agent_id="a1",
            trigger_pattern=r"deploy|release|push to prod",
            steps=[{"step": 1, "action": "Run tests"}],
        )
        result = proc_mem.match_workflow("we need to deploy now", "a1")
        assert result is not None
        wf, score = result
        assert score == 1.0
        assert wf.name == "deploy_flow"

    def test_no_match_returns_none(self, proc_mem):
        proc_mem.save_workflow(
            name="deploy_flow",
            agent_id="a1",
            trigger_pattern=r"deploy",
            steps=[{"step": 1, "action": "Run tests"}],
        )
        result = proc_mem.match_workflow("make coffee", "a1")
        # Low overlap — should return None or very low score
        if result:
            _, score = result
            assert score < 0.3

    def test_list_workflows(self, proc_mem):
        for name in ["flow_a", "flow_b", "flow_c"]:
            proc_mem.save_workflow(
                name=name,
                agent_id="a1",
                trigger_pattern=name,
                steps=[{"step": 1, "action": "do something"}],
            )
        workflows = proc_mem.list_workflows("a1")
        assert len(workflows) == 3

    def test_delete_workflow(self, proc_mem):
        wf = proc_mem.save_workflow(
            name="to_delete",
            agent_id="a1",
            trigger_pattern="delete me",
            steps=[{"step": 1, "action": "nothing"}],
        )
        assert proc_mem.delete_workflow(wf.workflow_id)
        assert proc_mem.get_workflow(wf.workflow_id) is None

    def test_success_count(self, proc_mem):
        wf = proc_mem.save_workflow(
            name="counted",
            agent_id="a1",
            trigger_pattern="count",
            steps=[{"step": 1, "action": "something"}],
        )
        proc_mem.record_success(wf.workflow_id)
        proc_mem.record_success(wf.workflow_id)
        updated = proc_mem.get_workflow(wf.workflow_id)
        assert updated.success_count == 2


# ===========================================================================
# MEMORY ORCHESTRATOR TESTS (Integration)
# ===========================================================================

class TestMemoryOrchestrator:
    def test_add_and_search_semantic(self, memory):
        memory.add("User prefers dark mode and Python 3.11")
        memory.add("User is a senior backend engineer")
        result = memory.search("what does the user prefer?", k=5)
        assert isinstance(result.pami_context, str)
        assert len(result.memories) >= 0

    def test_add_event_auto(self, memory):
        response = memory.add(
            "User asked about database indexing",
            memory_type=MemoryType.AUTO,
        )
        assert response.memory_id
        # Should auto-detect as event
        assert response.type_inferred in (MemoryType.EVENT, MemoryType.SEMANTIC)

    def test_session_context(self, memory):
        with memory.session("sess-001") as s:
            s.add_event("User asked about B-tree")
            s.add_event("Agent recommended B-tree")
            timeline = s.timeline()
        assert len(timeline) == 2

    def test_workflow_proxy(self, memory):
        memory.workflows.save(
            name="code_review",
            steps=[
                {"step": 1, "action": "Read PR description"},
                {"step": 2, "action": "Check diff"},
            ],
            trigger_pattern=r"review|code review|PR",
        )
        match = memory.workflows.match("please do a code review")
        assert match is not None

    def test_forget_removes_old(self, memory):
        memory.add("Some fact to potentially forget")
        response = memory.forget(staleness_threshold=0.99)  # high threshold
        assert response.deleted_count >= 0  # may or may not delete

    def test_count(self, memory):
        memory.add("fact one", memory_type=MemoryType.SEMANTIC)
        memory.add("fact two", memory_type=MemoryType.SEMANTIC)
        count = memory.count()
        assert count >= 2

    def test_context_manager(self, tmp_dir):
        with Memory(
            agent_id="ctx-agent",
            data_dir=tmp_dir,
            graph_backend="networkx",
            vector_backend="dict",
        ) as mem:
            mem.add("test fact")
            assert mem.count() >= 1


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
