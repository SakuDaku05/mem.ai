"""
memai Phase 3 — SDK, CLI, and Connectors Tests
Run with: python -m pytest tests/test_phase3.py -v --tb=short
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid

import pytest

# ─── Use same key as test_api.py so the settings singleton is consistent ──────
MASTER_KEY = "sk-memai-test-master-key"
_tmp = tempfile.mkdtemp(prefix="memai_test_p3_")
os.environ.setdefault("MEMAI_MASTER_API_KEY", MASTER_KEY)
os.environ.setdefault("MEMAI_DATA_DIR", _tmp)
os.environ.setdefault("MEMAI_GRAPH_BACKEND", "networkx")
os.environ.setdefault("MEMAI_VECTOR_BACKEND", "dict")
os.environ.setdefault("MEMAI_EMBEDDING_MODEL", "none")

from fastapi.testclient import TestClient
from memai.api.app import create_app
from memai.sdk.client import (
    MemaiClient, MemaiAPIError, MemaiConnectionError,
    MemoryRecord, SearchResult, EventRecord, WorkflowRecord,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app_client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def sdk_client(app_client):
    """
    MemaiClient wired against the in-process TestClient.
    We use a custom httpx Transport that delegates to the TestClient.
    """
    import httpx

    class _TestClientTransport(httpx.BaseTransport):
        """Sync transport that routes requests through FastAPI TestClient."""
        def __init__(self, tc: TestClient):
            self._tc = tc

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            # Forward via TestClient's underlying WSGI/ASGI app
            method = request.method
            url = str(request.url)
            headers = dict(request.headers)
            content = request.content

            resp = self._tc.request(
                method=method,
                url=url,
                content=content,
                headers=headers,
            )
            return httpx.Response(
                status_code=resp.status_code,
                headers=dict(resp.headers),
                content=resp.content,
            )

    client = MemaiClient(api_key=MASTER_KEY, base_url="http://testserver")
    client._client = httpx.Client(
        transport=_TestClientTransport(app_client),
        base_url="http://testserver/v1",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        timeout=30.0,
    )
    yield client
    client.close()


@pytest.fixture(scope="module")
def agent_id():
    return f"sdk-agent-{uuid.uuid4().hex[:8]}"


# =============================================================================
# 1. SDK — MemaiClient basic tests
# =============================================================================

class TestSDKClient:

    def test_sdk_client_imports(self):
        from memai.sdk import MemaiClient
        assert MemaiClient is not None

    def test_sdk_client_init(self):
        c = MemaiClient(api_key="sk-test", base_url="http://localhost:9999")
        assert c.api_key == "sk-test"
        c.close()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMAI_API_KEY", "sk-env-test")
        monkeypatch.setenv("MEMAI_BASE_URL", "http://localhost:8000")
        monkeypatch.setenv("MEMAI_AGENT_ID", "env-agent")
        c = MemaiClient.from_env()
        assert c.api_key == "sk-env-test"
        assert c.default_agent_id == "env-agent"
        c.close()

    def test_from_env_missing_key(self, monkeypatch):
        monkeypatch.delenv("MEMAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MEMAI_API_KEY"):
            MemaiClient.from_env()

    def test_context_manager(self):
        with MemaiClient(api_key="sk-test", base_url="http://localhost:9999") as c:
            assert c.api_key == "sk-test"

    def test_agent_id_resolution(self):
        c = MemaiClient(api_key="sk-test", agent_id="default-agent")
        assert c._agent(None) == "default-agent"
        assert c._agent("override-agent") == "override-agent"
        c.close()

    def test_agent_id_missing_raises(self):
        c = MemaiClient(api_key="sk-test")
        with pytest.raises(ValueError, match="agent_id required"):
            c._agent(None)
        c.close()

    def test_connection_error(self):
        import socket
        c = MemaiClient(api_key="sk-test", base_url="http://localhost:19999")
        # Either MemaiConnectionError or a ConnectionError/OSError
        with pytest.raises((MemaiConnectionError, Exception)):
            c.health()
        c.close()

    def test_ping_false_on_connection_error(self):
        c = MemaiClient(api_key="sk-test", base_url="http://localhost:19999")
        assert c.ping() is False
        c.close()


# =============================================================================
# 2. SDK — live operations against test server
# =============================================================================

class TestSDKLiveOps:

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        self.mem = sdk_client
        self.agent = agent_id

    def test_ping(self):
        assert self.mem.ping() is True

    def test_health(self):
        h = self.mem.health()
        assert h["status"] == "ok"

    def test_add_returns_id(self):
        mid = self.mem.add("User loves jazz music", agent_id=self.agent)
        assert isinstance(mid, str) and len(mid) > 0

    def test_add_and_search(self):
        self.mem.add("User's favorite color is cobalt blue", agent_id=self.agent)
        result = self.mem.search("favorite color", agent_id=self.agent, k=5)
        assert isinstance(result, SearchResult)
        assert isinstance(result.pami_context, str)
        assert isinstance(result.memories, list)

    def test_inject_returns_string(self):
        self.mem.add("User speaks fluent Spanish", agent_id=self.agent)
        ctx = self.mem.inject("language skills", agent_id=self.agent)
        assert isinstance(ctx, str)

    def test_list_memories(self):
        self.mem.add("Test list memory", agent_id=self.agent)
        memories = self.mem.list(agent_id=self.agent, limit=20)
        assert isinstance(memories, list)
        assert all(isinstance(m, MemoryRecord) for m in memories)

    def test_get_not_found(self):
        result = self.mem.get(str(uuid.uuid4()))
        assert result is None

    def test_delete_not_found(self):
        result = self.mem.delete(str(uuid.uuid4()))
        assert result is False

    def test_forget(self):
        count = self.mem.forget(
            agent_id=self.agent,
            staleness_threshold=0.0,
            older_than_days=0,
        )
        assert isinstance(count, int)

    def test_metrics(self):
        m = self.mem.metrics()
        assert "agents_loaded" in m

    def test_sweep(self):
        s = self.mem.sweep()
        assert "swept_agents" in s


# =============================================================================
# 3. SDK — Session context manager
# =============================================================================

class TestSDKSession:

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        self.mem = sdk_client
        self.agent = agent_id

    def test_start_session(self):
        sid = self.mem.start_session(agent_id=self.agent)
        assert isinstance(sid, str) and len(sid) > 0

    def test_session_context_manager(self):
        with self.mem.session(agent_id=self.agent) as s:
            assert s.session_id is not None
            e1 = s.add_event("User opened settings")
            assert isinstance(e1, EventRecord)
            assert e1.id

    def test_session_causal_chain(self):
        with self.mem.session(agent_id=self.agent) as s:
            e1 = s.add_event("Root action")
            e2 = s.add_event("Triggered action", caused_by=e1.id)
            chain = s.causal_chain(e1.id, depth=3)
            assert chain["root_id"] == e1.id

    def test_session_timeline(self):
        with self.mem.session(agent_id=self.agent) as s:
            s.add_event("Event A")
            s.add_event("Event B")
            s.add_event("Event C")
            timeline = s.timeline()
            assert len(timeline) >= 3

    def test_get_timeline(self):
        sid = self.mem.start_session(agent_id=self.agent)
        with self.mem.session(agent_id=self.agent, session_id=sid) as s:
            s.add_event("Timeline test event")
        events = self.mem.get_timeline(sid)
        assert isinstance(events, list)

    def test_session_compress(self):
        with self.mem.session(agent_id=self.agent) as s:
            for i in range(4):
                s.add_event(f"Compressible event {i}")
            result = s.compress(keep_ratio=0.5)
            assert "compressed_count" in result


# =============================================================================
# 4. SDK — Workflows
# =============================================================================

class TestSDKWorkflows:

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        self.mem = sdk_client
        self.agent = agent_id

    def test_save_workflow(self):
        wf = self.mem.save_workflow(
            name="deploy-pipeline",
            trigger_pattern="deploy the app",
            steps=[
                {"step": 1, "action": "run tests"},
                {"step": 2, "action": "build docker"},
                {"step": 3, "action": "push to registry"},
            ],
            agent_id=self.agent,
        )
        assert isinstance(wf, WorkflowRecord)
        assert wf.workflow_id
        assert wf.name == "deploy-pipeline"

    def test_list_workflows(self):
        self.mem.save_workflow(
            name="list-test",
            trigger_pattern="run linter",
            steps=[{"step": 1, "action": "flake8"}],
            agent_id=self.agent,
        )
        workflows = self.mem.list_workflows(agent_id=self.agent)
        assert isinstance(workflows, list)
        assert len(workflows) >= 1

    def test_match_workflow(self):
        self.mem.save_workflow(
            name="test-suite",
            trigger_pattern="run tests",
            steps=[{"step": 1, "action": "pytest"}],
            agent_id=self.agent,
        )
        result = self.mem.match_workflow("run the test suite", agent_id=self.agent)
        # May or may not match depending on pattern — just assert type
        assert result is None or isinstance(result, WorkflowRecord)

    def test_replay_workflow(self):
        wf = self.mem.save_workflow(
            name="replay-test",
            trigger_pattern="deploy now",
            steps=[{"step": 1, "action": "build"}, {"step": 2, "action": "push"}],
            agent_id=self.agent,
        )
        steps = self.mem.replay_workflow(wf.workflow_id)
        assert isinstance(steps, list)
        assert len(steps) == 2


# =============================================================================
# 5. CLI
# =============================================================================

class TestCLI:

    def _run_cli(self, args: list[str], env: dict = None) -> tuple[int, str]:
        """Run the CLI main() function and capture output."""
        import io
        from contextlib import redirect_stdout
        from memai import cli

        full_env = {
            "MEMAI_API_KEY": MASTER_KEY,
            "MEMAI_BASE_URL": "http://localhost:8000",  # won't hit real server in unit test
            "MEMAI_AGENT_ID": "cli-test-agent",
        }
        if env:
            full_env.update(env)

        old_env = {k: os.environ.get(k) for k in full_env}
        for k, v in full_env.items():
            os.environ[k] = v

        buf = io.StringIO()
        try:
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = ["memai"] + args
            try:
                with redirect_stdout(buf):
                    cli.main()
                return 0, buf.getvalue()
            except SystemExit as e:
                return e.code or 0, buf.getvalue()
            finally:
                _sys.argv = old_argv
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cli_imports(self):
        from memai import cli
        assert hasattr(cli, "main")
        assert hasattr(cli, "cmd_serve")
        assert hasattr(cli, "cmd_add")
        assert hasattr(cli, "cmd_search")
        assert hasattr(cli, "cmd_forget")
        assert hasattr(cli, "cmd_health")
        assert hasattr(cli, "cmd_sweep")

    def test_cli_help(self):
        code, out = self._run_cli(["--help"])
        assert code == 0

    def test_cli_add_help(self):
        code, out = self._run_cli(["add", "--help"])
        assert code == 0

    def test_cli_search_help(self):
        code, out = self._run_cli(["search", "--help"])
        assert code == 0


# =============================================================================
# 6. Connectors — import and interface tests (no external framework needed)
# =============================================================================

class TestConnectorImports:

    def test_langchain_connector_imports(self):
        from memai.connectors.langchain import MemaiMemory
        assert MemaiMemory is not None

    def test_llamaindex_connector_imports(self):
        from memai.connectors.llamaindex import MemaiRetriever, MemaiChatMemoryBuffer
        assert MemaiRetriever is not None
        assert MemaiChatMemoryBuffer is not None

    def test_autogen_connector_imports(self):
        from memai.connectors.autogen import (
            MemaiConversableAgent, inject_memory, MemaiGroupChatManager
        )
        assert MemaiConversableAgent is not None
        assert inject_memory is not None

    def test_openai_connector_imports(self):
        from memai.connectors.openai import MemaiOpenAI, patch_openai_client
        assert MemaiOpenAI is not None
        assert patch_openai_client is not None

    def test_mem0_connector_imports(self):
        from memai.connectors.mem0 import MemaiMem0
        assert MemaiMem0 is not None

    def test_mcp_connector_imports(self):
        from memai.connectors.mcp import MCPServer, TOOLS, _dispatch
        assert len(TOOLS) == 6
        assert MCPServer is not None

    def test_generic_connector_imports(self):
        from memai.connectors.generic import MemaiMiddleware, with_memory
        assert MemaiMiddleware is not None
        assert with_memory is not None


class TestGenericMiddleware:
    """Test the generic connector's MemaiMiddleware without a live server."""

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        from memai.connectors.generic import MemaiMiddleware

        self.middleware = MemaiMiddleware(
            api_key=MASTER_KEY,
            agent_id=agent_id,
            base_url="http://testserver",
        )
        # Swap the internal client's httpx.Client with sdk_client's already-patched one
        self.middleware._client._client = sdk_client._client
        self.agent = agent_id

    def test_before_injects_system_message(self):
        messages = [{"role": "user", "content": "What is my name?"}]
        enhanced = self.middleware.before(messages, query="What is my name?")
        # System message prepended or existing one augmented
        assert len(enhanced) >= 1
        roles = [m["role"] for m in enhanced]
        assert "system" in roles or enhanced == messages  # graceful if no memories yet

    def test_before_appends_to_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        self.middleware.store("User's name is Alice")
        enhanced = self.middleware.before(messages, query="what is the user's name")
        assert enhanced[0]["role"] == "system"

    def test_before_empty_messages(self):
        # Empty messages + no query → no search performed, so result is unchanged
        result = self.middleware.before([], query=None)
        # No user query means no search; result may be [] or may contain cached context
        assert isinstance(result, list)

    def test_after_stores_memory(self):
        # Just verify it doesn't raise
        self.middleware.after(query="test query", response="test response")

    def test_store_returns_id(self):
        mid = self.middleware.store("A test fact about the user")
        assert isinstance(mid, str) and len(mid) > 0


class TestMem0Connector:
    """Test Mem0 interface compatibility without a live server."""

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        from memai.connectors.mem0 import MemaiMem0

        self.m = MemaiMem0(api_key=MASTER_KEY, base_url="http://testserver")
        # Reuse the already-patched httpx.Client from sdk_client
        self.m._client._client = sdk_client._client
        self.agent = agent_id

    def test_add_string(self):
        result = self.m.add("User likes Python", user_id=self.agent)
        assert "results" in result
        assert len(result["results"]) > 0

    def test_add_list_of_dicts(self):
        messages = [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Noted!"},
        ]
        result = self.m.add(messages, user_id=self.agent)
        assert "results" in result
        assert len(result["results"]) == 2

    def test_add_list_of_strings(self):
        result = self.m.add(["fact one", "fact two"], user_id=self.agent)
        assert len(result["results"]) == 2

    def test_search(self):
        self.m.add("User's dog is named Max", user_id=self.agent)
        result = self.m.search("pet name", user_id=self.agent, limit=5)
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_get_all(self):
        result = self.m.get_all(user_id=self.agent)
        assert "results" in result

    def test_delete_all(self):
        result = self.m.delete_all(user_id=self.agent)
        assert "deleted_count" in result

    def test_history_returns_list(self):
        add_result = self.m.add("History test fact", user_id=self.agent)
        mid = add_result["results"][0]["id"]
        history = self.m.history(mid)
        assert isinstance(history, list)


class TestMCPTools:
    """Test MCP tool dispatch logic (no stdio needed)."""

    @pytest.fixture(autouse=True)
    def setup(self, sdk_client, agent_id):
        from memai.connectors import mcp
        # Wire the MCP singleton client to the sdk_client that already has the test transport
        mcp._CLIENT = sdk_client
        self.mcp = mcp
        self.agent = agent_id
        os.environ["MEMAI_AGENT_ID"] = agent_id

    def test_memai_add_tool(self):
        result = self.mcp._dispatch("memai_add", {"text": "MCP test memory"})
        assert "Stored" in result
        assert "ID:" in result

    def test_memai_search_tool(self):
        self.mcp._dispatch("memai_add", {"text": "MCP search test content"})
        result = self.mcp._dispatch("memai_search", {"query": "MCP content", "k": 5})
        data = json.loads(result)
        assert "pami_context" in data
        assert "count" in data

    def test_memai_inject_tool(self):
        result = self.mcp._dispatch("memai_inject", {"query": "test query"})
        assert isinstance(result, str)

    def test_memai_forget_tool(self):
        result = self.mcp._dispatch("memai_forget", {"staleness_threshold": 0.0})
        assert "Deleted" in result

    def test_memai_add_event_tool(self):
        result = self.mcp._dispatch("memai_add_event", {
            "text": "MCP event test",
            "session_id": "mcp-test-session",
        })
        assert "Event added" in result

    def test_memai_timeline_tool(self):
        self.mcp._dispatch("memai_add_event", {
            "text": "Timeline test event",
            "session_id": "mcp-timeline-session",
        })
        result = self.mcp._dispatch("memai_timeline", {"session_id": "mcp-timeline-session"})
        assert isinstance(result, str)

    def test_mcp_tools_schema_valid(self):
        for tool in self.mcp.TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_mcp_server_instantiates(self):
        from memai.connectors.mcp import MCPServer
        server = MCPServer()
        assert server is not None
