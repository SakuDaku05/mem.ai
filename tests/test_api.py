"""
memai Phase 2 — FastAPI API Tests
Run with: python -m pytest tests/test_api.py -v --tb=short

Uses FastAPI's TestClient (synchronous httpx wrapper) so no running server needed.
All tests use the master API key injected via env var override.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────
# Patch settings BEFORE importing the app so we
# get an isolated temp data dir and known API key
# ─────────────────────────────────────────────
MASTER_KEY = "sk-memai-test-master-key"
_tmp = tempfile.mkdtemp(prefix="memai_test_api_")

os.environ["MEMAI_MASTER_API_KEY"] = MASTER_KEY
os.environ["MEMAI_DATA_DIR"] = _tmp
os.environ["MEMAI_GRAPH_BACKEND"] = "networkx"    # fast, no Kuzu I/O
os.environ["MEMAI_VECTOR_BACKEND"] = "dict"        # fast, no ChromaDB I/O
os.environ["MEMAI_EMBEDDING_MODEL"] = "none"       # skip sentence-transformers


# Now import the app (settings are loaded lazily via get_settings())
from memai.api.app import create_app  # noqa: E402

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient — one app instance per test file."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers():
    return {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture(scope="module")
def agent_id():
    return f"test-agent-{uuid.uuid4().hex[:8]}"


# =============================================================================
# 1. Root + Health (no auth)
# =============================================================================

class TestRootAndHealth:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "memai"
        assert "docs" in body
        assert "health" in body

    def test_health_no_auth(self, client):
        """Health endpoint is open — no auth required."""
        r = client.get("/v1/admin/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "uptime_seconds" in body

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "memai — Unified Agentic Memory"


# =============================================================================
# 2. Auth
# =============================================================================

class TestAuth:
    def test_unauthenticated_request_returns_401(self, client):
        r = client.post("/v1/memory/add", json={"text": "hello", "agent_id": "x"})
        assert r.status_code == 401

    def test_master_key_auth(self, client, auth_headers):
        r = client.get("/v1/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["authenticated"] is True

    def test_create_agent_api_key(self, client, auth_headers):
        r = client.post(
            "/v1/auth/keys",
            json={"agent_id": "agent-alice"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == "agent-alice"
        assert body["api_key"].startswith("sk-memai-")

    def test_use_agent_api_key(self, client, auth_headers):
        # Create a key then use it
        r = client.post(
            "/v1/auth/keys",
            json={"agent_id": "agent-bob"},
            headers=auth_headers,
        )
        key = r.json()["api_key"]
        r2 = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {key}"})
        assert r2.status_code == 200
        assert r2.json()["agent_id"] == "agent-bob"

    def test_invalid_key_returns_401(self, client):
        r = client.get("/v1/auth/me", headers={"Authorization": "Bearer sk-memai-invalid"})
        assert r.status_code == 401

    def test_jwt_token_flow(self, client, auth_headers):
        # Get a JWT via /auth/token using master key
        r = client.post(
            "/v1/auth/token",
            data={"username": "__master__", "password": MASTER_KEY},
        )
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        # Use JWT
        jwt = body["access_token"]
        r2 = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {jwt}"})
        assert r2.status_code == 200


# =============================================================================
# 3. Memory Routes
# =============================================================================

class TestMemoryRoutes:
    @pytest.fixture(autouse=True)
    def setup(self, client, auth_headers, agent_id):
        self.client = client
        self.headers = auth_headers
        self.agent_id = agent_id

    def _add(self, text: str, memory_type: str = "semantic") -> dict:
        r = self.client.post(
            "/v1/memory/add",
            json={
                "text": text,
                "agent_id": self.agent_id,
                "memory_type": memory_type,
            },
            headers=self.headers,
        )
        assert r.status_code == 201, r.text
        return r.json()

    def test_add_semantic_memory(self):
        result = self._add("User prefers dark mode", "semantic")
        assert "memory_id" in result
        assert result["type_inferred"] in ("semantic", "auto")

    def test_add_auto_type_memory(self):
        result = self._add("User opened the settings panel", "auto")
        assert "memory_id" in result

    def test_add_event_memory(self):
        result = self._add("User logged in from NYC", "event")
        assert "memory_id" in result

    def test_add_procedural_memory(self):
        result = self._add(
            "deploy_app: step 1: run tests, step 2: build, step 3: push",
            "procedural",
        )
        assert "memory_id" in result

    def test_search_memories_returns_pami_context(self):
        # Add then search
        self._add("The user's favorite color is crimson red", "semantic")
        r = self.client.post(
            "/v1/memory/search",
            json={"query": "favorite color", "agent_id": self.agent_id, "k": 5},
            headers=self.headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "pami_context" in body
        assert "memories" in body
        assert isinstance(body["pami_context"], str)

    def test_search_empty_returns_empty(self):
        r = self.client.post(
            "/v1/memory/search",
            json={"query": "zzz_nonexistent_query_xyz", "agent_id": self.agent_id, "k": 5},
            headers=self.headers,
        )
        assert r.status_code == 200
        assert isinstance(r.json()["memories"], list)

    def test_list_memories(self):
        r = self.client.get(
            f"/v1/memory?limit=10&offset=0",
            headers=self.headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "memories" in body
        assert "total" in body

    def test_get_memory_by_id_not_found(self):
        r = self.client.get(
            f"/v1/memory/{uuid.uuid4()}",
            headers=self.headers,
        )
        assert r.status_code == 404

    def test_delete_memory_not_found(self):
        r = self.client.delete(
            f"/v1/memory/{uuid.uuid4()}",
            headers=self.headers,
        )
        assert r.status_code == 404

    def test_forget_memories(self):
        r = self.client.post(
            "/v1/memory/forget",
            json={
                "agent_id": self.agent_id,
                "older_than_days": 0,
                "staleness_threshold": 0.0,
            },
            headers=self.headers,
        )
        assert r.status_code == 200
        assert "deleted_count" in r.json()


# =============================================================================
# 4. Session Routes
# =============================================================================

class TestSessionRoutes:
    @pytest.fixture(autouse=True)
    def setup(self, client, auth_headers, agent_id):
        self.client = client
        self.headers = auth_headers
        self.agent_id = agent_id

    def _start_session(self) -> str:
        r = self.client.post(
            "/v1/session/start",
            json={"metadata": {"source": "test"}},
            headers=self.headers,
        )
        assert r.status_code == 201, r.text
        return r.json()["session_id"]

    def test_start_session_generates_id(self):
        session_id = self._start_session()
        assert isinstance(session_id, str) and len(session_id) > 0

    def test_start_session_with_explicit_id(self):
        my_id = "my-session-abc123"
        r = self.client.post(
            "/v1/session/start",
            json={"session_id": my_id},
            headers=self.headers,
        )
        assert r.status_code == 201
        assert r.json()["session_id"] == my_id

    def test_add_event_to_session(self):
        sid = self._start_session()
        r = self.client.post(
            f"/v1/session/{sid}/events",
            json={"text": "User clicked the dashboard button", "entities": ["dashboard"]},
            headers=self.headers,
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["text"] == "User clicked the dashboard button"
        assert "id" in body

    def test_add_causal_events(self):
        sid = self._start_session()
        r1 = self.client.post(
            f"/v1/session/{sid}/events",
            json={"text": "User submitted form"},
            headers=self.headers,
        )
        e1_id = r1.json()["id"]

        r2 = self.client.post(
            f"/v1/session/{sid}/events",
            json={"text": "Validation error triggered", "caused_by": e1_id},
            headers=self.headers,
        )
        assert r2.status_code == 201
        assert r2.json()["id"] != e1_id

    def test_get_session_timeline(self):
        sid = self._start_session()
        # Add 3 events
        for i in range(3):
            self.client.post(
                f"/v1/session/{sid}/events",
                json={"text": f"Event {i} in timeline test"},
                headers=self.headers,
            )
        r = self.client.get(f"/v1/session/{sid}/timeline", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert body["total"] >= 3

    def test_compress_session(self):
        sid = self._start_session()
        for i in range(5):
            self.client.post(
                f"/v1/session/{sid}/events",
                json={"text": f"Compressible event {i}"},
                headers=self.headers,
            )
        r = self.client.post(
            f"/v1/session/{sid}/compress",
            json={"keep_ratio": 0.5},
            headers=self.headers,
        )
        assert r.status_code == 200
        assert "compressed_count" in r.json()

    def test_causal_chain(self):
        sid = self._start_session()
        r1 = self.client.post(
            f"/v1/session/{sid}/events",
            json={"text": "Root event"},
            headers=self.headers,
        )
        e1_id = r1.json()["id"]
        r2 = self.client.post(
            f"/v1/session/{sid}/events",
            json={"text": "Child event", "caused_by": e1_id},
            headers=self.headers,
        )
        e2_id = r2.json()["id"]

        r = self.client.get(
            f"/v1/session/{sid}/causal/{e1_id}?depth=3",
            headers=self.headers,
        )
        assert r.status_code == 200
        chain = r.json()
        assert chain["root_id"] == e1_id
        assert chain["total"] >= 1


# =============================================================================
# 5. Workflow Routes
# =============================================================================

class TestWorkflowRoutes:
    @pytest.fixture(autouse=True)
    def setup(self, client, auth_headers, agent_id):
        self.client = client
        self.headers = auth_headers
        self.agent_id = agent_id

    def _save_workflow(self, name: str = "deploy") -> str:
        r = self.client.post(
            "/v1/workflow/save",
            json={
                "agent_id": self.agent_id,
                "name": name,
                "trigger_pattern": "deploy the app",
                "steps": [
                    {"step": 1, "action": "run pytest"},
                    {"step": 2, "action": "docker build"},
                    {"step": 3, "action": "kubectl apply"},
                ],
                "metadata": {},
            },
            headers=self.headers,
        )
        assert r.status_code == 201, r.text
        return r.json()["workflow_id"]

    def test_save_workflow(self):
        wid = self._save_workflow("deploy-pipeline")
        assert isinstance(wid, str) and len(wid) > 0

    def test_get_workflow_by_id(self):
        wid = self._save_workflow("get-test-workflow")
        r = self.client.get(f"/v1/workflow/{wid}", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert body["workflow_id"] == wid
        assert body["name"] == "get-test-workflow"

    def test_get_workflow_not_found(self):
        r = self.client.get(f"/v1/workflow/{uuid.uuid4()}", headers=self.headers)
        assert r.status_code == 404

    def test_list_workflows(self):
        self._save_workflow("list-test-wf-1")
        self._save_workflow("list-test-wf-2")
        r = self.client.get("/v1/workflow/list", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert "workflows" in body
        assert body["total"] >= 2

    def test_match_workflow(self):
        self._save_workflow("ci-cd-pipeline")
        r = self.client.post(
            "/v1/workflow/match",
            json={"query": "deploy the app to production", "agent_id": self.agent_id, "k": 3},
            headers=self.headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "matches" in body
        assert isinstance(body["matches"], list)

    def test_delete_workflow(self):
        wid = self._save_workflow("delete-me")
        r = self.client.delete(f"/v1/workflow/{wid}", headers=self.headers)
        assert r.status_code == 200
        # Verify gone
        r2 = self.client.get(f"/v1/workflow/{wid}", headers=self.headers)
        assert r2.status_code == 404

    def test_replay_workflow(self):
        wid = self._save_workflow("replay-test")
        r = self.client.post(
            f"/v1/workflow/{wid}/replay",
            json={"context": "deploying the new feature branch"},
            headers=self.headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "steps" in body


# =============================================================================
# 6. Admin Routes
# =============================================================================

class TestAdminRoutes:
    @pytest.fixture(autouse=True)
    def setup(self, client, auth_headers):
        self.client = client
        self.headers = auth_headers

    def test_metrics(self):
        r = self.client.get("/v1/admin/metrics", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert "agents_loaded" in body
        assert "agent_ids" in body

    def test_sweep(self):
        r = self.client.post("/v1/admin/sweep", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert "swept_agents" in body
        assert "total_deleted" in body

    def test_agents(self):
        r = self.client.get("/v1/admin/agents", headers=self.headers)
        assert r.status_code == 200
        body = r.json()
        assert "agents" in body
        assert "total" in body
