"""
memai FastAPI application — main entry point.

Creates the FastAPI app, registers all routers, sets up CORS,
lifespan events (startup/shutdown), and exception handlers.

Usage:
    uvicorn memai.api.app:app --host 0.0.0.0 --port 8000
    # OR via CLI:
    memai serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from memai import __version__
from memai.api.auth import auth_router
from memai.api.config import get_settings
from memai.api.manager import close_all
from memai.api.routes.admin import router as admin_router
from memai.api.routes.memory import router as memory_router
from memai.api.routes.session import router as session_router
from memai.api.routes.workflow import router as workflow_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("memai v%s starting — data_dir=%s", __version__, settings.data_dir)
    print(f"\n🧠 memai v{__version__} — Unified Agentic Memory")
    print(f"   Data dir  : {settings.data_dir}")
    print(f"   Graph     : {settings.graph_backend}")
    print(f"   Vectors   : {settings.vector_backend}")
    print(f"   Docs      : http://{settings.host}:{settings.port}/docs\n")
    yield
    logger.info("memai shutting down — closing all memory instances")
    await close_all()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="memai — Unified Agentic Memory",
        description="""
## memai REST API

The **memai** framework provides a unified, multi-layered memory system for LLM agents.

### Key Features
- **EventMemory** — Causal event graph (Kuzu) for temporal reasoning
- **SemanticMemory** — Vector store (ChromaDB) for semantic retrieval
- **ProceduralMemory** — Workflow store (SQLite) for instruction following
- **StalenessDetector** — Rule-based memory hygiene (R1-R4)
- **UtilityScorer** — Composite Q-style re-ranking
- **PAMI** — Position-Aware Memory Injection (solves lost-in-the-middle)

### Authentication
Use `Authorization: Bearer <api_key>` header.
Get your API key from the server startup logs or `POST /v1/auth/keys`.

### Quick Start
```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000/v1",
    headers={"Authorization": "Bearer sk-memai-..."}
)

# Add a memory
client.post("/memory/add", json={"text": "User prefers Python", "agent_id": "my-agent"})

# Search with PAMI context
result = client.post("/memory/search", json={"query": "user preferences", "agent_id": "my-agent"})
print(result.json()["pami_context"])  # inject into LLM prompt
```
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Register routers under /v1
    prefix = "/v1"
    app.include_router(auth_router, prefix=prefix)
    app.include_router(memory_router, prefix=prefix)
    app.include_router(session_router, prefix=prefix)
    app.include_router(workflow_router, prefix=prefix)
    app.include_router(admin_router, prefix=prefix)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "name": "memai",
                "version": __version__,
                "docs": "/docs",
                "health": "/v1/admin/health",
            }
        )

    return app


# Module-level app instance (for uvicorn)
app = create_app()
