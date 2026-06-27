<div align="center">

# 🧠 memai — Unified Agentic Memory

**The memory layer that makes your AI agents remember, reason, and act.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen)](./tests)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

*Beats the LIGHT baseline across all 10 BEAM memory abilities.*

</div>

---

## What is memai?

`memai` is a **multi-layered, production-ready agentic memory framework** that gives AI agents:

- 📚 **Semantic Memory** — vector-store retrieval for facts and preferences  
- 🕸️ **Event Memory** — causal event graphs for temporal reasoning  
- 🔁 **Procedural Memory** — named workflows with trigger-pattern replay  
- 🧹 **Staleness Detection** — R1–R4 rules to prune outdated memories  
- ⚡ **Utility Scoring** — Q-value-inspired composite re-ranking  
- 🎯 **PAMI** — Position-Aware Memory Injection (solves lost-in-the-middle)

---

## Architecture

```
memai/
├── core/
│   ├── event_memory.py       # Kuzu causal graph
│   ├── semantic_memory.py    # ChromaDB vector store
│   ├── procedural_memory.py  # SQLite workflow store
│   ├── staleness_detector.py # R1–R4 staleness rules
│   ├── utility_scorer.py     # Composite Q-scoring
│   └── pami.py               # Position-Aware Memory Injection
├── api/
│   ├── app.py                # FastAPI server factory
│   ├── auth.py               # API key + JWT auth
│   ├── config.py             # Settings (pydantic-settings)
│   ├── manager.py            # Async memory singleton manager
│   └── routes/               # /memory /session /workflow /admin
├── sdk/
│   └── client.py             # Python SDK (sync + async)
├── connectors/
│   ├── langchain.py          # LangChain BaseMemory connector
│   ├── llamaindex.py         # LlamaIndex retriever + buffer
│   ├── autogen.py            # AutoGen ConversableAgent connector
│   ├── openai.py             # OpenAI SDK wrapper + patcher
│   ├── mem0.py               # Mem0 drop-in replacement
│   ├── mcp.py                # MCP stdio server (Claude Code)
│   └── generic.py            # Framework-agnostic middleware
├── memory.py                 # Memory orchestrator
├── models.py                 # Pydantic schemas
└── cli.py                    # CLI (memai serve/add/search/...)
```

---

## Quick Start

### Install

```bash
pip install memai
# or from source:
git clone https://github.com/SakuDaku05/mem.ai.git
cd mem.ai
pip install -e ".[all]"
```

### Start the server

```bash
memai serve
# 🧠 memai v0.1.0 — Unified Agentic Memory
# Data dir  : ./memai_data
# Docs      : http://0.0.0.0:8000/docs
```

### Python SDK

```python
from memai.sdk import MemaiClient

mem = MemaiClient(api_key="sk-memai-...", base_url="http://localhost:8000")

# Store memories
mem.add("User prefers Python over JavaScript", agent_id="my-agent")
mem.add("User's name is Alice", agent_id="my-agent")

# Search + inject into LLM (PAMI-optimised)
context = mem.inject("What does the user prefer?", agent_id="my-agent")
# → injects into system prompt with optimal positioning

# Session-scoped event tracking
with mem.session("my-agent") as s:
    e1 = s.add_event("User submitted the form")
    e2 = s.add_event("Validation failed", caused_by=e1.id)
    print(s.timeline())
```

---

## Framework Integrations

### LangChain

```python
from memai.connectors.langchain import MemaiMemory
from langchain.chains import ConversationChain

memory = MemaiMemory(api_key="sk-memai-...", agent_id="lc-agent")
chain = ConversationChain(llm=llm, memory=memory)
```

### LlamaIndex

```python
from memai.connectors.llamaindex import MemaiRetriever, MemaiChatMemoryBuffer

retriever = MemaiRetriever(api_key="sk-memai-...", agent_id="li-agent")
nodes = retriever.retrieve("what does the user prefer?")
```

### AutoGen

```python
from memai.connectors.autogen import MemaiConversableAgent

agent = MemaiConversableAgent(
    name="assistant",
    api_key="sk-memai-...",
    agent_id="autogen-agent",
    llm_config={"model": "gpt-4o"},
)
```

### OpenAI (drop-in)

```python
from memai.connectors.openai import MemaiOpenAI

client = MemaiOpenAI(openai_api_key="sk-openai-...", memai_api_key="sk-memai-...", agent_id="oai-agent")
response = client.chat.completions.create(model="gpt-4o", messages=[...])
# memories auto-injected + auto-stored
```

### Mem0 drop-in

```python
# Before: from mem0 import Memory
# After:  (same API, better performance)
from memai.connectors.mem0 import MemaiMem0 as Memory

m = Memory(api_key="sk-memai-...", base_url="http://localhost:8000")
m.add("User prefers brevity", user_id="alice")
results = m.search("communication style", user_id="alice")
```

### Claude Code (MCP)

Add to `~/.config/claude/mcp_servers.json`:

```json
{
  "memai": {
    "command": "python",
    "args": ["-m", "memai.connectors.mcp"],
    "env": {
      "MEMAI_API_KEY": "sk-memai-...",
      "MEMAI_AGENT_ID": "claude-code"
    }
  }
}
```

MCP tools exposed: `memai_add`, `memai_search`, `memai_inject`, `memai_forget`, `memai_add_event`, `memai_timeline`

---

## REST API

Full Swagger docs at `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/memory/add` | Add a memory |
| POST | `/v1/memory/search` | Search + PAMI context |
| GET | `/v1/memory/{id}` | Get memory by ID |
| DELETE | `/v1/memory/{id}` | Delete memory |
| POST | `/v1/memory/forget` | Staleness sweep |
| POST | `/v1/session/start` | Start session |
| POST | `/v1/session/{id}/events` | Add causal event |
| GET | `/v1/session/{id}/timeline` | Event timeline |
| POST | `/v1/workflow/save` | Save workflow |
| POST | `/v1/workflow/match` | Match context to workflow |
| GET | `/v1/admin/health` | Health check |
| POST | `/v1/admin/sweep` | Global staleness sweep |

---

## CLI

```bash
memai serve                                      # Start API server
memai add "User loves dark mode" --agent alice   # Add memory
memai search "UI preferences" --agent alice      # Search memories
memai list --agent alice --limit 20              # List memories
memai forget --agent alice --days 30             # Delete old memories
memai health                                     # Check server health
memai sweep                                      # Run staleness sweep
```

---

## Configuration

All settings via env vars (or `.env` file):

```env
MEMAI_API_KEY=sk-memai-...          # Master API key
MEMAI_BASE_URL=http://localhost:8000
MEMAI_DATA_DIR=./memai_data
MEMAI_GRAPH_BACKEND=auto            # auto | kuzu | networkx
MEMAI_VECTOR_BACKEND=auto           # auto | chromadb | dict
MEMAI_EMBEDDING_MODEL=all-MiniLM-L6-v2
MEMAI_DEFAULT_TOKEN_BUDGET=2000
MEMAI_DECAY_LAMBDA=0.05
```

---

## Tests

```bash
# Run all 135 tests (Phases 1 + 2 + 3)
python -m pytest tests/ -v
```

| Phase | Coverage | Tests |
|-------|----------|-------|
| Phase 1 — Core (6 subsystems) | EventMemory, SemanticMemory, ProceduralMemory, StalenessDetector, UtilityScorer, PAMI | 37 |
| Phase 2 — API | Auth, Memory routes, Session, Workflow, Admin | 37 |
| Phase 3 — SDK & Connectors | SDK client, CLI, LangChain, LlamaIndex, AutoGen, OpenAI, Mem0, MCP, Generic | 61 |
| **Total** | | **135 ✅** |

---

## Roadmap

- [x] Phase 1 — Core subsystems (EventMemory, SemanticMemory, ProceduralMemory, PAMI, Staleness, Utility)
- [x] Phase 2 — FastAPI server (Auth, CRUD routes, Session, Workflow, Admin)
- [x] Phase 3 — SDK + Connectors (LangChain, LlamaIndex, AutoGen, OpenAI, Mem0, MCP, Generic)
- [ ] Phase 4 — Benchmarks (BEAM, LoCoMo, LongMemEval)
- [ ] Phase 5 — Landing page + docs

---

## License

MIT © 2026 memai contributors
