"""
memai CLI — command-line interface.

Usage:
    memai serve [--host 0.0.0.0] [--port 8000]
    memai add "text" --agent my-agent
    memai search "query" --agent my-agent
    memai forget --agent my-agent --days 30
    memai health
    memai sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _get_client():
    """Build a MemaiClient from env vars for CLI use."""
    from memai.sdk import MemaiClient
    api_key = os.environ.get("MEMAI_API_KEY", "")
    if not api_key:
        # Try reading from the default key file
        key_file = os.path.join(
            os.environ.get("MEMAI_DATA_DIR", "./memai_data"), ".master_key"
        )
        if os.path.exists(key_file):
            api_key = open(key_file).read().strip()
    if not api_key:
        print("Error: MEMAI_API_KEY not set. Run 'memai serve' first and set the key.")
        sys.exit(1)
    base_url = os.environ.get("MEMAI_BASE_URL", "http://localhost:8000")
    agent_id = os.environ.get("MEMAI_AGENT_ID", "cli-agent")
    return MemaiClient(api_key=api_key, base_url=base_url, agent_id=agent_id)


def cmd_serve(args):
    """Start the memai API server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed: pip install uvicorn")
        sys.exit(1)
    from memai.api.app import create_app
    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


def cmd_add(args):
    """Add a memory."""
    client = _get_client()
    memory_id = client.add(
        text=args.text,
        agent_id=args.agent,
        memory_type=args.type,
        session_id=args.session,
    )
    print(f"✅ Memory added: {memory_id}")
    client.close()


def cmd_search(args):
    """Search memories."""
    client = _get_client()
    result = client.search(
        query=args.query,
        agent_id=args.agent,
        k=args.k,
        context_budget=args.budget,
    )
    if args.json:
        print(json.dumps({
            "pami_context": result.pami_context,
            "total": len(result.memories),
            "dropped": result.dropped_count,
            "memories": [
                {"id": m.id, "text": m.text[:80], "score": m.utility_score}
                for m in result.memories
            ]
        }, indent=2))
    else:
        print(f"\n🧠 memai search results ({len(result.memories)} memories)\n")
        print("=" * 60)
        print("PAMI Context (inject into LLM prompt):")
        print("-" * 60)
        print(result.pami_context or "(no memories found)")
        print("=" * 60)
    client.close()


def cmd_forget(args):
    """Delete stale memories."""
    client = _get_client()
    count = client.forget(
        agent_id=args.agent,
        older_than_days=args.days,
        staleness_threshold=args.threshold,
    )
    print(f"🗑️  Deleted {count} stale memories")
    client.close()


def cmd_health(args):
    """Check server health."""
    client = _get_client()
    h = client.health()
    print(f"✅ Server: {h.get('status')} | Version: {h.get('version')} | Uptime: {h.get('uptime_seconds'):.1f}s | Agents: {h.get('agents_loaded')}")
    client.close()


def cmd_sweep(args):
    """Run staleness sweep."""
    client = _get_client()
    result = client.sweep()
    agents = result.get("swept_agents", [])
    deleted = result.get("total_deleted", 0)
    print(f"🧹 Swept {len(agents)} agent(s), deleted {deleted} stale memories")
    client.close()


def cmd_list(args):
    """List memories for an agent."""
    client = _get_client()
    memories = client.list(agent_id=args.agent, limit=args.limit, offset=args.offset)
    if not memories:
        print("No memories found.")
    else:
        print(f"\n📋 {len(memories)} memories for agent '{args.agent}':\n")
        for m in memories:
            print(f"  [{m.memory_type:10s}] {m.id[:8]}... | score={m.utility_score:.2f} | {m.text[:70]}")
    client.close()


def cmd_workflow_save(args):
    """Save a workflow from a JSON file."""
    import json as _json
    with open(args.file) as f:
        spec = _json.load(f)
    client = _get_client()
    wf = client.save_workflow(
        name=spec["name"],
        trigger_pattern=spec["trigger_pattern"],
        steps=spec["steps"],
        agent_id=args.agent or spec.get("agent_id"),
    )
    print(f"✅ Workflow saved: {wf.workflow_id} ({wf.name})")
    client.close()


def main():
    parser = argparse.ArgumentParser(
        prog="memai",
        description="memai — Unified Agentic Memory CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the memai API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--log-level", default="info")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot reload (dev mode)")
    p_serve.set_defaults(func=cmd_serve)

    # add
    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text content")
    p_add.add_argument("--agent", default=None, help="Agent ID")
    p_add.add_argument("--type", default="auto", choices=["auto", "semantic", "event", "procedural"])
    p_add.add_argument("--session", default=None, help="Session ID")
    p_add.set_defaults(func=cmd_add)

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--agent", default=None)
    p_search.add_argument("--k", type=int, default=10, help="Max memories to retrieve")
    p_search.add_argument("--budget", type=int, default=2000, help="Token budget for PAMI context")
    p_search.add_argument("--json", action="store_true", help="Output as JSON")
    p_search.set_defaults(func=cmd_search)

    # forget
    p_forget = sub.add_parser("forget", help="Delete stale memories")
    p_forget.add_argument("--agent", default=None)
    p_forget.add_argument("--days", type=int, default=None, help="Delete memories older than N days")
    p_forget.add_argument("--threshold", type=float, default=0.1, help="Staleness threshold (0-1)")
    p_forget.set_defaults(func=cmd_forget)

    # list
    p_list = sub.add_parser("list", help="List memories for an agent")
    p_list.add_argument("--agent", default=None)
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.set_defaults(func=cmd_list)

    # health
    p_health = sub.add_parser("health", help="Check server health")
    p_health.set_defaults(func=cmd_health)

    # sweep
    p_sweep = sub.add_parser("sweep", help="Run staleness sweep across all agents")
    p_sweep.set_defaults(func=cmd_sweep)

    # workflow save
    p_wf = sub.add_parser("workflow-save", help="Save a workflow from JSON file")
    p_wf.add_argument("file", help="Path to workflow JSON file")
    p_wf.add_argument("--agent", default=None)
    p_wf.set_defaults(func=cmd_workflow_save)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
