"""
MCP (Model Context Protocol) server for memai.

Exposes memai as an MCP stdio tool server for Claude Code.

Config (~/.config/claude/mcp_servers.json):
  {
    "memai": {
      "command": "python",
      "args": ["-m", "memai.connectors.mcp"],
      "env": {"MEMAI_API_KEY": "sk-memai-...", "MEMAI_AGENT_ID": "claude-code"}
    }
  }
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)
JSONRPC_VERSION = "2.0"
MCP_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "memai_add",
        "description": "Store a memory for future recall. Use when user shares a preference, fact, or context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "memory_type": {"type": "string", "enum": ["auto","semantic","event","procedural"], "default": "auto"},
                "session_id": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memai_search",
        "description": "Search memories. Returns PAMI context string — inject directly into LLM prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "context_budget": {"type": "integer", "default": 2000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memai_inject",
        "description": "Search + return just the PAMI context string for direct LLM injection.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 10}},
            "required": ["query"],
        },
    },
    {
        "name": "memai_forget",
        "description": "Delete stale or outdated memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "older_than_days": {"type": "integer"},
                "staleness_threshold": {"type": "number", "default": 0.1},
            },
        },
    },
    {
        "name": "memai_add_event",
        "description": "Add a causal event to the event graph for a session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "session_id": {"type": "string"},
                "caused_by": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memai_timeline",
        "description": "Get the chronological event timeline for a session.",
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


def _client():
    from memai.sdk import MemaiClient
    return MemaiClient(
        api_key=os.environ.get("MEMAI_API_KEY", ""),
        base_url=os.environ.get("MEMAI_BASE_URL", "http://localhost:8000"),
        agent_id=os.environ.get("MEMAI_AGENT_ID", "mcp-agent"),
    )


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()
    return _CLIENT


def _dispatch(name: str, args: dict) -> str:
    c = _get_client()
    agent_id = os.environ.get("MEMAI_AGENT_ID", "mcp-agent")

    if name == "memai_add":
        mid = c.add(
            text=args["text"],
            agent_id=agent_id,
            memory_type=args.get("memory_type", "auto"),
            session_id=args.get("session_id"),
        )
        return f"Stored. ID: {mid}"

    elif name == "memai_search":
        r = c.search(
            query=args["query"], agent_id=agent_id,
            k=args.get("k", 10), context_budget=args.get("context_budget", 2000),
        )
        return json.dumps({
            "pami_context": r.pami_context,
            "count": len(r.memories),
            "dropped": r.dropped_count,
        }, indent=2)

    elif name == "memai_inject":
        ctx = c.inject(query=args["query"], agent_id=agent_id, k=args.get("k", 10))
        return ctx or "(no relevant memories)"

    elif name == "memai_forget":
        n = c.forget(
            agent_id=agent_id,
            older_than_days=args.get("older_than_days"),
            staleness_threshold=args.get("staleness_threshold", 0.1),
        )
        return f"Deleted {n} memories."

    elif name == "memai_add_event":
        session_id = args.get("session_id", "mcp-default-session")
        resp = c._request("POST", f"/session/{session_id}/events", json={
            "text": args["text"],
            "entities": args.get("entities", []),
            "caused_by": args.get("caused_by"),
        })
        return f"Event added. ID: {resp.get('id', '?')}"

    elif name == "memai_timeline":
        events = c.get_timeline(args["session_id"])
        if not events:
            return "No events."
        return "\n".join(
            f"[{(e.timestamp or '')[:19]}] {e.id[:8]} | {e.text[:80]}"
            for e in events
        )

    return f"Unknown tool: {name}"


class MCPServer:
    def _send(self, req_id: Any, result: Any = None, error: Any = None):
        msg: dict = {"jsonrpc": JSONRPC_VERSION, "id": req_id}
        if error:
            msg["error"] = error
        else:
            msg["result"] = result
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def handle(self, line: str):
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._send(None, error={"code": -32700, "message": str(e)})
            return

        rid, method = req.get("id"), req.get("method", "")
        params = req.get("params", {})

        if method == "initialize":
            self._send(rid, {
                "protocolVersion": MCP_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "memai", "version": "0.1.0"},
            })
        elif method in ("notifications/initialized", "notifications/cancelled"):
            pass
        elif method == "tools/list":
            self._send(rid, {"tools": TOOLS})
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            try:
                text = _dispatch(name, args)
                self._send(rid, {"content": [{"type": "text", "text": text}], "isError": False})
            except Exception as e:
                self._send(rid, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
        elif method == "ping":
            self._send(rid, {})
        else:
            self._send(rid, error={"code": -32601, "message": f"Method not found: {method}"})

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if line:
                self.handle(line)


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    MCPServer().run()


if __name__ == "__main__":
    main()
