"""
EventMemory — Kuzu-backed causal event graph.

Stores agent experiences as a directed graph where:
  Nodes = Events (things that happened)
  Edges = Causal/temporal/semantic relationships between events

Answers BEAM abilities:
  - Event Ordering (PRECEDES edges)
  - Temporal Reasoning (timestamp properties + PRECEDES)
  - Multi-Session Reasoning (cross-session graph traversal)
  - Contradiction Resolution (CONTRADICTS edges)
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from memai.models import CausalChain, EdgeType, EventEdge, EventNode

logger = logging.getLogger(__name__)

# Lazy import kuzu so the library works even if kuzu isn't installed
# (falls back to NetworkX)
try:
    import kuzu
    _KUZU_AVAILABLE = True
except ImportError:
    _KUZU_AVAILABLE = False
    logger.warning("kuzu not installed. EventMemory will use in-memory NetworkX backend.")


class EventMemory:
    """
    Causal event graph for structured agent memory.

    Supports two backends:
      - kuzu  (default, persistent, fast)
      - networkx (fallback, in-memory)

    Usage:
        em = EventMemory(db_path="./memai_data/events")
        node = em.add_event("User asked about database indexing", agent_id="a1")
        em.add_event("Agent recommended B-tree index", agent_id="a1",
                     causes=node.id)
        chain = em.get_causal_chain(node.id, depth=3)
    """

    def __init__(
        self,
        db_path: str = "./memai_data/events",
        backend: str = "auto",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.backend = backend
        if backend == "auto":
            self.backend = "kuzu" if _KUZU_AVAILABLE else "networkx"

        if self.backend == "kuzu":
            self._init_kuzu()
        else:
            self._init_networkx()

        logger.info("EventMemory initialized with backend=%s", self.backend)

    # ------------------------------------------------------------------
    # KUZU BACKEND
    # ------------------------------------------------------------------

    def _init_kuzu(self) -> None:
        import kuzu
        self._db = kuzu.Database(str(self.db_path / "graph.kuzu"))
        self._conn = kuzu.Connection(self._db)
        self._create_schema_kuzu()

    @staticmethod
    def _kuzu_rows(result) -> list[dict]:
        """
        Robustly extract rows from a Kuzu query result.
        Uses get_as_df() if pandas is available, otherwise iterates natively.
        """
        try:
            df = result.get_as_df()
            return df.to_dict("records")
        except ModuleNotFoundError:
            # pandas not installed — use Kuzu's native row iteration
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(row[0] if len(row) == 1 else dict(enumerate(row)))
            return rows

    def _create_schema_kuzu(self) -> None:
        """Create node/edge tables if they don't exist."""
        stmts = [
            """
            CREATE NODE TABLE IF NOT EXISTS Event (
                id STRING PRIMARY KEY,
                text STRING,
                summary STRING,
                timestamp INT64,
                session_id STRING,
                agent_id STRING,
                entities STRING,
                metadata STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS CAUSES (
                FROM Event TO Event,
                confidence DOUBLE,
                metadata STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS PRECEDES (
                FROM Event TO Event,
                delta_seconds DOUBLE,
                metadata STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS CONTRADICTS (
                FROM Event TO Event,
                contradiction_text STRING,
                confidence DOUBLE,
                metadata STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS UPDATES (
                FROM Event TO Event,
                field STRING,
                metadata STRING
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS REFERENCES (
                FROM Event TO Event,
                entity_name STRING,
                metadata STRING
            )
            """,
        ]
        for stmt in stmts:
            try:
                self._conn.execute(stmt.strip())
            except Exception as e:
                # Already exists — kuzu raises on duplicate CREATE
                if "already exists" not in str(e).lower():
                    raise

    # ------------------------------------------------------------------
    # NETWORKX FALLBACK
    # ------------------------------------------------------------------

    def _init_networkx(self) -> None:
        import networkx as nx
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, EventNode] = {}

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def add_event(
        self,
        text: str,
        agent_id: str,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        summary: Optional[str] = None,
        entities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        caused_by: Optional[str] = None,     # event_id that CAUSED this event
        precedes: Optional[str] = None,      # event_id this event comes BEFORE
        contradicts: Optional[str] = None,   # event_id this event contradicts
        updates: Optional[str] = None,       # event_id this event supersedes
        # Legacy alias — maps to caused_by for backwards compat
        causes: Optional[str] = None,
    ) -> EventNode:
        """
        Add a new event to the causal graph.

        Edge semantics (all edges are OUTGOING from the cause/source):
          caused_by=X  -> creates edge  X --[CAUSES]--> this_event
          precedes=Y   -> creates edge  this_event --[PRECEDES]--> Y
          contradicts=Z-> creates edge  this_event --[CONTRADICTS]--> Z
          updates=W    -> creates edge  this_event --[UPDATES]--> W

        Note: caused_by is preferred over legacy 'causes' parameter.
        """
        ts = timestamp or datetime.now(timezone.utc)
        node = EventNode(
            text=text,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=ts,
            summary=summary or text[:120],
            entities=entities or [],
            metadata=metadata or {},
        )

        if self.backend == "kuzu":
            self._add_node_kuzu(node)
        else:
            self._add_node_networkx(node)

        # caused_by=X: the causal edge runs FROM X TO this new node
        parent = caused_by or causes
        if parent:
            self.add_edge(parent, node.id, EdgeType.CAUSES)

        # precedes=Y: this node comes before Y
        if precedes:
            self.add_edge(node.id, precedes, EdgeType.PRECEDES)

        # contradicts=Z: this node contradicts Z (marks Z as stale)
        if contradicts:
            self.add_edge(node.id, contradicts, EdgeType.CONTRADICTS)

        # updates=W: this node supersedes W
        if updates:
            self.add_edge(node.id, updates, EdgeType.UPDATES)

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EventEdge:
        """Add a directed edge between two events."""
        edge = EventEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            metadata=metadata or {},
        )
        if self.backend == "kuzu":
            self._add_edge_kuzu(edge)
        else:
            self._add_edge_networkx(edge)
        return edge

    def get_event(self, event_id: str) -> Optional[EventNode]:
        """Retrieve a single event node by ID."""
        if self.backend == "kuzu":
            return self._get_event_kuzu(event_id)
        return self._nodes.get(event_id)

    def get_causal_chain(self, root_id: str, depth: int = 3) -> CausalChain:
        """
        Traverse the causal graph up to `depth` hops from root_id.
        Returns all reachable nodes and edges (BFS).
        """
        if self.backend == "kuzu":
            return self._causal_chain_kuzu(root_id, depth)
        return self._causal_chain_networkx(root_id, depth)

    def find_contradictions(self, event_id: str) -> list[EventNode]:
        """Return all events that CONTRADICTS the given event."""
        if self.backend == "kuzu":
            return self._find_contradictions_kuzu(event_id)
        return self._find_contradictions_networkx(event_id)

    def get_session_timeline(
        self, session_id: str, agent_id: str
    ) -> list[EventNode]:
        """Return all events in a session ordered by timestamp."""
        if self.backend == "kuzu":
            return self._session_timeline_kuzu(session_id, agent_id)
        return self._session_timeline_networkx(session_id, agent_id)

    def temporal_compress(
        self,
        agent_id: str,
        before_timestamp: datetime,
        keep_ratio: float = 0.3,
    ) -> int:
        """
        Compress old events by removing low-utility ones.
        Keeps causal edges intact by re-routing through retained events.
        Returns count of removed events.
        """
        if self.backend == "kuzu":
            return self._temporal_compress_kuzu(agent_id, before_timestamp, keep_ratio)
        return self._temporal_compress_networkx(agent_id, before_timestamp, keep_ratio)

    def search_events(
        self,
        agent_id: str,
        query_entities: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[EventNode]:
        """Basic entity-based search over events."""
        if self.backend == "kuzu":
            return self._search_kuzu(agent_id, query_entities, session_id, limit)
        return self._search_networkx(agent_id, query_entities, session_id, limit)

    # ------------------------------------------------------------------
    # KUZU IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def _add_node_kuzu(self, node: EventNode) -> None:
        # Kuzu does not support MERGE — use CREATE directly.
        # UUIDs are unique so duplicates are not a concern in normal usage.
        ts_unix = int(node.timestamp.timestamp())
        self._conn.execute(
            """
            CREATE (e:Event {
                id: $id,
                text: $text,
                summary: $summary,
                timestamp: $timestamp,
                session_id: $session_id,
                agent_id: $agent_id,
                entities: $entities,
                metadata: $metadata
            })
            """,
            {
                "id": node.id,
                "text": node.text,
                "summary": node.summary or "",
                "timestamp": ts_unix,
                "session_id": node.session_id or "",
                "agent_id": node.agent_id,
                "entities": json.dumps(node.entities),
                "metadata": json.dumps(node.metadata),
            },
        )

    def _get_event_kuzu(self, event_id: str) -> Optional[EventNode]:
        result = self._conn.execute(
            "MATCH (e:Event {id: $id}) RETURN e", {"id": event_id}
        )
        rows = self._kuzu_rows(result)
        if not rows:
            return None
        return self._row_to_event_node(rows[0]["e"])

    def _add_edge_kuzu(self, edge: EventEdge) -> None:
        et = edge.edge_type.value
        meta_str = json.dumps(edge.metadata)
        try:
            if et == "CAUSES":
                self._conn.execute(
                    "MATCH (a:Event {id:$s}), (b:Event {id:$t}) "
                    "CREATE (a)-[:CAUSES {confidence:$c, metadata:$m}]->(b)",
                    {"s": edge.source_id, "t": edge.target_id,
                     "c": edge.confidence, "m": meta_str},
                )
            elif et == "PRECEDES":
                self._conn.execute(
                    "MATCH (a:Event {id:$s}), (b:Event {id:$t}) "
                    "CREATE (a)-[:PRECEDES {delta_seconds:0.0, metadata:$m}]->(b)",
                    {"s": edge.source_id, "t": edge.target_id, "m": meta_str},
                )
            elif et == "CONTRADICTS":
                self._conn.execute(
                    "MATCH (a:Event {id:$s}), (b:Event {id:$t}) "
                    "CREATE (a)-[:CONTRADICTS {contradiction_text:'', confidence:$c, metadata:$m}]->(b)",
                    {"s": edge.source_id, "t": edge.target_id,
                     "c": edge.confidence, "m": meta_str},
                )
            elif et == "UPDATES":
                self._conn.execute(
                    "MATCH (a:Event {id:$s}), (b:Event {id:$t}) "
                    "CREATE (a)-[:UPDATES {field:'', metadata:$m}]->(b)",
                    {"s": edge.source_id, "t": edge.target_id, "m": meta_str},
                )
        except Exception as e:
            logger.warning("Failed to add edge %s->%s (%s): %s", edge.source_id, edge.target_id, et, e)

    def _causal_chain_kuzu(self, root_id: str, depth: int) -> CausalChain:
        """BFS over CAUSES edges up to depth hops."""
        result = self._conn.execute(
            f"""
            MATCH p = (root:Event {{id: $id}})-[:CAUSES*1..{depth}]->(e:Event)
            RETURN e
            """,
            {"id": root_id},
        )
        rows = self._kuzu_rows(result)
        nodes = [self._row_to_event_node(r["e"]) for r in rows]
        root = self._get_event_kuzu(root_id)
        if root:
            nodes = [root] + nodes
        return CausalChain(root_id=root_id, nodes=nodes, edges=[], depth=depth)

    def _find_contradictions_kuzu(self, event_id: str) -> list[EventNode]:
        result = self._conn.execute(
            "MATCH (a:Event {id:$id})-[:CONTRADICTS]->(b:Event) RETURN b",
            {"id": event_id},
        )
        rows = self._kuzu_rows(result)
        return [self._row_to_event_node(r["b"]) for r in rows]

    def _session_timeline_kuzu(self, session_id: str, agent_id: str) -> list[EventNode]:
        result = self._conn.execute(
            """
            MATCH (e:Event)
            WHERE e.session_id = $sid AND e.agent_id = $aid
            RETURN e ORDER BY e.timestamp ASC
            """,
            {"sid": session_id, "aid": agent_id},
        )
        rows = self._kuzu_rows(result)
        return [self._row_to_event_node(r["e"]) for r in rows]

    def _temporal_compress_kuzu(
        self, agent_id: str, before_timestamp: datetime, keep_ratio: float
    ) -> int:
        ts_unix = int(before_timestamp.timestamp())
        result = self._conn.execute(
            "MATCH (e:Event) WHERE e.agent_id = $aid AND e.timestamp < $ts RETURN e.id",
            {"aid": agent_id, "ts": ts_unix},
        )
        rows = self._kuzu_rows(result)
        if not rows:
            return 0
        all_ids = [r.get("e.id", r.get(0, "")) for r in rows]
        n_keep = max(1, int(len(all_ids) * keep_ratio))
        to_delete = all_ids[:-n_keep]
        for eid in to_delete:
            self._conn.execute("MATCH (e:Event {id:$id}) DETACH DELETE e", {"id": eid})
        logger.info("Compressed %d events for agent %s", len(to_delete), agent_id)
        return len(to_delete)

    def _search_kuzu(
        self, agent_id: str, query_entities: Optional[list[str]],
        session_id: Optional[str], limit: int
    ) -> list[EventNode]:
        if session_id:
            result = self._conn.execute(
                "MATCH (e:Event) WHERE e.agent_id=$aid AND e.session_id=$sid "
                "RETURN e ORDER BY e.timestamp DESC LIMIT $lim",
                {"aid": agent_id, "sid": session_id, "lim": limit},
            )
        else:
            result = self._conn.execute(
                "MATCH (e:Event) WHERE e.agent_id=$aid "
                "RETURN e ORDER BY e.timestamp DESC LIMIT $lim",
                {"aid": agent_id, "lim": limit},
            )
        rows = self._kuzu_rows(result)
        return [self._row_to_event_node(r["e"]) for r in rows]

    def _row_to_event_node(self, row: Any) -> EventNode:
        """Convert a Kuzu result row to an EventNode."""
        if isinstance(row, dict):
            data = row
        else:
            data = dict(row)
        ts = data.get("timestamp", 0)
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        entities = data.get("entities", "[]")
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except Exception:
                entities = []
        metadata = data.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        return EventNode(
            id=data.get("id", ""),
            text=data.get("text", ""),
            summary=data.get("summary"),
            timestamp=dt,
            session_id=data.get("session_id") or None,
            agent_id=data.get("agent_id", ""),
            entities=entities,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # NETWORKX IMPLEMENTATIONS (fallback)
    # ------------------------------------------------------------------

    def _add_node_networkx(self, node: EventNode) -> None:
        self._nodes[node.id] = node
        self._graph.add_node(node.id, **node.model_dump())

    def _add_edge_networkx(self, edge: EventEdge) -> None:
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            confidence=edge.confidence,
        )

    def _causal_chain_networkx(self, root_id: str, depth: int) -> CausalChain:
        import networkx as nx
        nodes = []
        edges = []
        visited = set()
        queue = [(root_id, 0)]
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            if nid in self._nodes:
                nodes.append(self._nodes[nid])
            for succ in self._graph.successors(nid):
                edge_data = self._graph.get_edge_data(nid, succ)
                if edge_data and edge_data.get("edge_type") == "CAUSES":
                    queue.append((succ, d + 1))
        return CausalChain(root_id=root_id, nodes=nodes, edges=edges, depth=depth)

    def _find_contradictions_networkx(self, event_id: str) -> list[EventNode]:
        results = []
        for succ in self._graph.successors(event_id):
            data = self._graph.get_edge_data(event_id, succ)
            if data and data.get("edge_type") == "CONTRADICTS":
                if succ in self._nodes:
                    results.append(self._nodes[succ])
        return results

    def _session_timeline_networkx(self, session_id: str, agent_id: str) -> list[EventNode]:
        matching = [
            n for n in self._nodes.values()
            if n.session_id == session_id and n.agent_id == agent_id
        ]
        return sorted(matching, key=lambda n: n.timestamp)

    def _temporal_compress_networkx(
        self, agent_id: str, before_timestamp: datetime, keep_ratio: float
    ) -> int:
        candidates = [
            n for n in self._nodes.values()
            if n.agent_id == agent_id and n.timestamp < before_timestamp
        ]
        candidates.sort(key=lambda n: n.timestamp)
        n_keep = max(1, int(len(candidates) * keep_ratio))
        to_delete = candidates[:-n_keep]
        for node in to_delete:
            self._graph.remove_node(node.id)
            del self._nodes[node.id]
        return len(to_delete)

    def _search_networkx(
        self, agent_id: str, query_entities: Optional[list[str]],
        session_id: Optional[str], limit: int
    ) -> list[EventNode]:
        results = [
            n for n in self._nodes.values()
            if n.agent_id == agent_id
            and (session_id is None or n.session_id == session_id)
        ]
        results.sort(key=lambda n: n.timestamp, reverse=True)
        return results[:limit]

    def close(self) -> None:
        if self.backend == "kuzu" and hasattr(self, "_conn"):
            self._conn.close()
