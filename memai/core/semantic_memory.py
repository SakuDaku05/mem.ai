"""
SemanticMemory — ChromaDB-backed vector store for facts, preferences, summaries.

Answers BEAM abilities:
  - Information Extraction (dense retrieval of factual details)
  - Preference Following (tagged preference memories)
  - Summarization (stored compressed summaries)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from memai.models import MemoryItem, MemoryType

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed. SemanticMemory will use in-memory dict backend.")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class SemanticMemory:
    """
    Vector store for semantic/factual memories.

    Supports:
      - chromadb  (default, persistent)
      - dict      (fallback, in-memory, no embeddings)

    Usage:
        sm = SemanticMemory(db_path="./memai_data/semantic")
        sm.add("User prefers dark mode and Python 3.11", agent_id="a1")
        results = sm.search("what does the user prefer?", agent_id="a1", k=5)
    """

    # Default embedding model — lightweight, runs locally
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: str = "./memai_data/semantic",
        embedding_model: str = DEFAULT_MODEL,
        backend: str = "auto",
    ) -> None:
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self._embedder = None

        self.backend = backend
        if backend == "auto":
            self.backend = "chromadb" if _CHROMA_AVAILABLE else "dict"

        if self.backend == "chromadb":
            self._init_chroma()
        else:
            self._store: dict[str, MemoryItem] = {}

        logger.info("SemanticMemory initialized with backend=%s", self.backend)

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        self._client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # One collection per semantic memory type
        self._collection = self._client.get_or_create_collection(
            name="memai_semantic",
            metadata={"hnsw:space": "cosine"},
        )

    # Sentinel values that mean "no embedding — use keyword search"
    _NO_EMBED_SENTINELS = {"none", "disabled", "off", "no", "", "null"}

    def _get_embedder(self):
        """Lazy-load the sentence transformer.

        Returns None if embedding_model is a no-op sentinel or ST is unavailable.
        This causes automatic fallback to keyword-overlap search.
        """
        if self._embedder is not None:
            return self._embedder

        if self.embedding_model_name.lower() in self._NO_EMBED_SENTINELS:
            logger.info("Embedding model set to '%s' — using keyword search fallback", self.embedding_model_name)
            return None

        if not _ST_AVAILABLE:
            logger.warning("sentence-transformers not installed — using keyword search fallback")
            return None

        try:
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info("Loaded embedding model: %s", self.embedding_model_name)
        except Exception as e:
            logger.warning("Failed to load embedding model '%s': %s — falling back to keyword search", self.embedding_model_name, e)
            self._embedder = None
        return self._embedder

    def _embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed texts; returns None if no embedder available."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        return embedder.encode(texts, normalize_embeddings=True).tolist()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        agent_id: str,
        session_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryItem:
        """Embed and store a new semantic memory."""
        item = MemoryItem(
            text=text,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            metadata=metadata or {},
        )
        embeddings = self._embed([text])
        if embeddings:
            item.embedding = embeddings[0]

        if self.backend == "chromadb":
            self._add_chroma(item)
        else:
            self._store[item.id] = item

        return item

    def add_batch(self, items: list[dict[str, Any]], agent_id: str) -> list[MemoryItem]:
        """Batch-add multiple memories efficiently."""
        texts = [it["text"] for it in items]
        embeddings = self._embed(texts) or [None] * len(texts)

        results = []
        memory_items = []
        chroma_ids, chroma_docs, chroma_embeds, chroma_metas = [], [], [], []

        for it, emb in zip(items, embeddings):
            item = MemoryItem(
                text=it["text"],
                agent_id=agent_id,
                session_id=it.get("session_id"),
                memory_type=it.get("memory_type", MemoryType.SEMANTIC),
                metadata=it.get("metadata", {}),
                embedding=emb,
            )
            results.append(item)
            if self.backend == "chromadb":
                if emb:
                    chroma_ids.append(item.id)
                    chroma_docs.append(item.text)
                    chroma_embeds.append(emb)
                    chroma_metas.append(self._item_to_meta(item))
                else:
                    # No embedding available — add without embedding vector
                    # ChromaDB will use its internal embedder or skip
                    self._collection.add(
                        ids=[item.id],
                        documents=[item.text],
                        metadatas=[self._item_to_meta(item)],
                    )
            else:
                self._store[item.id] = item

        if self.backend == "chromadb" and chroma_ids:
            self._collection.add(
                ids=chroma_ids,
                documents=chroma_docs,
                embeddings=chroma_embeds,
                metadatas=chroma_metas,
            )
        return results

    def search(
        self,
        query: str,
        agent_id: str,
        k: int = 10,
        session_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryItem]:
        """Semantic search using cosine similarity."""
        if self.backend == "chromadb":
            return self._search_chroma(query, agent_id, k, session_id, filters)
        return self._search_dict(query, agent_id, k, session_id)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID."""
        if self.backend == "chromadb":
            return self._get_chroma(memory_id)
        return self._store.get(memory_id)

    def update(self, memory_id: str, text: str, agent_id: str) -> Optional[MemoryItem]:
        """Update an existing memory's text and re-embed."""
        item = self.get(memory_id)
        if not item:
            return None
        item.text = text
        embeddings = self._embed([text])
        if embeddings:
            item.embedding = embeddings[0]
        if self.backend == "chromadb":
            self._collection.update(
                ids=[memory_id],
                documents=[text],
                embeddings=[item.embedding] if item.embedding else None,
                metadatas=[self._item_to_meta(item)],
            )
        else:
            self._store[memory_id] = item
        return item

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if self.backend == "chromadb":
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False
        return self._store.pop(memory_id, None) is not None

    def delete_agent(self, agent_id: str) -> int:
        """Delete all memories for an agent."""
        if self.backend == "chromadb":
            results = self._collection.get(where={"agent_id": agent_id})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        to_del = [k for k, v in self._store.items() if v.agent_id == agent_id]
        for k in to_del:
            del self._store[k]
        return len(to_del)

    def list_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List all memories for an agent."""
        if self.backend == "chromadb":
            results = self._collection.get(
                where={"agent_id": agent_id},
                limit=limit,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            return self._chroma_results_to_items(results)
        items = [v for v in self._store.values() if v.agent_id == agent_id]
        return items[offset: offset + limit]

    def count(self, agent_id: str) -> int:
        """Count memories for an agent."""
        if self.backend == "chromadb":
            results = self._collection.get(where={"agent_id": agent_id})
            return len(results.get("ids", []))
        return sum(1 for v in self._store.values() if v.agent_id == agent_id)

    # ------------------------------------------------------------------
    # CHROMADB IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def _add_chroma(self, item: MemoryItem) -> None:
        kwargs: dict[str, Any] = {
            "ids": [item.id],
            "documents": [item.text],
            "metadatas": [self._item_to_meta(item)],
        }
        if item.embedding:
            kwargs["embeddings"] = [item.embedding]
        self._collection.add(**kwargs)

    def _search_chroma(
        self,
        query: str,
        agent_id: str,
        k: int,
        session_id: Optional[str],
        filters: Optional[dict[str, Any]],
    ) -> list[MemoryItem]:
        where: dict[str, Any] = {"agent_id": agent_id}
        if session_id:
            where["session_id"] = session_id
        if filters:
            where.update(filters)

        query_embs = self._embed([query])
        # Guard: ChromaDB raises if n_results < 1 or > collection size
        total = self._collection.count()
        if total == 0:
            return []  # nothing to search
        n = max(1, min(k, total))

        if query_embs:
            results = self._collection.query(
                query_embeddings=[query_embs[0]],
                n_results=n,
                where=where if where else None,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=n,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        return self._chroma_results_to_items(results, is_query=True)

    def _get_chroma(self, memory_id: str) -> Optional[MemoryItem]:
        results = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas", "embeddings"],
        )
        items = self._chroma_results_to_items(results)
        return items[0] if items else None

    def _item_to_meta(self, item: MemoryItem) -> dict[str, Any]:
        """Convert MemoryItem to ChromaDB metadata dict (must be str/int/float/bool)."""
        return {
            "agent_id": item.agent_id,
            "session_id": item.session_id or "",
            "memory_type": item.memory_type.value,
            "created_at": item.created_at.isoformat(),
            "access_count": item.access_count,
            "utility_score": item.utility_score,
            **{f"meta_{k}": str(v) for k, v in (item.metadata or {}).items()},
        }

    def _chroma_results_to_items(
        self,
        results: dict[str, Any],
        is_query: bool = False,
    ) -> list[MemoryItem]:
        items = []
        ids_list = results.get("ids", [[]] if is_query else [])
        docs_list = results.get("documents", [[]] if is_query else [])
        metas_list = results.get("metadatas", [[]] if is_query else [])
        embs_list = results.get("embeddings", None)

        # Query results are nested lists; get results are flat
        if is_query:
            ids_list = ids_list[0] if ids_list else []
            docs_list = docs_list[0] if docs_list else []
            metas_list = metas_list[0] if metas_list else []
            if embs_list:
                embs_list = embs_list[0]

        for i, (mid, doc, meta) in enumerate(zip(ids_list, docs_list, metas_list)):
            emb = embs_list[i] if embs_list else None
            item = MemoryItem(
                id=mid,
                text=doc,
                agent_id=meta.get("agent_id", ""),
                session_id=meta.get("session_id") or None,
                memory_type=MemoryType(meta.get("memory_type", "semantic")),
                access_count=int(meta.get("access_count", 0)),
                utility_score=float(meta.get("utility_score", 0.5)),
                embedding=list(emb) if emb is not None else None,
            )
            items.append(item)
        return items

    # ------------------------------------------------------------------
    # DICT FALLBACK IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def _search_dict(
        self,
        query: str,
        agent_id: str,
        k: int,
        session_id: Optional[str],
    ) -> list[MemoryItem]:
        """Simple keyword-overlap search when no vector backend available."""
        query_words = set(query.lower().split())
        scored = []
        for item in self._store.values():
            if item.agent_id != agent_id:
                continue
            if session_id and item.session_id != session_id:
                continue
            item_words = set(item.text.lower().split())
            overlap = len(query_words & item_words) / max(len(query_words), 1)
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]
