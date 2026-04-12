from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


@dataclass
class MemoryChunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "memory_chunks"):
        self.client = self._init_client(persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    @staticmethod
    def _init_client(persist_dir: str) -> chromadb.ClientAPI:
        """Create a PersistentClient, recovering from corrupt/incompatible databases.

        ChromaDB 0.6+ with the Rust backend can fail on startup with an
        AttributeError ('RustBindingsAPI' has no attribute 'bindings') when
        native extensions aren't available, or a ValueError about the default
        tenant when the on-disk database was created by an incompatible version.
        On first failure we delete the directory and retry fresh; if that also
        fails we fall back to an in-memory EphemeralClient so the app stays up.
        """
        try:
            return chromadb.PersistentClient(path=persist_dir)
        except (ValueError, AttributeError, Exception) as exc:
            logger.warning(
                "ChromaDB PersistentClient failed (%s: %s); "
                "removing %r and retrying.",
                type(exc).__name__, exc, persist_dir,
            )
        # Retry with a fresh directory.
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
            return chromadb.PersistentClient(path=persist_dir)
        except Exception as exc2:
            logger.error(
                "ChromaDB PersistentClient still failing after reset (%s: %s); "
                "falling back to in-memory EphemeralClient. "
                "Vector memory will not persist across restarts.",
                type(exc2).__name__, exc2,
            )
            return chromadb.EphemeralClient()

    def add(self, chunks: List[MemoryChunk]):
        if not chunks:
            return
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def search(
        self,
        query: str,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryChunk]:
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filters or {},
        )
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        chunks: List[MemoryChunk] = []
        for cid, text, meta in zip(ids, docs, metas):
            chunks.append(MemoryChunk(id=cid, text=text, metadata=meta or {}))
        return chunks
