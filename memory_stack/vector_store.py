from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


@dataclass
class MemoryChunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "memory_chunks"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

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
