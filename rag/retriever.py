"""
Two-stage hybrid retrieval: dense (ChromaDB) + sparse (BM25).
Merges results with Reciprocal Rank Fusion (RRF).
Returns top-K candidates before reranking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from ingestion.embedder import CHROMA_COLLECTION_CVES, CHROMA_COLLECTION_TECHNIQUES, get_chroma_client

logger = structlog.get_logger(__name__)

RRF_K = 60  # standard RRF constant
TOP_K = 20   # candidates before reranking


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def cve_id(self) -> str:
        return self.metadata.get("cve_id", self.chunk_id)


class HybridRetriever:
    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_CVES,
        persist_dir: str | None = None,
        openai_api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.chroma = get_chroma_client(persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.oai = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] | None = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in ChromaDB collection."""
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Collection is empty, BM25 index not built")
                return

            # Fetch all docs (BM25 needs the full corpus)
            results = self.collection.get(include=["documents", "metadatas"])
            docs = results.get("documents", [])
            if not docs:
                return

            tokenized = [doc.lower().split() for doc in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_docs = [
                {"text": doc, "metadata": meta, "id": id_}
                for doc, meta, id_ in zip(
                    docs, results.get("metadatas", []), results.get("ids", [])
                )
            ]
            logger.info("BM25 index built", doc_count=len(docs))
        except Exception as e:
            logger.warning("BM25 index build failed", error=str(e))

    async def _embed_query(self, query: str) -> list[float]:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        resp = await self.oai.embeddings.create(input=[query], model=model)
        return resp.data[0].embedding

    def _dense_search(
        self, query_embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict]:
        """ChromaDB cosine similarity search."""
        where = _build_chroma_filter(filters) if filters else None
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(1, self.collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        dense_results = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            dense_results.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine similarity
                "rank": i,
            })
        return dense_results

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 sparse retrieval."""
        if not self._bm25 or not self._bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self._bm25_docs[i]["id"],
                "text": self._bm25_docs[i]["text"],
                "metadata": self._bm25_docs[i]["metadata"],
                "score": float(scores[i]),
                "rank": rank,
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:
        """Merge dense and sparse rankings with RRF."""
        rrf_scores: dict[str, float] = {}
        all_docs: dict[str, dict] = {}

        for rank, doc in enumerate(dense):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            all_docs[doc_id] = {**doc, "dense_rank": rank}

        for rank, doc in enumerate(sparse):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
            if doc_id not in all_docs:
                all_docs[doc_id] = {**doc}
            all_docs[doc_id]["sparse_rank"] = rank

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        chunks = []
        for doc_id in sorted_ids:
            doc = all_docs[doc_id]
            chunks.append(RetrievedChunk(
                chunk_id=doc_id,
                text=doc["text"],
                metadata=doc["metadata"],
                dense_rank=doc.get("dense_rank"),
                sparse_rank=doc.get("sparse_rank"),
                rrf_score=rrf_scores[doc_id],
            ))
        return chunks

    async def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: dense + sparse → RRF merge.
        filters: {min_cvss, vendor, year_range}
        """
        query_embedding = await self._embed_query(query)

        dense = self._dense_search(query_embedding, top_k=top_k, filters=filters)
        sparse = self._sparse_search(query, top_k=top_k)

        merged = self._reciprocal_rank_fusion(dense, sparse, top_k=top_k)

        logger.info(
            "Hybrid retrieval",
            query=query[:60],
            dense_count=len(dense),
            sparse_count=len(sparse),
            merged_count=len(merged),
        )
        return merged

    def refresh_bm25(self) -> None:
        """Rebuild BM25 index after new data is ingested."""
        self._build_bm25_index()


def _build_chroma_filter(filters: dict) -> dict | None:
    """Convert API filter dict to ChromaDB $and/$or query syntax."""
    conditions = []

    min_cvss = filters.get("min_cvss")
    if min_cvss is not None:
        conditions.append({"cvss_score": {"$gte": float(min_cvss)}})

    vendor = filters.get("vendor")
    if vendor:
        conditions.append({"vendors": {"$contains": vendor}})

    year_range = filters.get("year_range")
    if year_range and len(year_range) == 2:
        start_year, end_year = year_range
        conditions.append({"published": {"$gte": f"{start_year}-01-01"}})
        conditions.append({"published": {"$lte": f"{end_year}-12-31"}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
