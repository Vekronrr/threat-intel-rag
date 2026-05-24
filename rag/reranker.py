"""
Cross-encoder reranking using cross-encoder/ms-marco-MiniLM-L-6-v2.
Reranks top-20 candidates down to top-5 for precision.
"""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from rag.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 5


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        logger.info("Loading cross-encoder reranker", model=model_name)
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = DEFAULT_TOP_N,
    ) -> list[RetrievedChunk]:
        """
        Score all (query, chunk) pairs with cross-encoder and return top_n.
        Cross-encoder attends to both query and passage jointly — much higher quality
        than bi-encoder similarity at the cost of O(n) inference.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for chunk, score in scored[:top_n]:
            chunk.rrf_score = float(score)  # overwrite with reranker score
            reranked.append(chunk)

        logger.info(
            "Reranking complete",
            input_count=len(chunks),
            output_count=len(reranked),
            top_score=float(scored[0][1]) if scored else 0,
        )
        return reranked
