"""
Per-sentence faithfulness scoring using the reranker cross-encoder.
Flags answer sentences not grounded in retrieved context.
"""

from __future__ import annotations

import re

import structlog
from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_SENTENCE_LEN = 10


def _split_sentences(text: str) -> list[str]:
    """Split on '. ' and '.\n', strip, deduplicate empties."""
    parts = re.split(r"\.\s+|\.\n", text)
    return [s.strip() for s in parts if len(s.strip()) >= MIN_SENTENCE_LEN]


class FaithfulnessChecker:
    def __init__(self, model_name: str = RERANK_MODEL):
        logger.info("Loading faithfulness checker", model=model_name)
        self.model = CrossEncoder(model_name)

    def check(
        self,
        answer: str,
        context_chunks: list[str],
        threshold: float = 0.3,
    ) -> dict:
        """
        Score each answer sentence against all context chunks.
        A sentence is 'verified' if any context chunk scores >= threshold.

        Returns:
          verified_sentences: list[str]
          unverified_sentences: list[str]
          faithfulness_score: float (0.0–1.0)
          unverified_count: int
        """
        sentences = _split_sentences(answer)
        if not sentences or not context_chunks:
            return {
                "verified_sentences": [],
                "unverified_sentences": [],
                "faithfulness_score": 1.0 if not sentences else 0.0,
                "unverified_count": 0,
            }

        verified: list[str] = []
        unverified: list[str] = []

        for sentence in sentences:
            pairs = [(sentence, chunk) for chunk in context_chunks]
            scores = self.model.predict(pairs)
            max_score = float(max(scores))

            if max_score >= threshold:
                verified.append(sentence)
            else:
                unverified.append(sentence)

        total = len(sentences)
        faithfulness_score = len(verified) / total if total > 0 else 0.0

        logger.debug(
            "Faithfulness check",
            total=total,
            verified=len(verified),
            unverified=len(unverified),
            score=round(faithfulness_score, 3),
        )

        return {
            "verified_sentences": verified,
            "unverified_sentences": unverified,
            "faithfulness_score": round(faithfulness_score, 4),
            "unverified_count": len(unverified),
        }
