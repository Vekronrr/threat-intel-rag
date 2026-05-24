"""
Redis semantic cache to avoid redundant LLM calls.
Uses embedding cosine similarity to match semantically equivalent queries.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import redis.asyncio as aioredis
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

CACHE_TTL = 3600 * 24  # 24 hours
SIMILARITY_THRESHOLD = 0.92  # cosine similarity threshold for cache hit
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_CACHE_SCAN = 50  # max cached embeddings to compare per request


class SemanticCache:
    def __init__(
        self,
        redis_url: str | None = None,
        openai_api_key: str | None = None,
        ttl: int = CACHE_TTL,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.oai = AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.ttl = ttl
        self.threshold = similarity_threshold
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(self.redis_url, decode_responses=False)
        return self._redis

    async def _embed(self, text: str) -> list[float]:
        resp = await self.oai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return resp.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _cache_key(self, query: str) -> str:
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"vulnmind:cache:{h}"

    async def get(self, query: str) -> dict | None:
        """
        Check semantic cache. Returns cached response if a similar query exists.
        Strategy: exact hash check first, then embedding similarity scan.
        """
        try:
            r = await self._get_redis()

            # 1. Exact hash check (free)
            key = self._cache_key(query)
            raw = await r.get(key)
            if raw:
                logger.info("Cache hit (exact)", query=query[:60])
                return json.loads(raw)

            # 2. Semantic similarity scan
            query_emb = await self._embed(query)
            query_emb_bytes = json.dumps(query_emb).encode()

            # Get recent cache entries with embeddings
            pattern = "vulnmind:emb:*"
            cursor = 0
            candidates = []
            async for key_bytes in r.scan_iter(pattern, count=MAX_CACHE_SCAN):
                emb_raw = await r.get(key_bytes)
                if emb_raw:
                    stored = json.loads(emb_raw)
                    sim = self._cosine_similarity(query_emb, stored["embedding"])
                    if sim >= self.threshold:
                        candidates.append((sim, stored["result_key"]))

            if candidates:
                candidates.sort(reverse=True)
                best_key = candidates[0][1]
                result_raw = await r.get(best_key)
                if result_raw:
                    logger.info("Cache hit (semantic)", similarity=candidates[0][0], query=query[:60])
                    return json.loads(result_raw)

        except Exception as e:
            logger.warning("Cache get error", error=str(e))
        return None

    async def set(self, query: str, response: dict) -> None:
        """Store query response with embedding for future semantic matching."""
        try:
            r = await self._get_redis()

            # Store result by exact hash
            key = self._cache_key(query)
            await r.setex(key, self.ttl, json.dumps(response))

            # Store embedding index for semantic lookup
            query_emb = await self._embed(query)
            emb_key = f"vulnmind:emb:{key}"
            await r.setex(
                emb_key,
                self.ttl,
                json.dumps({"embedding": query_emb, "result_key": key}),
            )

            logger.info("Cached response", query=query[:60])
        except Exception as e:
            logger.warning("Cache set error", error=str(e))

    async def invalidate(self, pattern: str = "vulnmind:*") -> int:
        """Clear cache entries matching pattern."""
        r = await self._get_redis()
        count = 0
        async for key in r.scan_iter(pattern):
            await r.delete(key)
            count += 1
        return count

    async def ping(self) -> bool:
        try:
            r = await self._get_redis()
            return await r.ping()
        except Exception:
            return False
