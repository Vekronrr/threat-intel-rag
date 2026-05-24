"""
Batch embedding with OpenAI text-embedding-3-large.
Handles rate limits, retries, and ChromaDB persistence.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.chunker import CVEChunk, chunk_cve, chunk_mitre_technique

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
BATCH_SIZE = 100  # OpenAI max is 2048 inputs, but keep batches small
CHROMA_COLLECTION_CVES = "cve_chunks"
CHROMA_COLLECTION_TECHNIQUES = "mitre_techniques"


def get_chroma_client(persist_dir: str | None = None) -> chromadb.ClientAPI:
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
async def _embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


async def embed_and_store(
    chunks: list[CVEChunk],
    collection_name: str = CHROMA_COLLECTION_CVES,
    persist_dir: str | None = None,
    openai_api_key: str | None = None,
) -> chromadb.Collection:
    """
    Embed chunks in batches and upsert to ChromaDB.
    Uses upsert to support incremental updates without duplication.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    oai_client = AsyncOpenAI(api_key=api_key)
    chroma = get_chroma_client(persist_dir)

    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    logger.info("Starting embedding", total=total, collection=collection_name)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c.text for c in batch]

        embeddings = await _embed_batch(oai_client, texts)

        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in batch],
        )

        logger.info(
            "Embedded batch",
            start=batch_start,
            end=batch_start + len(batch),
            total=total,
        )
        # Avoid rate limits: ~0.5s between batches
        await asyncio.sleep(0.5)

    logger.info("Embedding complete", collection=collection_name, count=total)
    return collection


async def run_ingestion_pipeline(
    nvd_path: Path = Path("./data/nvd_cves.jsonl"),
    mitre_path: Path = Path("./data/mitre_techniques.jsonl"),
    epss_path: Path = Path("./data/epss_scores.json"),
    persist_dir: str | None = None,
) -> None:
    """Full ingestion pipeline: load data → chunk → embed → store."""
    import json

    # Load EPSS
    epss_scores: dict[str, dict] = {}
    if epss_path.exists():
        epss_scores = json.loads(epss_path.read_text())
        logger.info("Loaded EPSS scores", count=len(epss_scores))

    # Embed CVE chunks
    if nvd_path.exists():
        cve_chunks: list[CVEChunk] = []
        with nvd_path.open() as f:
            for line in f:
                cve = json.loads(line.strip())
                epss = epss_scores.get(cve["cve_id"])
                chunks = chunk_cve(cve, epss=epss)
                cve_chunks.extend(chunks)
        logger.info("CVE chunks created", count=len(cve_chunks))
        await embed_and_store(cve_chunks, CHROMA_COLLECTION_CVES, persist_dir)

    # Embed MITRE technique chunks
    if mitre_path.exists():
        tech_chunks: list[CVEChunk] = []
        with mitre_path.open() as f:
            for line in f:
                tech = json.loads(line.strip())
                tech_chunks.append(chunk_mitre_technique(tech))
        logger.info("Technique chunks created", count=len(tech_chunks))
        await embed_and_store(tech_chunks, CHROMA_COLLECTION_TECHNIQUES, persist_dir)


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())
