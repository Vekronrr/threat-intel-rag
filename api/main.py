"""
VulnMind FastAPI server with Server-Sent Events streaming.
POST /query          — main threat intelligence endpoint
GET  /health         — system health check
POST /ingest         — trigger data ingestion
GET  /actor/{id}     — threat actor profile
GET  /trending       — EPSS trending CVEs
POST /analyze-surface — attack surface analysis
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from api.cache import SemanticCache
from api.models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryFilters,
    QueryRequest,
    QueryResponse,
    SourceAttribution,
    SurfaceAnalysisRequest,
    SurfaceAnalysisResponse,
)
from api.surface_analyzer import AttackSurfaceAnalyzer
from graph.graph_retriever import GraphRetriever
from graph.risk_scorer import RiskScorer
from ingestion.embedder import CHROMA_COLLECTION_CVES, CHROMA_COLLECTION_TECHNIQUES, get_chroma_client
from monitoring.drift_detector import EpssMonitor
from rag.agent import VulnMindAgent
from rag.faithfulness_checker import FaithfulnessChecker
from rag.prompt_builder import build_prompt
from rag.reranker import CrossEncoderReranker
from rag.retriever import HybridRetriever

load_dotenv()
logger = structlog.get_logger(__name__)

# Global singletons — initialized at startup
_retriever: HybridRetriever | None = None
_technique_retriever: HybridRetriever | None = None
_graph: GraphRetriever | None = None
_scorer: RiskScorer | None = None
_reranker: CrossEncoderReranker | None = None
_agent: VulnMindAgent | None = None
_cache: SemanticCache | None = None
_oai: AsyncOpenAI | None = None
_faithfulness_checker: FaithfulnessChecker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _retriever, _technique_retriever, _graph, _scorer, _reranker, _agent, _cache, _oai, _faithfulness_checker

    logger.info("VulnMind API starting up")

    _oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _cache = SemanticCache()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    _retriever = HybridRetriever(CHROMA_COLLECTION_CVES, persist_dir)
    _technique_retriever = HybridRetriever(CHROMA_COLLECTION_TECHNIQUES, persist_dir)
    _reranker = CrossEncoderReranker()
    _faithfulness_checker = FaithfulnessChecker()

    graph_path = Path("./data/vulnmind_graph.graphml")
    if graph_path.exists():
        _graph = GraphRetriever(graph_path)
        _scorer = RiskScorer(graph_path)
        _agent = VulnMindAgent(
            retriever=_retriever,
            technique_retriever=_technique_retriever,
            graph_retriever=_graph,
            risk_scorer=_scorer,
            reranker=_reranker,
        )
    else:
        logger.warning("Graph not found — graph-dependent features disabled. Run ingestion first.")

    logger.info("VulnMind API ready")
    yield

    logger.info("VulnMind API shutting down")


app = FastAPI(
    title="VulnMind Threat Intelligence API",
    description="Agentic RAG system for CVE and ATT&CK threat intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    chroma = get_chroma_client()
    try:
        cve_col = chroma.get_or_create_collection(CHROMA_COLLECTION_CVES)
        chroma_count = cve_col.count()
    except Exception:
        chroma_count = -1

    graph_nodes, graph_edges = 0, 0
    if _graph:
        graph_nodes = _graph.G.number_of_nodes()
        graph_edges = _graph.G.number_of_edges()

    redis_ok = await _cache.ping() if _cache else False

    return HealthResponse(
        status="healthy" if chroma_count >= 0 else "degraded",
        chroma_count=chroma_count,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        redis_connected=redis_ok,
    )


@app.post("/query")
async def query(req: QueryRequest) -> StreamingResponse | QueryResponse:
    """
    Main threat intelligence query endpoint.
    Supports streaming SSE when req.stream=True.
    """
    if not _retriever:
        raise HTTPException(503, "Retriever not initialized")

    filters_dict = req.filters.model_dump(exclude_none=True) if req.filters else None

    # Check semantic cache first
    cache_key = f"{req.question}|{json.dumps(filters_dict, sort_keys=True) if filters_dict else ''}"
    if _cache:
        cached = await _cache.get(cache_key)
        if cached:
            if req.stream:
                return _stream_cached(cached)
            return QueryResponse(**cached)

    if req.stream:
        return StreamingResponse(
            _stream_response(req.question, filters_dict, cache_key),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: use agent if available, else direct RAG
    if _agent:
        result = await _agent.arun(req.question, filters_dict)
        answer = result["answer"]
        agent_steps = result.get("intermediate_steps", [])
        faithfulness_result = None
    else:
        answer, faithfulness_result = await _direct_rag(req.question, filters_dict)
        agent_steps = []

    response = QueryResponse(
        answer=answer,
        query_id=str(uuid.uuid4()),
        agent_steps=agent_steps,
        faithfulness=faithfulness_result,
    )
    if _cache:
        await _cache.set(cache_key, response.model_dump())
    return response


async def _direct_rag(question: str, filters: dict | None) -> tuple[str, dict | None]:
    """Fallback when agent is unavailable: pure retrieval + LLM + faithfulness check."""
    chunks = await _retriever.retrieve(question, filters=filters)
    reranked = _reranker.rerank(question, chunks) if _reranker else chunks[:5]

    risk_scores = {}
    if _scorer:
        for chunk in reranked:
            cve_id = chunk.metadata.get("cve_id", "")
            if cve_id.startswith("CVE-"):
                risk_scores[cve_id] = _scorer.score(cve_id)

    system_prompt, user_message = build_prompt(question, reranked, risk_scores)

    resp = await _oai.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content

    faithfulness_result = None
    if _faithfulness_checker and reranked:
        faithfulness_result = _faithfulness_checker.check(
            answer, [c.text for c in reranked]
        )

    return answer, faithfulness_result


async def _stream_response(
    question: str, filters: dict | None, cache_key: str
) -> AsyncIterator[str]:
    """Stream LLM tokens via SSE."""
    chunks = await _retriever.retrieve(question, filters=filters)
    reranked = _reranker.rerank(question, chunks) if _reranker else chunks[:5]

    risk_scores = {}
    if _scorer:
        for chunk in reranked:
            cve_id = chunk.metadata.get("cve_id", "")
            if cve_id.startswith("CVE-"):
                risk_scores[cve_id] = _scorer.score(cve_id)

    from rag.prompt_builder import build_prompt
    system_prompt, user_message = build_prompt(question, reranked, risk_scores)

    # Send source metadata first
    sources = [
        {
            "chunk_id": c.chunk_id,
            "cve_id": c.cve_id,
            "text_preview": c.text[:200],
            "cvss_score": c.metadata.get("cvss_score"),
            "vulnmind_score": risk_scores.get(c.cve_id, {}).get("vulnmind_score"),
            "severity": c.metadata.get("severity"),
            "source": c.metadata.get("source", "nvd"),
        }
        for c in reranked
    ]
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

    # Stream LLM tokens
    full_answer = []
    stream = await _oai.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        stream=True,
    )

    async for chunk_event in stream:
        delta = chunk_event.choices[0].delta
        if delta.content:
            full_answer.append(delta.content)
            yield f"data: {json.dumps({'type': 'token', 'data': delta.content})}\n\n"

    # Faithfulness check on complete answer
    complete = "".join(full_answer)
    if _faithfulness_checker and reranked:
        faith_result = _faithfulness_checker.check(complete, [c.text for c in reranked])
        yield f"data: {json.dumps({'type': 'faithfulness', 'data': faith_result})}\n\n"

    # Cache complete response
    if _cache:
        response_dict = {"answer": complete, "sources": sources, "risk_scores": [], "agent_steps": []}
        await _cache.set(cache_key, response_dict)

    yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"


def _stream_cached(cached: dict) -> StreamingResponse:
    """Stream a previously cached response."""
    async def _gen():
        yield f"data: {json.dumps({'type': 'sources', 'data': cached.get('sources', [])})}\n\n"
        answer = cached.get("answer", "")
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            yield f"data: {json.dumps({'type': 'token', 'data': answer[i:i+chunk_size]})}\n\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Trigger data ingestion pipeline (async, returns immediately)."""
    asyncio.create_task(_run_ingestion(req.sources, req.delta_days))
    return IngestResponse(
        status="started",
        counts={},
        message="Ingestion started in background. Check /health for progress.",
    )


async def _run_ingestion(sources: list[str], delta_days: int | None) -> None:
    from ingestion.fetch_nvd import fetch_nvd
    from ingestion.fetch_mitre import fetch_mitre
    from ingestion.fetch_epss import fetch_epss
    from ingestion.embedder import run_ingestion_pipeline
    from graph.build_graph import build_graph

    if "nvd" in sources:
        await fetch_nvd(delta_days=delta_days)
    if "mitre" in sources:
        await fetch_mitre()
    if "epss" in sources:
        await fetch_epss()

    await run_ingestion_pipeline()
    build_graph()
    logger.info("Ingestion pipeline complete")


@app.get("/actor/{group_id}")
async def actor_profile(group_id: str) -> dict:
    """Return threat actor profile with associated techniques and CVEs."""
    if not _graph:
        raise HTTPException(503, "Graph not initialized. Run ingestion first.")

    profile = _graph.get_actor_profile(group_id.upper())
    if "error" in profile:
        raise HTTPException(404, profile["error"])

    cves = _graph.get_cves_by_actor(group_id.upper())

    # Enrich CVEs with VulnMind scores
    if _scorer:
        for cve in cves:
            score_data = _scorer.score(cve["cve_id"])
            cve["vulnmind_score"] = score_data.get("vulnmind_score")
            cve["kev_confirmed"] = score_data.get("kev_confirmed", False)

    return {
        "group_id": profile["group_id"],
        "name": profile.get("name", ""),
        "aliases": profile.get("aliases", []),
        "description": profile.get("description", ""),
        "techniques_used": profile.get("techniques_used", []),
        "associated_cves": cves,
    }


@app.get("/trending")
async def trending() -> dict:
    """Return top 10 CVEs with the largest recent EPSS increase."""
    monitor = EpssMonitor()
    trending_cves = monitor.get_trending(top_n=10)

    if _scorer:
        for item in trending_cves:
            cve_id = item["cve_id"]
            score_data = _scorer.score(cve_id)
            item["vulnmind_score"] = score_data.get("vulnmind_score")
            item["severity_label"] = score_data.get("severity_label", "Unknown")
            item["kev_confirmed"] = score_data.get("kev_confirmed", False)
    else:
        for item in trending_cves:
            item["vulnmind_score"] = None
            item["severity_label"] = "Unknown"
            item["kev_confirmed"] = False

    return {"trending": trending_cves, "count": len(trending_cves)}


@app.post("/analyze-surface", response_model=SurfaceAnalysisResponse)
async def analyze_surface(req: SurfaceAnalysisRequest) -> SurfaceAnalysisResponse:
    """Analyze attack surface for a list of software components."""
    if not _retriever or not _reranker:
        raise HTTPException(503, "Retriever not initialized")
    if not _graph or not _scorer:
        raise HTTPException(503, "Graph not initialized. Run ingestion first.")

    cache_key = f"surface|{json.dumps(sorted(req.software))}"
    if _cache:
        cached = await _cache.get(cache_key)
        if cached:
            return SurfaceAnalysisResponse(**cached)

    analyzer = AttackSurfaceAnalyzer(
        graph=_graph,
        scorer=_scorer,
        retriever=_retriever,
        reranker=_reranker,
    )
    result = await analyzer.analyze(req.software)

    response = SurfaceAnalysisResponse(**result)
    if _cache:
        # 1 hour TTL for surface analysis results
        await _cache.set(cache_key, result)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
