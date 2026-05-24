"""
LangChain ReAct agent with VulnMind tools.
Tools: search_cves, graph_lookup, score_risk, explain_technique
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

from graph.graph_retriever import GraphRetriever
from graph.risk_scorer import RiskScorer
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from ingestion.embedder import CHROMA_COLLECTION_TECHNIQUES

logger = structlog.get_logger(__name__)

REACT_PROMPT_HUB = "hwchase17/react"


class VulnMindAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        technique_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        risk_scorer: RiskScorer,
        reranker: CrossEncoderReranker,
        openai_api_key: str | None = None,
    ):
        self.retriever = retriever
        self.technique_retriever = technique_retriever
        self.graph = graph_retriever
        self.scorer = risk_scorer
        self.reranker = reranker

        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0,
            streaming=True,
        )

        self._executor = self._build_agent()

    def _build_agent(self) -> AgentExecutor:
        tools = [
            Tool(
                name="search_cves",
                description=(
                    "Search the CVE knowledge base using hybrid retrieval. "
                    "Input: JSON with keys 'query' (required), 'filters' (optional dict with "
                    "min_cvss, vendor, year_range). "
                    "Returns top matching CVEs with descriptions and risk scores."
                ),
                func=self._tool_search_cves,
                coroutine=self._async_tool_search_cves,
            ),
            Tool(
                name="graph_lookup",
                description=(
                    "Look up a CVE or ATT&CK technique in the knowledge graph. "
                    "Input: a CVE ID (e.g. CVE-2021-44228) or technique ID (e.g. T1190). "
                    "Returns related nodes: techniques, vendors, CWEs within 2 hops."
                ),
                func=self._tool_graph_lookup,
            ),
            Tool(
                name="score_risk",
                description=(
                    "Compute the VulnMind composite risk score for a CVE. "
                    "Input: a CVE ID string. "
                    "Returns score 0-100 with CVSS, EPSS, and graph centrality components."
                ),
                func=self._tool_score_risk,
            ),
            Tool(
                name="explain_technique",
                description=(
                    "Retrieve detailed information about a MITRE ATT&CK technique. "
                    "Input: technique ID (e.g. T1190, T1059.001). "
                    "Returns technique description, tactics, platforms, and detection guidance."
                ),
                func=self._tool_explain_technique,
                coroutine=self._async_tool_explain_technique,
            ),
        ]

        prompt = hub.pull(REACT_PROMPT_HUB)

        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def _tool_search_cves(self, input_str: str) -> str:
        """Sync wrapper for async search."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_tool_search_cves(input_str)
        )

    async def _async_tool_search_cves(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            query = data.get("query", input_str)
            filters = data.get("filters")
        except (json.JSONDecodeError, AttributeError):
            query = str(input_str)
            filters = None

        chunks = await self.retriever.retrieve(query, filters=filters)
        reranked = self.reranker.rerank(query, chunks, top_n=5)

        results = []
        for chunk in reranked:
            cve_id = chunk.metadata.get("cve_id", "")
            score = self.scorer.score(cve_id) if cve_id.startswith("CVE-") else {}
            results.append({
                "cve_id": cve_id,
                "text": chunk.text[:500],
                "cvss_score": chunk.metadata.get("cvss_score"),
                "vulnmind_score": score.get("vulnmind_score"),
                "severity": chunk.metadata.get("severity"),
            })

        return json.dumps(results, indent=2)

    def _tool_graph_lookup(self, node_id: str) -> str:
        node_id = node_id.strip()
        result = self.graph.get_neighbors(node_id, depth=2)
        if "error" in result:
            paths = []
        else:
            paths = self.graph.find_attack_paths(node_id) if node_id.startswith("CVE-") else []
            result["attack_paths"] = paths
        return json.dumps(result, indent=2)

    def _tool_score_risk(self, cve_id: str) -> str:
        cve_id = cve_id.strip()
        score = self.scorer.score(cve_id)
        return json.dumps(score, indent=2)

    def _tool_explain_technique(self, technique_id: str) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self._async_tool_explain_technique(technique_id)
        )

    async def _async_tool_explain_technique(self, technique_id: str) -> str:
        technique_id = technique_id.strip()
        chunks = await self.technique_retriever.retrieve(
            f"ATT&CK technique {technique_id}",
            top_k=5,
        )
        if not chunks:
            return f"No information found for technique {technique_id}"

        parts = [c.text for c in chunks[:3]]
        return "\n\n---\n\n".join(parts)

    async def arun(self, question: str, filters: dict | None = None) -> dict[str, Any]:
        """Run the ReAct agent asynchronously."""
        input_text = question
        if filters:
            input_text += f"\n\nFilters to apply: {json.dumps(filters)}"

        result = await self._executor.ainvoke({"input": input_text})
        return {
            "answer": result.get("output", ""),
            "intermediate_steps": [
                {
                    "tool": step[0].tool,
                    "tool_input": step[0].tool_input,
                    "observation": str(step[1])[:200],
                }
                for step in result.get("intermediate_steps", [])
            ],
        }

    def run(self, question: str, filters: dict | None = None) -> dict[str, Any]:
        """Sync wrapper."""
        return asyncio.get_event_loop().run_until_complete(self.arun(question, filters))
