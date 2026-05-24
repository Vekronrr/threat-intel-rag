"""
Attack Surface Analyzer — given a list of software names, finds all relevant CVEs,
scores them, and returns a prioritized remediation plan.
"""

from __future__ import annotations

from collections import Counter

import structlog

from graph.graph_retriever import GraphRetriever
from graph.risk_scorer import RiskScorer
from rag.reranker import CrossEncoderReranker
from rag.retriever import HybridRetriever

logger = structlog.get_logger(__name__)

PATCH_TEMPLATE = (
    "Apply vendor security patch for {cve_id} ({severity}, VulnMind: {score}/100). "
    "CVSS: {cvss}. EPSS exploitation probability: {epss:.1%}. "
    "Check vendor advisory for {cve_id}."
)


class AttackSurfaceAnalyzer:
    def __init__(
        self,
        graph: GraphRetriever,
        scorer: RiskScorer,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ):
        self.graph = graph
        self.scorer = scorer
        self.retriever = retriever
        self.reranker = reranker

    async def analyze(self, software_list: list[str]) -> dict:
        """
        For each software entry, retrieve relevant CVEs, deduplicate, score,
        and return a prioritized attack surface report.
        """
        all_cve_ids: set[str] = set()
        cve_metadata: dict[str, dict] = {}

        for software in software_list:
            query = f"vulnerabilities affecting {software}"
            logger.info("Analyzing software", software=software)

            chunks = await self.retriever.retrieve(query, top_k=20)
            reranked = self.reranker.rerank(query, chunks, top_n=10)

            for chunk in reranked:
                cve_id = chunk.metadata.get("cve_id", "")
                if cve_id.startswith("CVE-") and cve_id not in all_cve_ids:
                    all_cve_ids.add(cve_id)
                    cve_metadata[cve_id] = {
                        "cvss_score": chunk.metadata.get("cvss_score"),
                        "severity": chunk.metadata.get("severity", "Unknown"),
                        "epss": chunk.metadata.get("epss"),
                        "software_match": software,
                    }

        # Score all unique CVEs
        scored_cves: list[dict] = []
        technique_counts: Counter = Counter()

        for cve_id in all_cve_ids:
            score_result = self.scorer.score(cve_id)
            meta = cve_metadata.get(cve_id, {})

            # Collect ATT&CK techniques via graph neighbors
            neighbors = self.graph.get_neighbors(cve_id, depth=1)
            for rel in neighbors.get("related", []):
                if rel.get("node_type") == "technique":
                    technique_counts[rel["id"]] += 1

            scored_cves.append({
                **score_result,
                "cvss_score": meta.get("cvss_score") or score_result["raw_values"].get("cvss"),
                "epss": meta.get("epss") or score_result["raw_values"].get("epss"),
                "software_match": meta.get("software_match", ""),
                "kev_confirmed": score_result.get("kev_confirmed", False),
            })

        scored_cves.sort(key=lambda x: x.get("vulnmind_score", 0), reverse=True)

        critical = [c for c in scored_cves if c.get("vulnmind_score", 0) >= 80]
        high = [c for c in scored_cves if 60 <= c.get("vulnmind_score", 0) < 80]
        top_techniques = [tid for tid, _ in technique_counts.most_common(5)]

        immediate_action = []
        for cve in scored_cves[:5]:
            cve_id = cve["cve_id"]
            score = cve.get("vulnmind_score", 0)
            severity = cve.get("severity_label", "Unknown")
            cvss = cve.get("cvss_score", "N/A")
            epss = cve.get("epss") or 0.0
            immediate_action.append({
                "cve_id": cve_id,
                "vulnmind_score": score,
                "severity_label": severity,
                "kev_confirmed": cve.get("kev_confirmed", False),
                "patch_recommendation": PATCH_TEMPLATE.format(
                    cve_id=cve_id,
                    severity=severity,
                    score=score,
                    cvss=cvss,
                    epss=float(epss),
                ),
            })

        logger.info(
            "Attack surface analysis complete",
            software_count=len(software_list),
            total_cves=len(scored_cves),
            critical=len(critical),
            high=len(high),
        )

        return {
            "software_analyzed": software_list,
            "total_cves_found": len(scored_cves),
            "critical_count": len(critical),
            "high_count": len(high),
            "cves": scored_cves,
            "top_techniques": top_techniques,
            "immediate_action": immediate_action,
        }
