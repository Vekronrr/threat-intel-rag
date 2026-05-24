"""
Composite VulnMind Risk Score.
Score = 0.4 * normalized_CVSS + 0.4 * EPSS_score + 0.2 * graph_centrality_percentile
KEV boost: multiply raw score by 1.15 if confirmed in CISA KEV catalog, capped at 100.
Output: 0-100 VulnMind Risk Score.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import structlog

from graph.build_graph import GRAPH_PATH, NODE_CVE, load_graph
from ingestion.fetch_kev import load_kev

logger = structlog.get_logger(__name__)

KEV_BOOST = 1.15


class RiskScorer:
    def __init__(self, graph_path: Path = GRAPH_PATH):
        self.G = load_graph(graph_path)
        self._kev: dict[str, dict] = load_kev()
        self._compute_percentiles()
        logger.info("KEV catalog loaded", kev_count=len(self._kev))

    def _compute_percentiles(self) -> None:
        """Pre-compute PageRank percentiles across all CVE nodes for normalization."""
        cve_nodes = [
            (n, d) for n, d in self.G.nodes(data=True) if d.get("node_type") == NODE_CVE
        ]
        if not cve_nodes:
            self._pagerank_values = np.array([0.0])
            return

        self._pagerank_values = np.array([d.get("pagerank", 0) for _, d in cve_nodes])
        logger.info(
            "Risk scorer initialized",
            cve_count=len(cve_nodes),
            pr_max=float(self._pagerank_values.max()),
            pr_mean=float(self._pagerank_values.mean()),
        )

    def _pagerank_percentile(self, pr_value: float) -> float:
        """Return 0-1 percentile of this CVE's PageRank among all CVEs."""
        if len(self._pagerank_values) == 0:
            return 0.0
        return float(np.mean(self._pagerank_values <= pr_value))

    def score(
        self,
        cve_id: str,
        cvss_score: float | None = None,
        epss_score: float | None = None,
    ) -> dict:
        """
        Compute composite VulnMind Risk Score for a CVE.
        Returns score (0-100) with component breakdown.
        """
        node = self.G.nodes.get(cve_id, {})

        # CVSS: use provided or graph-stored value
        raw_cvss = cvss_score if cvss_score is not None else node.get("cvss_score", 0.0)
        normalized_cvss = float(raw_cvss) / 10.0  # CVSS is 0-10

        # EPSS: use provided or graph-stored value (already 0-1)
        raw_epss = epss_score if epss_score is not None else node.get("epss", 0.0)
        epss_component = float(raw_epss)

        # Graph centrality percentile
        pagerank = node.get("pagerank", 0.0)
        centrality_percentile = self._pagerank_percentile(float(pagerank))

        # Weighted composite
        raw_score = (
            0.4 * normalized_cvss
            + 0.4 * epss_component
            + 0.2 * centrality_percentile
        )

        # KEV boost: confirmed exploited in the wild gets 1.15x multiplier
        kev_entry = self._kev.get(cve_id)
        kev_confirmed = kev_entry is not None
        if kev_confirmed:
            raw_score = min(raw_score * KEV_BOOST, 1.0)

        vulnmind_score = round(raw_score * 100, 1)

        return {
            "cve_id": cve_id,
            "vulnmind_score": vulnmind_score,
            "components": {
                "cvss_normalized": round(normalized_cvss, 4),
                "epss_score": round(epss_component, 4),
                "graph_centrality_percentile": round(centrality_percentile, 4),
            },
            "raw_values": {
                "cvss": raw_cvss,
                "epss": raw_epss,
                "pagerank": pagerank,
            },
            "severity_label": _score_label(vulnmind_score),
            "kev_confirmed": kev_confirmed,
            "kev_date_added": kev_entry["date_added"] if kev_entry else None,
            "kev_required_action": kev_entry["required_action"] if kev_entry else None,
        }

    def batch_score(self, cve_ids: list[str]) -> list[dict]:
        return [self.score(cve_id) for cve_id in cve_ids]


def _score_label(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    if score >= 20:
        return "Low"
    return "Informational"
