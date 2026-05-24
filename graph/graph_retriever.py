"""
Graph traversal to find related CVEs by attack chain.
Exposes neighbors, attack paths, and cluster analysis.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import networkx as nx
import structlog

from graph.build_graph import GRAPH_PATH, NODE_CVE, NODE_GROUP, NODE_TECHNIQUE, NODE_VENDOR, load_graph

logger = structlog.get_logger(__name__)


class GraphRetriever:
    def __init__(self, graph_path: Path = GRAPH_PATH):
        self.G = load_graph(graph_path)
        logger.info(
            "Graph loaded",
            nodes=self.G.number_of_nodes(),
            edges=self.G.number_of_edges(),
        )

    def get_neighbors(self, node_id: str, depth: int = 1) -> dict:
        """Return all neighbors within `depth` hops with edge relation types."""
        if node_id not in self.G:
            return {"error": f"Node {node_id} not found in graph"}

        neighbors = {"node": node_id, "depth": depth, "related": []}
        visited = {node_id}
        frontier = [(node_id, 0)]

        while frontier:
            current, d = frontier.pop(0)
            if d >= depth:
                continue

            for succ in self.G.successors(current):
                edge_data = self.G.edges[current, succ]
                node_data = dict(self.G.nodes[succ])
                if succ not in visited:
                    visited.add(succ)
                    neighbors["related"].append({
                        "id": succ,
                        "relation": edge_data.get("relation", "related"),
                        "node_type": node_data.get("node_type", "unknown"),
                        "name": node_data.get("name", succ),
                        "hop": d + 1,
                    })
                    frontier.append((succ, d + 1))

            for pred in self.G.predecessors(current):
                edge_data = self.G.edges[pred, current]
                node_data = dict(self.G.nodes[pred])
                if pred not in visited:
                    visited.add(pred)
                    neighbors["related"].append({
                        "id": pred,
                        "relation": f"inverse:{edge_data.get('relation', 'related')}",
                        "node_type": node_data.get("node_type", "unknown"),
                        "name": node_data.get("name", pred),
                        "hop": d + 1,
                    })
                    frontier.append((pred, d + 1))

        return neighbors

    def get_cves_by_technique(self, technique_id: str) -> list[dict]:
        """Find all CVEs that exploit a given ATT&CK technique."""
        if technique_id not in self.G:
            return []

        cves = []
        for pred in self.G.predecessors(technique_id):
            node = self.G.nodes[pred]
            if node.get("node_type") == NODE_CVE:
                cves.append({
                    "cve_id": pred,
                    "cvss_score": node.get("cvss_score", 0),
                    "epss": node.get("epss", 0),
                    "pagerank": node.get("pagerank", 0),
                    "severity": node.get("severity", "Unknown"),
                })

        return sorted(cves, key=lambda x: x.get("pagerank", 0), reverse=True)

    def get_cves_by_vendor(self, vendor: str) -> list[dict]:
        """Find all CVEs affecting a given vendor."""
        vendor_node = f"vendor:{vendor}"
        if vendor_node not in self.G:
            return []

        cves = []
        for pred in self.G.predecessors(vendor_node):
            node = self.G.nodes[pred]
            if node.get("node_type") == NODE_CVE:
                cves.append({
                    "cve_id": pred,
                    "cvss_score": node.get("cvss_score", 0),
                    "epss": node.get("epss", 0),
                    "pagerank": node.get("pagerank", 0),
                })

        return sorted(cves, key=lambda x: x.get("cvss_score", 0), reverse=True)

    def find_attack_paths(self, source_cve: str, max_paths: int = 3) -> list[list[str]]:
        """
        Find attack chains: CVE → CWE → ATT&CK technique paths.
        Useful for showing how a CVE fits into broader attack campaigns.
        """
        paths = []
        if source_cve not in self.G:
            return paths

        for succ in self.G.successors(source_cve):
            node = self.G.nodes[succ]
            rel = self.G.edges[source_cve, succ].get("relation", "")
            if rel == "exploits" and node.get("node_type") == NODE_TECHNIQUE:
                paths.append([source_cve, succ])
            elif rel == "classified_as":
                for tech in self.G.successors(succ):
                    if self.G.nodes[tech].get("node_type") == NODE_TECHNIQUE:
                        paths.append([source_cve, succ, tech])

            if len(paths) >= max_paths:
                break

        return paths

    def get_high_centrality_cves(self, top_n: int = 20) -> list[dict]:
        """Return top CVEs by PageRank — the most pivotal in attack chains."""
        cve_nodes = [
            (n, d) for n, d in self.G.nodes(data=True) if d.get("node_type") == NODE_CVE
        ]
        sorted_cves = sorted(
            cve_nodes,
            key=lambda x: x[1].get("pagerank", 0),
            reverse=True,
        )
        return [
            {
                "cve_id": n,
                "pagerank": d.get("pagerank", 0),
                "cvss_score": d.get("cvss_score", 0),
                "epss": d.get("epss", 0),
                "severity": d.get("severity", "Unknown"),
            }
            for n, d in sorted_cves[:top_n]
        ]

    def get_cves_by_actor(self, group_id: str) -> list[dict]:
        """
        Find all CVEs associated with a threat actor group.
        Path: group → (uses) → technique ← (exploits) ← CVE
        Returns top 20 CVEs by pagerank, each annotated with the via_technique.
        """
        if group_id not in self.G:
            return []

        seen_cves: dict[str, dict] = {}

        for technique_id in self.G.successors(group_id):
            edge = self.G.edges[group_id, technique_id]
            if edge.get("relation") != "uses":
                continue
            if self.G.nodes[technique_id].get("node_type") != NODE_TECHNIQUE:
                continue

            for cve_id in self.G.predecessors(technique_id):
                node = self.G.nodes[cve_id]
                if node.get("node_type") != NODE_CVE:
                    continue
                if cve_id not in seen_cves:
                    seen_cves[cve_id] = {
                        "cve_id": cve_id,
                        "cvss_score": node.get("cvss_score", 0),
                        "epss": node.get("epss", 0),
                        "pagerank": node.get("pagerank", 0),
                        "severity": node.get("severity", "Unknown"),
                        "via_technique": technique_id,
                    }

        sorted_cves = sorted(seen_cves.values(), key=lambda x: x.get("pagerank", 0), reverse=True)
        return sorted_cves[:20]

    def get_actor_profile(self, group_id: str) -> dict:
        """
        Return full threat actor profile: node attributes, techniques used, top CVEs.
        """
        if group_id not in self.G:
            return {"error": f"Actor {group_id} not found in graph"}

        node_data = dict(self.G.nodes[group_id])

        techniques_used = []
        for tid in self.G.successors(group_id):
            edge = self.G.edges[group_id, tid]
            if edge.get("relation") == "uses":
                tech_node = self.G.nodes.get(tid, {})
                techniques_used.append({
                    "technique_id": tid,
                    "name": tech_node.get("name", ""),
                    "tactics": tech_node.get("tactics", ""),
                })

        top_cves = self.get_cves_by_actor(group_id)[:10]

        return {
            "group_id": group_id,
            "name": node_data.get("name", ""),
            "aliases": [a for a in node_data.get("aliases", "").split(",") if a],
            "description": node_data.get("description", ""),
            "techniques_used": techniques_used,
            "top_cves": top_cves,
        }

    def node_info(self, node_id: str) -> dict:
        """Return full node attributes."""
        if node_id not in self.G:
            return {}
        return {"id": node_id, **dict(self.G.nodes[node_id])}
