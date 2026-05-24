"""
Builds a directed NetworkX knowledge graph.
Nodes: CVE, CWE, ATT&CK technique, vendor, threat actor group
Edges: CVE→exploits→technique, CVE→affects→vendor, CVE→classified_as→CWE, group→uses→technique
Stores as GraphML for persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)

GRAPH_PATH = Path("./data/vulnmind_graph.graphml")

# Node type constants
NODE_CVE = "cve"
NODE_CWE = "cwe"
NODE_TECHNIQUE = "technique"
NODE_VENDOR = "vendor"
NODE_GROUP = "group"


# CWE → ATT&CK technique mapping (curated subset)
CWE_TO_TECHNIQUE: dict[str, list[str]] = {
    "CWE-79": ["T1059.007"],   # XSS → JavaScript execution
    "CWE-89": ["T1190"],        # SQLi → Exploit public-facing app
    "CWE-78": ["T1059"],        # OS command injection → Command execution
    "CWE-22": ["T1083"],        # Path traversal → File & directory discovery
    "CWE-119": ["T1203"],       # Buffer overflow → Exploitation for client execution
    "CWE-120": ["T1203"],
    "CWE-787": ["T1203"],
    "CWE-416": ["T1203"],       # Use-after-free
    "CWE-200": ["T1552"],       # Info disclosure → Unsecured credentials
    "CWE-287": ["T1078"],       # Auth bypass → Valid accounts
    "CWE-798": ["T1078.001"],   # Hard-coded creds
    "CWE-502": ["T1059"],       # Deserialization → Command execution
    "CWE-611": ["T1190"],       # XXE → Exploit public-facing
    "CWE-918": ["T1090"],       # SSRF → Proxy
    "CWE-20": ["T1190"],        # Improper input validation
    "CWE-601": ["T1534"],       # Open redirect → Internal spearphishing
    "CWE-362": ["T1055"],       # Race condition → Process injection
    "CWE-732": ["T1222"],       # Incorrect permissions → File permissions
    "CWE-284": ["T1078"],       # Improper access control
    "CWE-352": ["T1185"],       # CSRF → Browser session hijacking
}


def build_graph(
    nvd_path: Path = Path("./data/nvd_cves.jsonl"),
    mitre_path: Path = Path("./data/mitre_techniques.jsonl"),
    epss_path: Path = Path("./data/epss_scores.json"),
    actors_path: Path = Path("./data/threat_actors.jsonl"),
    output_path: Path = GRAPH_PATH,
) -> nx.DiGraph:
    """Build the full VulnMind knowledge graph and save as GraphML."""
    G = nx.DiGraph()

    # Load EPSS
    epss_scores: dict[str, dict] = {}
    if epss_path.exists():
        epss_scores = json.loads(epss_path.read_text())

    # Load MITRE techniques → add as nodes
    techniques: dict[str, dict] = {}
    if mitre_path.exists():
        with mitre_path.open() as f:
            for line in f:
                tech = json.loads(line.strip())
                tid = tech["technique_id"]
                techniques[tid] = tech
                G.add_node(
                    tid,
                    node_type=NODE_TECHNIQUE,
                    name=tech.get("name", ""),
                    tactics=",".join(tech.get("tactics", [])),
                )

    # Load CVEs → add nodes + edges
    cve_count = 0
    if nvd_path.exists():
        with nvd_path.open() as f:
            for line in f:
                cve = json.loads(line.strip())
                cve_id = cve["cve_id"]
                cvss = cve.get("cvss_v3") or {}
                score = cvss.get("baseScore")
                epss_data = epss_scores.get(cve_id, {})

                G.add_node(
                    cve_id,
                    node_type=NODE_CVE,
                    cvss_score=score or 0.0,
                    epss=epss_data.get("epss", 0.0),
                    epss_percentile=epss_data.get("percentile", 0.0),
                    published=cve.get("published", ""),
                    severity=cvss.get("baseSeverity", "Unknown"),
                )

                # CVE → CWE edges
                for cwe in cve.get("cwe_ids", []):
                    if not G.has_node(cwe):
                        G.add_node(cwe, node_type=NODE_CWE, name=cwe)
                    G.add_edge(cve_id, cwe, relation="classified_as")

                    # CWE → ATT&CK technique (via curated mapping)
                    for tid in CWE_TO_TECHNIQUE.get(cwe, []):
                        if tid in techniques:
                            G.add_edge(cve_id, tid, relation="exploits")

                # CVE → vendor edges
                for vendor in cve.get("vendors", [])[:10]:  # cap at 10 vendors
                    vendor_node = f"vendor:{vendor}"
                    if not G.has_node(vendor_node):
                        G.add_node(vendor_node, node_type=NODE_VENDOR, name=vendor)
                    G.add_edge(cve_id, vendor_node, relation="affects")

                cve_count += 1

    # Load threat actor groups → add nodes + edges to techniques
    group_count = 0
    group_technique_edge_count = 0
    if actors_path.exists():
        with actors_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                group = json.loads(line)
                gid = group["group_id"]
                G.add_node(
                    gid,
                    node_type=NODE_GROUP,
                    name=group.get("name", ""),
                    aliases=",".join(group.get("aliases", [])),
                    description=group.get("description", "")[:500],
                )
                for tid in group.get("associated_techniques", []):
                    if G.has_node(tid):
                        G.add_edge(gid, tid, relation="uses")
                        group_technique_edge_count += 1
                group_count += 1
        logger.info(
            "Threat actor nodes added",
            group_count=group_count,
            group_technique_edge_count=group_technique_edge_count,
        )

    logger.info(
        "Graph built",
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        cves=cve_count,
        groups=group_count,
    )

    # Compute PageRank centrality for CVE nodes
    pagerank = nx.pagerank(G, alpha=0.85)
    for node, pr in pagerank.items():
        G.nodes[node]["pagerank"] = pr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info("Graph saved", path=str(output_path))

    return G


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Graph not found at {path}. Run build_graph first.")
    return nx.read_graphml(str(path))


if __name__ == "__main__":
    build_graph()
