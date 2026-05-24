"""
Structured CVE chunking with metadata preservation.
Creates semantically coherent chunks optimized for embedding and retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CVEChunk:
    chunk_id: str
    cve_id: str
    chunk_type: str  # "summary" | "technical" | "impact" | "remediation"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "cve_id": self.cve_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": self.metadata,
        }


def _severity_label(score: float | None) -> str:
    if score is None:
        return "Unknown"
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    return "Low"


def _build_summary_chunk(cve: dict, epss: dict | None = None) -> CVEChunk:
    """Primary chunk: CVE ID + description + CVSS summary. Best for semantic search."""
    cvss = cve.get("cvss_v3") or {}
    score = cvss.get("baseScore")
    severity = cvss.get("baseSeverity") or _severity_label(score)
    vendors = cve.get("vendors", [])
    cwes = cve.get("cwe_ids", [])

    epss_info = ""
    if epss:
        epss_val = epss.get("epss", 0)
        pct = epss.get("percentile", 0)
        epss_info = f" EPSS exploit probability: {epss_val:.4f} ({pct*100:.1f}th percentile)."

    vendor_str = ", ".join(vendors[:5]) if vendors else "unspecified"
    cwe_str = ", ".join(cwes) if cwes else "unspecified"

    text = (
        f"{cve['cve_id']} — {severity} severity"
        + (f" (CVSS {score})" if score else "")
        + f".\n"
        f"Affected vendors: {vendor_str}.\n"
        f"Weakness classifications: {cwe_str}.\n"
        f"{epss_info}\n"
        f"Description: {cve.get('description', '')}"
    ).strip()

    return CVEChunk(
        chunk_id=f"{cve['cve_id']}_summary",
        cve_id=cve["cve_id"],
        chunk_type="summary",
        text=text,
        metadata={
            "cve_id": cve["cve_id"],
            "cvss_score": score,
            "severity": severity,
            "vendors": vendors,
            "cwe_ids": cwes,
            "published": cve.get("published", ""),
            "epss": epss.get("epss") if epss else None,
            "epss_percentile": epss.get("percentile") if epss else None,
            "source": "nvd",
        },
    )


def _build_technical_chunk(cve: dict) -> CVEChunk | None:
    """Technical details: attack vector, complexity, privileges."""
    cvss = cve.get("cvss_v3")
    if not cvss:
        return None

    lines = [
        f"Technical details for {cve['cve_id']}:",
        f"Attack Vector: {cvss.get('attackVector', 'N/A')}",
        f"Attack Complexity: {cvss.get('attackComplexity', 'N/A')}",
        f"Privileges Required: {cvss.get('privilegesRequired', 'N/A')}",
        f"User Interaction: {cvss.get('userInteraction', 'N/A')}",
        f"Scope: {cvss.get('scope', 'N/A')}",
        f"Confidentiality Impact: {cvss.get('confidentialityImpact', 'N/A')}",
        f"Integrity Impact: {cvss.get('integrityImpact', 'N/A')}",
        f"Availability Impact: {cvss.get('availabilityImpact', 'N/A')}",
        f"CVSS Vector: {cvss.get('vectorString', 'N/A')}",
    ]

    return CVEChunk(
        chunk_id=f"{cve['cve_id']}_technical",
        cve_id=cve["cve_id"],
        chunk_type="technical",
        text="\n".join(lines),
        metadata={
            "cve_id": cve["cve_id"],
            "cvss_score": cvss.get("baseScore"),
            "attack_vector": cvss.get("attackVector"),
            "source": "nvd",
        },
    )


def _build_cwe_chunk(cve: dict, mitre_techniques: list[dict] | None = None) -> CVEChunk | None:
    """CWE + ATT&CK mapping chunk for graph-assisted retrieval."""
    cwes = cve.get("cwe_ids", [])
    if not cwes:
        return None

    technique_info = ""
    if mitre_techniques:
        related = [t for t in mitre_techniques[:3]]
        if related:
            names = [f"{t['technique_id']} ({t['name']})" for t in related]
            technique_info = f"\nRelated ATT&CK techniques: {', '.join(names)}."

    text = (
        f"{cve['cve_id']} weakness classification.\n"
        f"CWE categories: {', '.join(cwes)}.\n"
        + technique_info
    ).strip()

    return CVEChunk(
        chunk_id=f"{cve['cve_id']}_cwe",
        cve_id=cve["cve_id"],
        chunk_type="technical",
        text=text,
        metadata={
            "cve_id": cve["cve_id"],
            "cwe_ids": cwes,
            "source": "nvd",
        },
    )


def chunk_cve(
    cve: dict,
    epss: dict | None = None,
    mitre_techniques: list[dict] | None = None,
) -> list[CVEChunk]:
    """
    Produce 1-3 chunks per CVE depending on available data.
    Always produces a summary chunk; technical and CWE chunks when data exists.
    """
    chunks: list[CVEChunk] = []

    summary = _build_summary_chunk(cve, epss)
    chunks.append(summary)

    tech = _build_technical_chunk(cve)
    if tech:
        chunks.append(tech)

    cwe_chunk = _build_cwe_chunk(cve, mitre_techniques)
    if cwe_chunk:
        chunks.append(cwe_chunk)

    return chunks


def chunk_mitre_technique(technique: dict) -> CVEChunk:
    """Chunk a MITRE ATT&CK technique for embedding."""
    tactics_str = ", ".join(technique.get("tactics", []))
    platforms_str = ", ".join(technique.get("platforms", []))

    text = (
        f"ATT&CK Technique {technique['technique_id']}: {technique['name']}\n"
        f"Tactics: {tactics_str}\n"
        f"Platforms: {platforms_str}\n"
        f"Description: {technique.get('description', '')[:1500]}\n"
        f"Detection: {technique.get('detection', '')[:500]}"
    ).strip()

    return CVEChunk(
        chunk_id=f"technique_{technique['technique_id']}",
        cve_id=technique["technique_id"],
        chunk_type="technique",
        text=text,
        metadata={
            "technique_id": technique["technique_id"],
            "tactics": technique.get("tactics", []),
            "platforms": technique.get("platforms", []),
            "source": "mitre_attack",
        },
    )
