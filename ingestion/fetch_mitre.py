"""
MITRE ATT&CK ingestion via STIX JSON from GitHub.
Extracts techniques, tactics, and relationships.
"""

import json
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

MITRE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
async def _download_stix(url: str) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _extract_techniques(stix_bundle: dict) -> list[dict]:
    techniques = []
    for obj in stix_bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        technique_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id")
                break

        if not technique_id:
            continue

        kill_chain_phases = obj.get("kill_chain_phases", [])
        tactics = [p["phase_name"] for p in kill_chain_phases if p.get("kill_chain_name") == "mitre-attack"]

        techniques.append({
            "technique_id": technique_id,
            "name": obj.get("name", ""),
            "description": obj.get("description", "")[:2000],
            "tactics": tactics,
            "platforms": obj.get("x_mitre_platforms", []),
            "detection": obj.get("x_mitre_detection", ""),
            "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
            "stix_id": obj.get("id"),
            "source": "mitre_attack",
        })

    return techniques


def _extract_relationships(stix_bundle: dict, techniques_by_stix: dict) -> list[dict]:
    """Extract technique→technique relationships (e.g., subtechnique-of)."""
    rels = []
    for obj in stix_bundle.get("objects", []):
        if obj.get("type") != "relationship":
            continue
        rel_type = obj.get("relationship_type", "")
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        if src in techniques_by_stix and tgt in techniques_by_stix:
            rels.append({
                "source": techniques_by_stix[src],
                "target": techniques_by_stix[tgt],
                "relationship_type": rel_type,
            })
    return rels


def _extract_mitigations(stix_bundle: dict, techniques_by_stix: dict) -> list[dict]:
    mitigations_by_stix = {}
    for obj in stix_bundle.get("objects", []):
        if obj.get("type") != "course-of-action":
            continue
        mid = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                mid = ref.get("external_id")
                break
        if mid:
            mitigations_by_stix[obj["id"]] = {
                "mitigation_id": mid,
                "name": obj.get("name", ""),
                "description": obj.get("description", "")[:1000],
            }

    technique_mitigations = []
    for obj in stix_bundle.get("objects", []):
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "mitigates":
            continue
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        if src in mitigations_by_stix and tgt in techniques_by_stix:
            technique_mitigations.append({
                "technique_id": techniques_by_stix[tgt],
                "mitigation": mitigations_by_stix[src],
            })

    return technique_mitigations


async def fetch_mitre(
    output_path: Path = Path("./data/mitre_techniques.jsonl"),
    relationships_path: Path = Path("./data/mitre_relationships.jsonl"),
) -> tuple[list[dict], list[dict]]:
    """
    Fetch and parse MITRE ATT&CK enterprise data.
    Returns (techniques, relationships).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MITRE ATT&CK STIX bundle")
    stix_bundle = await _download_stix(MITRE_URL)

    techniques = _extract_techniques(stix_bundle)
    techniques_by_stix = {t["stix_id"]: t["technique_id"] for t in techniques if t.get("stix_id")}

    relationships = _extract_relationships(stix_bundle, techniques_by_stix)
    mitigations = _extract_mitigations(stix_bundle, techniques_by_stix)

    with output_path.open("w", encoding="utf-8") as f:
        for tech in techniques:
            f.write(json.dumps(tech) + "\n")

    with relationships_path.open("w", encoding="utf-8") as f:
        for rel in relationships + mitigations:
            f.write(json.dumps(rel) + "\n")

    logger.info(
        "MITRE fetch complete",
        techniques=len(techniques),
        relationships=len(relationships),
        mitigations=len(mitigations),
    )
    return techniques, relationships


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_mitre())
