"""
MITRE ATT&CK threat actor (intrusion-set) extraction.
Reuses the STIX bundle already fetched by fetch_mitre.py.
Extracts APT groups with associated techniques via relationship objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

MITRE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)
STIX_CACHE_PATH = Path("./data/enterprise_attack_raw.json")
DEFAULT_OUTPUT = Path("./data/threat_actors.jsonl")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
async def _download_stix(url: str) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _load_stix_bundle() -> dict:
    """Load STIX bundle from cache or download fresh."""
    if STIX_CACHE_PATH.exists():
        logger.info("Loading MITRE STIX bundle from cache", path=str(STIX_CACHE_PATH))
        return json.loads(STIX_CACHE_PATH.read_text())

    logger.info("Downloading MITRE ATT&CK STIX bundle for threat actor extraction")
    bundle = await _download_stix(MITRE_URL)
    STIX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STIX_CACHE_PATH.write_text(json.dumps(bundle))
    return bundle


def _build_technique_stix_map(objects: list[dict]) -> dict[str, str]:
    """Build map: stix_id -> technique_id for attack-pattern objects."""
    mapping: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id", "")
                if ext_id:
                    mapping[obj["id"]] = ext_id
                    break
    return mapping


def _extract_groups(
    objects: list[dict],
    technique_stix_map: dict[str, str],
) -> list[dict]:
    """Extract intrusion-set objects and their associated techniques."""
    # Build group stix_id -> group_id map
    groups: dict[str, dict] = {}
    for obj in objects:
        if obj.get("type") != "intrusion-set":
            continue
        group_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                group_id = ref.get("external_id", "")
                break
        if not group_id:
            continue

        aliases = obj.get("aliases", [])
        # Remove the group name itself from aliases if present
        name = obj.get("name", "")
        aliases = [a for a in aliases if a != name]

        groups[obj["id"]] = {
            "group_id": group_id,
            "stix_id": obj["id"],
            "name": name,
            "aliases": aliases,
            "description": obj.get("description", "")[:1000],
            "associated_techniques": [],
        }

    # Traverse relationship objects to find technique associations
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        if src in groups and tgt in technique_stix_map:
            tid = technique_stix_map[tgt]
            if tid not in groups[src]["associated_techniques"]:
                groups[src]["associated_techniques"].append(tid)

    return list(groups.values())


async def fetch_threat_actors(
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, dict]:
    """
    Extract APT groups from MITRE ATT&CK STIX bundle.
    Returns dict mapping group_id -> entry dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = await _load_stix_bundle()
    objects = bundle.get("objects", [])

    technique_stix_map = _build_technique_stix_map(objects)
    logger.info("Built technique STIX map", count=len(technique_stix_map))

    groups = _extract_groups(objects, technique_stix_map)

    result: dict[str, dict] = {}
    with output_path.open("w", encoding="utf-8") as f:
        for group in groups:
            f.write(json.dumps(group) + "\n")
            result[group["group_id"]] = group

    logger.info(
        "Threat actor extraction complete",
        groups=len(groups),
        output=str(output_path),
    )
    return result


def load_threat_actors(path: Path = DEFAULT_OUTPUT) -> dict[str, dict]:
    """Load threat actors from JSONL. Returns dict mapping group_id -> entry."""
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                result[entry["group_id"]] = entry
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_threat_actors())
