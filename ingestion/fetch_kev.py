"""
CISA Known Exploited Vulnerabilities catalog ingestion.
Source: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
DEFAULT_OUTPUT = Path("./data/kev_catalog.json")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
async def _download_kev(url: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def fetch_kev(output_path: Path = DEFAULT_OUTPUT) -> dict[str, dict]:
    """
    Fetch CISA KEV catalog and return dict mapping cveID -> entry.
    Each entry: cveID, vendorProject, product, vulnerabilityName,
                dateAdded, shortDescription, requiredAction, dueDate.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching CISA KEV catalog", url=KEV_URL)
    raw = await _download_kev(KEV_URL)

    catalog: dict[str, dict] = {}
    for entry in raw.get("vulnerabilities", []):
        cve_id = entry.get("cveID", "")
        if cve_id:
            catalog[cve_id] = {
                "cve_id": cve_id,
                "vendor_project": entry.get("vendorProject", ""),
                "product": entry.get("product", ""),
                "vulnerability_name": entry.get("vulnerabilityName", ""),
                "date_added": entry.get("dateAdded", ""),
                "short_description": entry.get("shortDescription", ""),
                "required_action": entry.get("requiredAction", ""),
                "due_date": entry.get("dueDate", ""),
            }

    output_path.write_text(json.dumps(catalog, indent=2))
    logger.info("CISA KEV fetch complete", count=len(catalog), output=str(output_path))
    return catalog


def load_kev(path: Path = DEFAULT_OUTPUT) -> dict[str, dict]:
    """Load KEV catalog from disk. Returns empty dict if file not found."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_kev())
