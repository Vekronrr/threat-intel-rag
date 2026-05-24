"""
EPSS exploit prediction scores from FIRST.org.
Joins to CVE IDs for risk scoring.
"""

import asyncio
import csv
import gzip
import io
import json
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

EPSS_API_URL = "https://api.first.org/data/v1/epss"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
async def _fetch_epss_page(client: httpx.AsyncClient, offset: int, limit: int = 10000) -> dict:
    resp = await client.get(
        EPSS_API_URL,
        params={"offset": offset, "limit": limit},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_epss(
    output_path: Path = Path("./data/epss_scores.json"),
) -> dict[str, dict]:
    """
    Fetch all EPSS scores. Returns dict mapping CVE ID -> {epss, percentile}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        # Fetch first page to get total count
        logger.info("Fetching EPSS scores page 0")
        data = await _fetch_epss_page(client, offset=0)
        total = data.get("total", 0)
        page_limit = 10000

        for item in data.get("data", []):
            cve_id = item.get("cve")
            if cve_id:
                scores[cve_id] = {
                    "epss": float(item.get("epss", 0)),
                    "percentile": float(item.get("percentile", 0)),
                    "date": item.get("date", ""),
                }

        # Paginate remaining
        offset = page_limit
        while offset < total:
            logger.info("Fetching EPSS page", offset=offset, total=total)
            await asyncio.sleep(1.0)
            data = await _fetch_epss_page(client, offset=offset)
            for item in data.get("data", []):
                cve_id = item.get("cve")
                if cve_id:
                    scores[cve_id] = {
                        "epss": float(item.get("epss", 0)),
                        "percentile": float(item.get("percentile", 0)),
                        "date": item.get("date", ""),
                    }
            offset += page_limit

    output_path.write_text(json.dumps(scores, indent=2))
    logger.info("EPSS fetch complete", count=len(scores))
    return scores


def load_epss(path: Path = Path("./data/epss_scores.json")) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    asyncio.run(fetch_epss())
