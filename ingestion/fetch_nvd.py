"""
NVD CVE feed ingestion with pagination, rate limiting, and delta updates.
Fetches from https://services.nvd.nist.gov/rest/json/cves/2.0
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 2000
RATE_LIMIT_DELAY = 6.0  # NVD enforces 5 req/30s without API key; 50 req/30s with key
STATE_FILE = Path("./data/nvd_sync_state.json")


def _load_sync_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_modified": None, "total_fetched": 0}


def _save_sync_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _extract_cvss_v3(cve_item: dict) -> dict | None:
    metrics = cve_item.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            data = entries[0].get("cvssData", {})
            return {
                "baseScore": data.get("baseScore"),
                "baseSeverity": data.get("baseSeverity"),
                "vectorString": data.get("vectorString"),
                "attackVector": data.get("attackVector"),
                "attackComplexity": data.get("attackComplexity"),
                "privilegesRequired": data.get("privilegesRequired"),
                "userInteraction": data.get("userInteraction"),
                "scope": data.get("scope"),
                "confidentialityImpact": data.get("confidentialityImpact"),
                "integrityImpact": data.get("integrityImpact"),
                "availabilityImpact": data.get("availabilityImpact"),
            }
    return None


def _extract_cwes(cve_item: dict) -> list[str]:
    cwes = []
    for weakness in cve_item.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-"):
                cwes.append(val)
    return list(set(cwes))


def _extract_cpe_vendors(cve_item: dict) -> list[str]:
    vendors = set()
    for config in cve_item.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                cpe = match.get("criteria", "")
                parts = cpe.split(":")
                if len(parts) > 3:
                    vendors.add(parts[3])
    return list(vendors)


def _parse_cve(cve_item: dict) -> dict:
    cve_id = cve_item.get("id", "")
    descriptions = cve_item.get("descriptions", [])
    en_desc = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"), ""
    )
    published = cve_item.get("published", "")
    last_modified = cve_item.get("lastModified", "")

    return {
        "cve_id": cve_id,
        "description": en_desc,
        "published": published,
        "last_modified": last_modified,
        "cvss_v3": _extract_cvss_v3(cve_item),
        "cwe_ids": _extract_cwes(cve_item),
        "vendors": _extract_cpe_vendors(cve_item),
        "source": "nvd",
    }


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
async def _fetch_page(
    client: httpx.AsyncClient,
    params: dict,
    api_key: str | None,
) -> dict:
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    resp = await client.get(NVD_BASE_URL, params=params, headers=headers, timeout=60.0)

    if resp.status_code == 403:
        logger.warning("NVD rate limited, backing off 30s")
        await asyncio.sleep(30)
        raise Exception("Rate limited")

    resp.raise_for_status()
    return resp.json()


async def fetch_nvd(
    output_path: Path = Path("./data/nvd_cves.jsonl"),
    delta_days: int | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """
    Fetch all CVEs from NVD with full pagination.
    If delta_days is set, only fetch CVEs modified in the last N days.
    Returns list of parsed CVE dicts.
    """
    api_key = api_key or os.getenv("NVD_API_KEY")
    state = _load_sync_state()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {"resultsPerPage": RESULTS_PER_PAGE, "startIndex": 0}

    if delta_days is not None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=delta_days)
        params["lastModStartDate"] = start.strftime("%Y-%m-%dT%H:%M:%S.000")
        params["lastModEndDate"] = end.strftime("%Y-%m-%dT%H:%M:%S.000")
        logger.info("Delta fetch", start=params["lastModStartDate"], end=params["lastModEndDate"])
    elif state["last_modified"]:
        params["lastModStartDate"] = state["last_modified"]
        params["lastModEndDate"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")

    all_cves: list[dict] = []
    delay = RATE_LIMIT_DELAY / 2 if api_key else RATE_LIMIT_DELAY

    async with httpx.AsyncClient() as client:
        # Initial request to get total count
        logger.info("Fetching NVD page 0")
        data = await _fetch_page(client, params, api_key)
        total_results = data.get("totalResults", 0)
        cves = [_parse_cve(item["cve"]) for item in data.get("vulnerabilities", [])]
        all_cves.extend(cves)

        logger.info("NVD total results", total=total_results, fetched=len(all_cves))

        start_index = RESULTS_PER_PAGE
        while start_index < total_results:
            await asyncio.sleep(delay)
            page_params = {**params, "startIndex": start_index}
            logger.info("Fetching NVD page", start_index=start_index, total=total_results)
            data = await _fetch_page(client, page_params, api_key)
            page_cves = [_parse_cve(item["cve"]) for item in data.get("vulnerabilities", [])]
            all_cves.extend(page_cves)
            start_index += RESULTS_PER_PAGE

    # Persist to JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for cve in all_cves:
            f.write(json.dumps(cve) + "\n")

    state["last_modified"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")
    state["total_fetched"] = state.get("total_fetched", 0) + len(all_cves)
    _save_sync_state(state)

    logger.info("NVD fetch complete", count=len(all_cves), output=str(output_path))
    return all_cves


if __name__ == "__main__":
    asyncio.run(fetch_nvd())
