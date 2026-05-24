"""
Temporal EPSS drift detection.
Tracks EPSS score history per CVE and flags sudden spikes in exploit probability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_HISTORY_PATH = Path("./data/epss_history.json")


class EpssMonitor:
    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH):
        self.history_path = history_path
        self._history: dict[str, list[dict]] = self._load_history()

    def _load_history(self) -> dict[str, list[dict]]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return {}

    def _save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self._history))

    async def update(self) -> list[dict]:
        """
        Fetch current EPSS scores, append to history, detect and return spikes.
        """
        from ingestion.fetch_epss import fetch_epss

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Updating EPSS history", date=today)

        scores = await fetch_epss()

        for cve_id, data in scores.items():
            entry = {
                "date": today,
                "epss": data["epss"],
                "percentile": data["percentile"],
            }
            if cve_id not in self._history:
                self._history[cve_id] = []

            # Avoid duplicate entries for the same date
            existing_dates = {h["date"] for h in self._history[cve_id]}
            if today not in existing_dates:
                self._history[cve_id].append(entry)

        self._save_history()
        logger.info("EPSS history updated", cve_count=len(self._history))

        return self.detect_spikes()

    def detect_spikes(
        self,
        threshold: float = 0.2,
        window_days: int = 7,
    ) -> list[dict]:
        """
        Flag CVEs where EPSS increased by >= threshold within window_days.
        Returns list sorted by delta descending.
        """
        spikes = []
        today = datetime.now(timezone.utc)

        for cve_id, history in self._history.items():
            if len(history) < 2:
                continue

            # Sort by date ascending
            sorted_history = sorted(history, key=lambda h: h["date"])
            current = sorted_history[-1]
            current_epss = current["epss"]
            current_date = current["date"]

            # Find the entry closest to window_days ago
            reference = None
            for entry in reversed(sorted_history[:-1]):
                try:
                    entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    delta_days = (today - entry_date).days
                    if delta_days >= window_days:
                        reference = entry
                        break
                    reference = entry  # keep moving back until we exceed window
                except ValueError:
                    continue

            if reference is None:
                reference = sorted_history[0]

            prev_epss = reference["epss"]
            delta = current_epss - prev_epss

            if delta >= threshold:
                spikes.append({
                    "cve_id": cve_id,
                    "current_epss": current_epss,
                    "previous_epss": prev_epss,
                    "delta": round(delta, 4),
                    "spike_date": current_date,
                    "reference_date": reference["date"],
                })

        spikes.sort(key=lambda x: x["delta"], reverse=True)
        return spikes

    def get_trending(self, top_n: int = 10) -> list[dict]:
        """
        Return top_n CVEs sorted by highest recent EPSS delta (any window).
        """
        deltas = []
        for cve_id, history in self._history.items():
            if len(history) < 2:
                continue
            sorted_history = sorted(history, key=lambda h: h["date"])
            current_epss = sorted_history[-1]["epss"]
            prev_epss = sorted_history[-2]["epss"]
            delta = current_epss - prev_epss
            deltas.append({
                "cve_id": cve_id,
                "current_epss": current_epss,
                "previous_epss": prev_epss,
                "delta": round(delta, 4),
                "spike_date": sorted_history[-1]["date"],
            })

        deltas.sort(key=lambda x: x["delta"], reverse=True)
        return deltas[:top_n]
