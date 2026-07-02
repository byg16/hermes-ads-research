"""Polymarket Gamma API wrapper — read-only market search.

Public docs: https://docs.polymarket.com/
No API key required for reading markets via the Gamma API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from core.logging_config import get_logger
from core.schemas import MarketCandidate, Venue, Asset

logger = get_logger(__name__)

# Keywords used to find short-horizon up/down crypto markets on Polymarket.
_ASSET_KEYWORDS = {
    Asset.BTC: ["bitcoin", "btc"],
    Asset.ETH: ["ethereum", "eth"],
}


class PolymarketClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.polymarket_gamma_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def _search_markets(self, query: str, limit: int = 20) -> list[dict]:
        params = {"search": query, "active": "true", "closed": "false", "limit": limit}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            return resp.json()

    def find_5min_crypto_markets(self, asset: Asset) -> List[MarketCandidate]:
        """Search Polymarket for short-horizon (~5 minute) up/down markets
        for the given asset. Falls back to an empty list on failure (logged,
        not raised) so the scanner can still aggregate Kalshi results."""
        keywords = _ASSET_KEYWORDS[asset]
        candidates: List[MarketCandidate] = []
        for kw in keywords:
            try:
                raw_markets = self._search_markets(kw)
            except httpx.HTTPError as exc:
                logger.error("polymarket_search_failed", asset=asset, keyword=kw, error=str(exc))
                continue

            for m in raw_markets:
                title = (m.get("question") or m.get("title") or "").lower()
                if "5 min" not in title and "5-minute" not in title and "5min" not in title:
                    continue
                try:
                    yes_price = float(m.get("outcomePrices", ["0.5", "0.5"])[0])
                except (ValueError, TypeError, IndexError):
                    yes_price = 0.5
                candidates.append(
                    MarketCandidate(
                        venue=Venue.POLYMARKET,
                        asset=asset,
                        market_id=str(m.get("id", m.get("conditionId", "unknown"))),
                        title=m.get("question", title),
                        yes_price=yes_price,
                        no_price=round(1 - yes_price, 4),
                        close_time=_parse_dt(m.get("endDate")),
                        horizon_minutes=5,
                        url=f"https://polymarket.com/event/{m.get('slug', '')}",
                    )
                )
        logger.info("polymarket_scan_complete", asset=asset, found=len(candidates))
        return candidates


def _parse_dt(value) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
