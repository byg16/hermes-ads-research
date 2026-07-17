"""Kalshi REST API wrapper — read-only market search.

Public docs: https://trading-api.readme.io/reference/getting-started
Public markets can be listed without authentication; authenticated calls
(placing orders) would need KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH,
which are not required for this research-only pipeline.
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

_ASSET_SERIES_TICKERS = {
    Asset.BTC: ["KXBTC", "BTC"],
    Asset.ETH: ["KXETH", "ETH"],
}


class KalshiClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.kalshi_base_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def _list_markets(self, series_ticker: str, limit: int = 50) -> list[dict]:
        params = {"series_ticker": series_ticker, "status": "open", "limit": limit}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            return resp.json().get("markets", [])

    def find_5min_crypto_markets(self, asset: Asset) -> List[MarketCandidate]:
        candidates: List[MarketCandidate] = []
        for series in _ASSET_SERIES_TICKERS[asset]:
            try:
                raw_markets = self._list_markets(series)
            except httpx.HTTPError as exc:
                logger.error("kalshi_list_failed", asset=asset, series=series, error=str(exc))
                continue

            for m in raw_markets:
                title = (m.get("title") or "").lower()
                if not any(t in title for t in ("5 min", "5-minute", "5min")):
                    continue
                yes_price = (m.get("yes_bid", 50) or 50) / 100.0
                candidates.append(
                    MarketCandidate(
                        venue=Venue.KALSHI,
                        asset=asset,
                        market_id=m.get("ticker", "unknown"),
                        title=m.get("title", title),
                        yes_price=round(yes_price, 4),
                        no_price=round(1 - yes_price, 4),
                        close_time=_parse_dt(m.get("close_time")),
                        horizon_minutes=5,
                        url=f"https://kalshi.com/markets/{m.get('ticker', '')}",
                    )
                )
        logger.info("kalshi_scan_complete", asset=asset, found=len(candidates))
        return candidates


def _parse_dt(value) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
