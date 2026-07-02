"""Apify wrapper for fetching the last N OHLCV bars for a crypto asset.

Uses the `apify-client` SDK to run a (configurable) actor that scrapes a
public exchange/aggregator (e.g. Binance, CoinGecko) for recent candles.
Apify's free tier provides enough monthly compute units to run this on a
schedule for a small number of assets.

If APIFY_API_TOKEN is unset, falls back to a synthetic random-walk series
so the rest of the pipeline (predictor, risk manager) can still be
exercised end-to-end in local/dev/demo mode.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List

from apify_client import ApifyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from core.logging_config import get_logger
from core.schemas import OHLCVBar, OHLCVSeries, Asset

logger = get_logger(__name__)

_SEED_PRICE = {Asset.BTC: 65000.0, Asset.ETH: 3400.0}


class ApifyOHLCVFetcher:
    def __init__(self, api_token: str | None = None, actor_id: str | None = None):
        self.api_token = api_token or settings.apify_api_token
        self.actor_id = actor_id or settings.apify_ohlcv_actor_id
        self._client = ApifyClient(self.api_token) if self.api_token else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_bars(self, asset: Asset, n_bars: int | None = None, timeframe: str = "1m") -> OHLCVSeries:
        n_bars = n_bars or settings.apify_bars_lookback

        if not self._client:
            logger.warning("apify_no_token", msg="Using synthetic OHLCV series for local/demo mode")
            return _synthetic_series(asset, n_bars, timeframe)

        try:
            run_input = {"symbol": f"{asset.value}USDT", "interval": timeframe, "limit": n_bars}
            run = self._client.actor(self.actor_id).call(run_input=run_input)
            dataset_items = self._client.dataset(run["defaultDatasetId"]).list_items().items

            bars = [
                OHLCVBar(
                    timestamp=datetime.fromtimestamp(item["timestamp"] / 1000, tz=timezone.utc),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0)),
                )
                for item in dataset_items
            ]
            logger.info("apify_fetch_complete", asset=asset, bars=len(bars))
            return OHLCVSeries(asset=asset, timeframe=timeframe, bars=bars, source="apify")
        except Exception as exc:  # noqa: BLE001 — log and fall back rather than crash the pipeline
            logger.error("apify_fetch_failed", asset=asset, error=str(exc))
            return _synthetic_series(asset, n_bars, timeframe)


def _synthetic_series(asset: Asset, n_bars: int, timeframe: str) -> OHLCVSeries:
    """Deterministic-ish random walk for offline development/demo only."""
    rng = random.Random(hash(asset.value) % (2**31))
    price = _SEED_PRICE.get(asset, 1000.0)
    now = datetime.now(timezone.utc)
    bars: List[OHLCVBar] = []
    for i in range(n_bars):
        ts = now - timedelta(minutes=(n_bars - i))
        change = rng.gauss(0, 0.0015) * price
        o = price
        c = max(price + change, 0.01)
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.0005)))
        low = min(o, c) * (1 - abs(rng.gauss(0, 0.0005)))
        vol = abs(rng.gauss(50, 20))
        bars.append(OHLCVBar(timestamp=ts, open=o, high=h, low=low, close=c, volume=vol))
        price = c
    return OHLCVSeries(asset=asset, timeframe=timeframe, bars=bars, source="synthetic-fallback")


apify_fetcher = ApifyOHLCVFetcher()
