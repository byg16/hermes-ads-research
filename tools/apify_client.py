"""Data fetcher using Binance public API (free, no key needed)
   Falls back to synthetic data if Binance is unreachable.
   Apify token is still stored in config and shown as configured.
"""
from __future__ import annotations

import httpx
from datetime import datetime, timezone, timedelta
from typing import List
import random

from config import settings
from core.logging_config import get_logger
from core.schemas import OHLCVBar, OHLCVSeries, Asset

logger = get_logger(__name__)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
_SEED_PRICE = {Asset.BTC: 65000.0, Asset.ETH: 3400.0}


class ApifyOHLCVFetcher:
    def __init__(self, api_token=None, actor_id=None):
        self.api_token = api_token or settings.apify_api_token
        logger.info("apify_configured", token_set=bool(self.api_token))

    def fetch_bars(self, asset: Asset, n_bars: int = None,
                   timeframe: str = "1m") -> OHLCVSeries:
        n_bars = n_bars or settings.apify_bars_lookback
        try:
            return self._fetch_binance(asset, n_bars, timeframe)
        except Exception as exc:
            logger.warning("binance_fallback_to_synthetic", error=str(exc))
            return _synthetic_series(asset, n_bars, timeframe)

    def _fetch_binance(self, asset: Asset, n_bars: int,
                       timeframe: str) -> OHLCVSeries:
        symbol = f"{asset.value}USDT"
        params = {"symbol": symbol, "interval": timeframe, "limit": min(n_bars, 1000)}
        logger.info("fetching_from_binance", symbol=symbol, bars=n_bars)
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(BINANCE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        bars = [
            OHLCVBar(
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            )
            for k in data
        ]
        logger.info("binance_fetch_complete", asset=asset, bars=len(bars))
        return OHLCVSeries(asset=asset, timeframe=timeframe,
                           bars=bars, source="binance-live")


def _synthetic_series(asset: Asset, n_bars: int,
                      timeframe: str) -> OHLCVSeries:
    rng = random.Random(hash(asset.value) % (2 ** 31))
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
        bars.append(OHLCVBar(timestamp=ts, open=o, high=h,
                             low=low, close=c, volume=vol))
        price = c
    return OHLCVSeries(asset=asset, timeframe=timeframe,
                       bars=bars, source="synthetic-fallback")


apify_fetcher = ApifyOHLCVFetcher()