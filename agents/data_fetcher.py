"""Agent 2: fetches the last N OHLCV bars for a crypto asset via Apify."""
from __future__ import annotations

from core.logging_config import get_logger
from core.schemas import Asset, OHLCVSeries
from tools.apify_client import ApifyOHLCVFetcher

logger = get_logger(__name__)


class DataFetcherAgent:
    name = "data_fetcher"

    def __init__(self, fetcher: ApifyOHLCVFetcher | None = None):
        self.fetcher = fetcher or ApifyOHLCVFetcher()

    def run(self, asset: Asset, n_bars: int = 1000, timeframe: str = "1m") -> OHLCVSeries:
        logger.info("fetching_bars", asset=asset, n_bars=n_bars, timeframe=timeframe)
        series = self.fetcher.fetch_bars(asset, n_bars=n_bars, timeframe=timeframe)
        logger.info("bars_fetched", asset=asset, bars=len(series.bars), source=series.source)
        return series
