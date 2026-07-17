"""Agent 1: scans Polymarket and Kalshi for BTC/ETH 5-minute up/down markets."""
from __future__ import annotations

from typing import List

from core.logging_config import get_logger
from core.schemas import MarketCandidate, Asset
from tools.polymarket_client import PolymarketClient
from tools.kalshi_client import KalshiClient

logger = get_logger(__name__)


class MarketScannerAgent:
    """Finds tradeable, currently-open 5-minute crypto direction markets
    across both supported venues, for the configured asset universe."""

    name = "market_scanner"

    def __init__(self):
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()

    def run(self, assets: List[Asset]) -> List[MarketCandidate]:
        all_candidates: List[MarketCandidate] = []
        for asset in assets:
            try:
                all_candidates.extend(self.polymarket.find_5min_crypto_markets(asset))
            except Exception as exc:  # noqa: BLE001 — one venue failing shouldn't kill the scan
                logger.error("polymarket_scan_error", asset=asset, error=str(exc))
            try:
                all_candidates.extend(self.kalshi.find_5min_crypto_markets(asset))
            except Exception as exc:  # noqa: BLE001
                logger.error("kalshi_scan_error", asset=asset, error=str(exc))

        logger.info("market_scan_complete", total_candidates=len(all_candidates))
        return all_candidates
