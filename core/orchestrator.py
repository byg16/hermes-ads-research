"""Wires the 5 agents into a Hermes Agent graph + feedback loop.

Tries to import the real `hermes_agent` package first; falls back to the
local-compatible shim (`core/_hermes_shim.py`) if it's unavailable, so the
project always runs.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import List

try:
    from hermes_agent import Agent, Loop  # type: ignore  # real framework, if installed
    _USING_REAL_HERMES = True
except ImportError:
    from core._hermes_shim import Agent, Loop  # type: ignore
    _USING_REAL_HERMES = False

from agents.market_scanner import MarketScannerAgent
from agents.data_fetcher import DataFetcherAgent
from agents.predictor import PredictorAgent
from agents.risk_manager import RiskManagerAgent
from agents.feedback_loop import FeedbackLoopAgent
from config import settings
from core.logging_config import get_logger
from core.schemas import Asset, Decision, ResolvedOutcome

logger = get_logger(__name__)

# Shared run history, exposed to scale/dashboard.py for visibility.
LATEST_DECISIONS: List[Decision] = []
LATEST_AD_BRIEFS: list = []


class Orchestrator:
    """Single-cycle pipeline: scan -> fetch -> predict -> size risk ->
    (simulate resolution) -> feedback. Designed to be called repeatedly
    by `Loop` on a fixed interval."""

    def __init__(self):
        self.scanner = MarketScannerAgent()
        self.fetcher = DataFetcherAgent()
        self.predictor = PredictorAgent()
        self.risk_manager = RiskManagerAgent()
        self.feedback = FeedbackLoopAgent()
        logger.info("orchestrator_initialized", using_real_hermes_agent=_USING_REAL_HERMES)

    def run_cycle(self) -> List[Decision]:
        assets = [Asset(a) for a in settings.asset_list if a in Asset.__members__]
        decisions: List[Decision] = []

        candidates = self.scanner.run(assets)
        if not candidates:
            logger.warning("no_market_candidates_found")

        # Group candidates by asset so we fetch bars / predict once per asset,
        # not once per market.
        candidates_by_asset: dict[Asset, list] = {}
        for c in candidates:
            candidates_by_asset.setdefault(c.asset, []).append(c)

        for asset, markets in candidates_by_asset.items():
            series = self.fetcher.run(asset, n_bars=settings.apify_bars_lookback)
            feedback_context = self.feedback.context_for_predictor(asset)
            prediction = self.predictor.run(series, horizon_minutes=5, context=feedback_context)

            for market in markets:
                decision = self.risk_manager.run(market, prediction)
                decisions.append(decision)

                if decision.side != "NONE":
                    ad_brief = self.feedback.build_ad_brief(asset, decision)
                    LATEST_AD_BRIEFS.append(ad_brief)
                    logger.info("ad_brief_generated", asset=asset, headline=ad_brief.headline)

                # In paper-trading mode we simulate resolution after the
                # market's horizon using the predicted direction vs a
                # synthetic coin-flip weighted by predicted probability, so
                # the feedback loop has something to learn from in a demo
                # run. Swap this for real settlement polling in production.
                if settings.paper_trading and decision.side != "NONE":
                    outcome = _simulate_resolution(decision)
                    self.feedback.record_outcome(outcome)

        LATEST_DECISIONS.clear()
        LATEST_DECISIONS.extend(decisions)
        return decisions


def _simulate_resolution(decision: Decision) -> ResolvedOutcome:
    p = decision.prediction.p_up if decision.side == "YES" else decision.prediction.p_down
    correct = random.random() < p
    predicted_dir = "UP" if decision.side == "YES" else "DOWN"
    actual_dir = predicted_dir if correct else ("DOWN" if predicted_dir == "UP" else "UP")
    pnl = decision.stake_usd * (1 / max(decision.market.yes_price, 0.01) - 1) if correct else -decision.stake_usd
    return ResolvedOutcome(
        market_id=decision.market.market_id,
        asset=decision.market.asset,
        actual_direction=actual_dir,
        predicted_direction=predicted_dir,
        correct=correct,
        pnl_usd=round(pnl, 2),
        resolved_at=datetime.now(timezone.utc),
    )


def build_loop(orchestrator: Orchestrator | None = None) -> Loop:
    orch = orchestrator or Orchestrator()
    return Loop(
        cycle_fn=orch.run_cycle,
        on_cycle_complete=lambda decisions: logger.info("cycle_complete", n_decisions=len(decisions)),
        interval_seconds=settings.poll_interval_seconds,
    )
