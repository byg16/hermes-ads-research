"""Agent 5: Hermes-loop feedback — scores resolved predictions, maintains a
running scoreboard, feeds context back into the next Predictor call, and
emits an `AdBrief` (marketing-ready summary) per cycle.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict

from core.logging_config import get_logger
from core.schemas import AdBrief, Asset, Decision, ResolvedOutcome
from tools.llm_client import OpenRouterClient

logger = get_logger(__name__)

_HISTORY_LEN = 50


class FeedbackLoopAgent:
    name = "feedback_loop"

    def __init__(self, llm: OpenRouterClient | None = None):
        self.llm = llm or OpenRouterClient()
        # per-asset rolling window of resolved outcomes
        self._history: Dict[Asset, Deque[ResolvedOutcome]] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))

    def record_outcome(self, outcome: ResolvedOutcome) -> None:
        self._history[outcome.asset].append(outcome)
        logger.info(
            "outcome_recorded",
            asset=outcome.asset,
            correct=outcome.correct,
            pnl_usd=outcome.pnl_usd,
            running_hit_rate=self.hit_rate(outcome.asset),
        )

    def hit_rate(self, asset: Asset) -> float:
        hist = self._history[asset]
        if not hist:
            return 0.5
        return sum(1 for o in hist if o.correct) / len(hist)

    def context_for_predictor(self, asset: Asset) -> str:
        """This string is what closes the loop: it's passed back into
        PredictorAgent on the *next* cycle so the model has visibility into
        recent calibration."""
        hist = self._history[asset]
        if not hist:
            return ""
        n = len(hist)
        hr = self.hit_rate(asset)
        recent_dirs = [o.predicted_direction for o in list(hist)[-5:]]
        return f"last {n} resolved: hit_rate={hr:.2%}, recent predicted directions={recent_dirs}"

    def build_ad_brief(self, asset: Asset, latest_decision: Decision) -> AdBrief:
        hist = self._history[asset]
        hr = self.hit_rate(asset)
        n = len(hist)

        prompt = (
            f"Write a short, punchy 2-sentence marketing blurb (no hashtags, no emoji) "
            f"for a crypto prediction-market bot. Asset: {asset.value}. "
            f"Recent hit rate over last {n} resolved 5-minute predictions: {hr:.0%}. "
            f"Latest call: {latest_decision.side} with {latest_decision.kelly_fraction:.1%} Kelly stake "
            f"and edge {latest_decision.edge:.1%}. Keep it factual, no hype words like 'guaranteed'."
        )
        try:
            body = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=120)
        except Exception as exc:  # noqa: BLE001 — ad copy is non-critical, never break the pipeline
            logger.error("ad_brief_llm_failed", error=str(exc))
            body = (
                f"Our {asset.value} 5-minute prediction model is hitting {hr:.0%} over the last {n} calls. "
                f"Latest signal: {latest_decision.side} ({latest_decision.edge:.1%} edge)."
            )

        return AdBrief(
            asset=asset,
            headline=f"{asset.value} 5-min prediction model: {hr:.0%} hit rate ({n} samples)",
            body=body.strip(),
            hit_rate_recent=round(hr, 4),
            sample_size=n,
            generated_at=datetime.now(timezone.utc),
        )
