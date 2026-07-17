"""Scaling idea implementations:

1. cascade_predict: predict 1-min bars n+1..n+5 with Kronos, then derive
   a 5-minute-bar up/down call for n+1 by compounding those five 1-min
   predictions. Cheaper/faster than running Kronos directly on a 5-min
   series, and gives a second, cross-checkable signal.

2. scan_arbitrage: compares the model's *implied* 15-minute up-probability
   (compounding three consecutive independent 5-min predictions) against
   the *market's quoted* 15-minute contract price, to flag a potential
   internal arbitrage between the 15-min market and the chain of three
   5-min markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.logging_config import get_logger
from core.schemas import MarketCandidate, OHLCVSeries, Prediction
from tools.kronos_client import KronosClient

logger = get_logger(__name__)


def cascade_predict(series: OHLCVSeries, kronos: KronosClient, n_steps: int = 5) -> Prediction:
    """Run Kronos at 1-min granularity for `n_steps` ahead, then compound
    the per-step up-probabilities into a single 5-minute-bar up-probability
    via P(5-min up) = P(net log-return over 5 steps > 0), approximated here
    by treating each step's P(up) as an independent shift and combining
    means (a simple, fast approximation - not a substitute for directly
    modeling the 5-min bar, hence "cross-checkable signal" rather than
    ground truth)."""
    step_predictions: List[Prediction] = []
    working_series = series
    for step in range(n_steps):
        pred = kronos.predict(working_series, horizon_minutes=1)
        step_predictions.append(pred)
        # NOTE: in a full implementation we'd append the model's sampled
        # next bar to `working_series` here to get true autoregressive
        # rollout. For the stub/demo model we just reuse the same series,
        # which is acceptable for illustrating the cascade architecture.

    avg_p_up = sum(p.p_up for p in step_predictions) / len(step_predictions)
    # Compounding heuristic: a slight push away from 0.5 because 5
    # consecutive same-direction micro-moves reinforce a directional read.
    pull = (avg_p_up - 0.5) * 1.15
    p_up_5min = max(0.0, min(1.0, 0.5 + pull))

    return Prediction(
        asset=series.asset,
        horizon_minutes=5,
        p_up=round(p_up_5min, 4),
        p_down=round(1 - p_up_5min, 4),
        model_name="kronos-cascade-1m-to-5m",
        generated_at=step_predictions[-1].generated_at,
        rationale=f"cascaded from {n_steps} x 1-min predictions, avg_p_up={avg_p_up:.4f}",
    )


@dataclass
class ArbitrageSignal:
    asset: str
    market_15min_price_yes: float
    implied_15min_p_up: float
    spread: float
    actionable: bool


def scan_arbitrage(
    five_min_predictions: List[Prediction],
    fifteen_min_market: MarketCandidate,
    threshold: float = 0.05,
) -> ArbitrageSignal:
    """`five_min_predictions` should be the three consecutive 5-min-horizon
    predictions that together span the 15-min market's window. We compound
    their up-probabilities (assuming rough independence, a simplification)
    to get the model's implied 15-min up-probability, then compare to the
    market's quoted price."""
    if len(five_min_predictions) != 3:
        raise ValueError("scan_arbitrage expects exactly 3 five-minute predictions")

    implied_p_up = 1.0
    for p in five_min_predictions:
        implied_p_up *= p.p_up
    # Renormalize: pure multiplication underestimates correlated moves, so
    # blend with the simple average as a damping factor.
    avg_p_up = sum(p.p_up for p in five_min_predictions) / 3
    implied_p_up = 0.5 * implied_p_up + 0.5 * avg_p_up

    spread = implied_p_up - fifteen_min_market.yes_price
    actionable = abs(spread) >= threshold

    signal = ArbitrageSignal(
        asset=fifteen_min_market.asset.value,
        market_15min_price_yes=fifteen_min_market.yes_price,
        implied_15min_p_up=round(implied_p_up, 4),
        spread=round(spread, 4),
        actionable=actionable,
    )
    if actionable:
        logger.info("arbitrage_signal_found", **signal.__dict__)
    return signal
