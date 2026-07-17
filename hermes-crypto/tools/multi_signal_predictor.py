"""
Enhanced multi-signal predictor.
Combines RSI + momentum + volume confirmation for a real edge.
Replaces the stub momentum model in tools/kronos_client.py — 
original file is untouched, this is a drop-in upgrade.

Used by agents/hermes_llm_agent.py only.
Original predictor still works for demo.py and main.py.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from core.logging_config import get_logger
from core.schemas import OHLCVSeries, Prediction, Asset

logger = get_logger(__name__)


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index — measures overbought/oversold."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_volume_signal(volumes: np.ndarray, closes: np.ndarray,
                           lookback: int = 10) -> float:
    """Volume confirmation signal.
    High volume on up moves = bullish confirmation.
    High volume on down moves = bearish confirmation.
    Returns value between -1 (bearish) and +1 (bullish).
    """
    if len(volumes) < lookback + 1:
        return 0.0
    recent_vols = volumes[-lookback:]
    recent_closes = closes[-lookback:]
    avg_vol = recent_vols.mean()
    signal = 0.0
    for i in range(1, len(recent_closes)):
        direction = 1.0 if recent_closes[i] > recent_closes[i - 1] else -1.0
        vol_weight = recent_vols[i] / (avg_vol + 1e-9)
        signal += direction * min(vol_weight, 3.0)
    return max(-1.0, min(1.0, signal / lookback))


def compute_trend_filter(closes: np.ndarray,
                          short: int = 10, long: int = 30) -> float:
    """Simple trend filter using EMA crossover.
    Returns +1 if short EMA above long EMA (uptrend),
    -1 if below (downtrend), 0 if flat."""
    if len(closes) < long:
        return 0.0
    short_ema = float(np.mean(closes[-short:]))
    long_ema = float(np.mean(closes[-long:]))
    diff_pct = (short_ema - long_ema) / (long_ema + 1e-9)
    if diff_pct > 0.001:
        return 1.0
    elif diff_pct < -0.001:
        return -1.0
    return 0.0


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))


class MultiSignalPredictor:
    """
    Combines RSI + momentum + volume + trend filter into
    a single P(up) estimate. Much stronger than pure momentum.

    Signal weights (tunable):
      - Momentum  : 35%
      - RSI       : 25%
      - Volume    : 25%
      - Trend     : 15%
    """

    def predict(self, series: OHLCVSeries,
                horizon_minutes: int = 5) -> Prediction:
        closes = np.array([b.close for b in series.bars], dtype=float)
        volumes = np.array([b.volume for b in series.bars], dtype=float)

        if len(closes) < 35:
            return Prediction(
                asset=series.asset,
                horizon_minutes=horizon_minutes,
                p_up=0.5, p_down=0.5,
                model_name="multi-signal-v1",
                generated_at=datetime.now(timezone.utc),
                rationale="insufficient history",
            )

        # ── Signal 1: Momentum (normalized) ──
        returns = np.diff(np.log(closes))
        short_mom = returns[-5:].mean()
        long_mom = returns[-20:].mean()
        vol = returns[-20:].std()
        vol = max(vol, 1e-6)
        mom_signal = (0.6 * short_mom + 0.4 * long_mom) / vol

        # ── Signal 2: RSI ──
        rsi = compute_rsi(closes, period=14)
        # RSI below 30 = oversold = likely to go up
        # RSI above 70 = overbought = likely to go down
        rsi_signal = -(rsi - 50.0) / 50.0  # maps to [-1, +1]

        # ── Signal 3: Volume confirmation ──
        vol_signal = compute_volume_signal(volumes, closes, lookback=10)

        # ── Signal 4: Trend filter ──
        trend = compute_trend_filter(closes, short=10, long=30)

        # ── Combine with weights ──
        combined = (
            0.35 * mom_signal * math.sqrt(horizon_minutes) +
            0.25 * rsi_signal * 2.0 +
            0.25 * vol_signal * 2.0 +
            0.15 * trend * 2.0
        )

        p_up = _logistic(combined)

        rationale = (
            f"mom={mom_signal:.4f} rsi={rsi:.1f}({rsi_signal:.3f}) "
            f"vol_sig={vol_signal:.3f} trend={trend:.1f} "
            f"combined={combined:.4f}"
        )

        logger.info("multi_signal_prediction",
                    asset=series.asset,
                    p_up=round(p_up, 4),
                    rsi=round(rsi, 1),
                    trend=trend,
                    vol_signal=round(vol_signal, 3))

        return Prediction(
            asset=series.asset,
            horizon_minutes=horizon_minutes,
            p_up=round(float(p_up), 4),
            p_down=round(float(1 - p_up), 4),
            model_name="multi-signal-v1",
            generated_at=datetime.now(timezone.utc),
            rationale=rationale,
        )
