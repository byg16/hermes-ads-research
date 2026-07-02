"""Wrapper around the Kronos foundation model for K-line forecasting.

Kronos (https://github.com/shiyu-coder/Kronos) is an open-source
decoder-only transformer pretrained on candlestick data, used here to
forecast the probability that the next bar(s) close higher than they open.

Two modes:
  - "local": loads the actual Kronos checkpoint via the `kronos` package
    (`pip install git+https://github.com/shiyu-coder/Kronos.git`) and the
    `Kronos` / `KronosTokenizer` / `KronosPredictor` classes from its repo.
    This is gated behind KRONOS_MODE=local since it requires downloading
    model weights and (ideally) a GPU.
  - "stub": a lightweight, dependency-free statistical fallback (EWMA
    momentum + volatility-normalized z-score -> sigmoid) so the full
    pipeline is runnable and testable without the model weights. This is
    the default.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

from config import settings
from core.logging_config import get_logger
from core.schemas import OHLCVSeries, Prediction, Asset

logger = get_logger(__name__)


class KronosClient:
    def __init__(self, mode: str | None = None):
        self.mode = (mode or settings.kronos_mode).lower()
        self._predictor = None
        if self.mode == "local":
            self._try_load_local_model()

    def _try_load_local_model(self) -> None:
        try:
            from kronos import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

            tokenizer = KronosTokenizer.from_pretrained(f"{settings.kronos_model_name}-tokenizer")
            model = Kronos.from_pretrained(settings.kronos_model_name)
            self._predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=settings.kronos_lookback)
            logger.info("kronos_local_model_loaded", model=settings.kronos_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kronos_local_load_failed_falling_back_to_stub", error=str(exc))
            self.mode = "stub"

    def predict(self, series: OHLCVSeries, horizon_minutes: int = 5) -> Prediction:
        if self.mode == "local" and self._predictor is not None:
            return self._predict_local(series, horizon_minutes)
        return self._predict_stub(series, horizon_minutes)

    # ------------------------------------------------------------------ #
    def _predict_local(self, series: OHLCVSeries, horizon_minutes: int) -> Prediction:
        df = pd.DataFrame([b.model_dump() for b in series.bars])
        df = df.rename(columns={"timestamp": "timestamps"})
        lookback = min(settings.kronos_lookback, len(df))
        x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]]
        x_timestamp = df.tail(lookback)["timestamps"]
        y_timestamp = pd.Series(
            [x_timestamp.iloc[-1] + pd.Timedelta(minutes=i + 1) for i in range(horizon_minutes)]
        )
        try:
            pred_df = self._predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=horizon_minutes, T=1.0, top_p=0.9, sample_count=1,
            )
            last_close = float(x_df["close"].iloc[-1])
            forecast_close = float(pred_df["close"].iloc[-1])
            p_up = _logistic_from_pct_move((forecast_close - last_close) / last_close)
            rationale = f"Kronos local forecast: last_close={last_close:.2f} -> forecast={forecast_close:.2f}"
        except Exception as exc:  # noqa: BLE001
            logger.error("kronos_local_predict_failed", error=str(exc))
            return self._predict_stub(series, horizon_minutes)

        return Prediction(
            asset=series.asset,
            horizon_minutes=horizon_minutes,
            p_up=round(p_up, 4),
            p_down=round(1 - p_up, 4),
            model_name=settings.kronos_model_name,
            generated_at=datetime.now(timezone.utc),
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    def _predict_stub(self, series: OHLCVSeries, horizon_minutes: int) -> Prediction:
        """Momentum + mean-reversion blend, volatility-normalized, mapped
        through a logistic function -> P(up). No external dependencies;
        purely a placeholder until real Kronos weights are wired in."""
        closes = np.array([b.close for b in series.bars], dtype=float)
        if len(closes) < 20:
            p_up = 0.5
            rationale = "insufficient history, defaulting to 0.5"
        else:
            returns = np.diff(np.log(closes))
            short_mom = returns[-5:].mean()
            long_mom = returns[-30:].mean() if len(returns) >= 30 else returns.mean()
            vol = returns[-30:].std() if len(returns) >= 30 else returns.std()
            vol = max(vol, 1e-6)
            momentum_signal = (0.7 * short_mom + 0.3 * long_mom) / vol
            p_up = _logistic(momentum_signal * math.sqrt(horizon_minutes))
            rationale = (
                f"stub momentum model: short_mom={short_mom:.6f} long_mom={long_mom:.6f} "
                f"vol={vol:.6f} signal={momentum_signal:.4f}"
            )

        return Prediction(
            asset=series.asset,
            horizon_minutes=horizon_minutes,
            p_up=round(float(p_up), 4),
            p_down=round(float(1 - p_up), 4),
            model_name="kronos-stub-momentum-v1",
            generated_at=datetime.now(timezone.utc),
            rationale=rationale,
        )


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logistic_from_pct_move(pct_move: float, scale: float = 50.0) -> float:
    return _logistic(pct_move * scale)


kronos_client = KronosClient()
