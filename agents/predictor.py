"""Agent 3: predicts next up/down move using the Kronos model wrapper."""
from __future__ import annotations

from core.logging_config import get_logger
from core.schemas import OHLCVSeries, Prediction
from tools.kronos_client import KronosClient

logger = get_logger(__name__)


class PredictorAgent:
    name = "predictor"

    def __init__(self, kronos: KronosClient | None = None):
        self.kronos = kronos or KronosClient()

    def run(self, series: OHLCVSeries, horizon_minutes: int = 5, context: str = "") -> Prediction:
        """`context` lets the feedback loop inject recent performance stats
        (e.g. 'last 10 predictions: 7 correct, slight up-bias') so the
        prediction can be calibrated/adjusted over time — the closed-loop
        piece of the pipeline."""
        prediction = self.kronos.predict(series, horizon_minutes=horizon_minutes)
        if context:
            prediction.rationale = f"{prediction.rationale} | feedback_context: {context}"
        logger.info(
            "prediction_made",
            asset=series.asset,
            p_up=prediction.p_up,
            model=prediction.model_name,
        )
        return prediction
