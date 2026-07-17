"""Pydantic data contracts passed between agents in the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Venue(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"


class MarketCandidate(BaseModel):
    """A single 5-minute up/down style prediction market found by Agent 1."""

    venue: Venue
    asset: Asset
    market_id: str
    title: str
    yes_price: float = Field(ge=0, le=1, description="Current implied probability of 'up'/'yes'")
    no_price: float = Field(ge=0, le=1)
    close_time: datetime
    horizon_minutes: int = 5
    url: Optional[str] = None


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVSeries(BaseModel):
    asset: Asset
    timeframe: str = "1m"
    bars: List[OHLCVBar]
    source: str = "apify"


class Prediction(BaseModel):
    asset: Asset
    horizon_minutes: int
    p_up: float = Field(ge=0, le=1)
    p_down: float = Field(ge=0, le=1)
    model_name: str
    generated_at: datetime
    rationale: str = ""


class Decision(BaseModel):
    market: MarketCandidate
    prediction: Prediction
    kelly_fraction: float
    stake_usd: float
    side: str  # "YES" or "NO"
    edge: float
    notes: str = ""


class ResolvedOutcome(BaseModel):
    market_id: str
    asset: Asset
    actual_direction: str  # "UP" or "DOWN"
    predicted_direction: str
    correct: bool
    pnl_usd: float
    resolved_at: datetime


class AdBrief(BaseModel):
    """Marketing/ads-ready summary of model performance, for downstream
    ad-copy generation ("creating Ads" requirement)."""

    asset: Asset
    headline: str
    body: str
    hit_rate_recent: float
    sample_size: int
    cta: str = "Try the live prediction feed"
    generated_at: datetime
