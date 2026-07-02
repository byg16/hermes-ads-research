"""Centralized, env-driven configuration for the whole pipeline."""
from __future__ import annotations

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Apify
    apify_api_token: str = ""
    apify_ohlcv_actor_id: str = "your-username~crypto-ohlcv-scraper"
    apify_bars_lookback: int = 1000

    # Polymarket
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"

    # Kalshi
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""

    # Kronos
    kronos_mode: str = "stub"  # "local" | "stub"
    kronos_model_name: str = "NeoQuasar/Kronos-small"
    kronos_lookback: int = 512
    kronos_horizon: int = 5

    # Risk
    kelly_fraction_cap: float = 0.5
    max_stake_pct: float = 0.05
    bankroll_usd: float = 1000.0
    paper_trading: bool = True

    # Orchestration
    poll_interval_seconds: int = 60
    log_level: str = "INFO"
    assets: str = "BTC,ETH"

    @property
    def asset_list(self) -> List[str]:
        return [a.strip().upper() for a in self.assets.split(",") if a.strip()]


settings = Settings()
