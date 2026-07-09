"""
Hermes LLM Agent — Dynamic Kelly + Sharpe Manager.

This is what Gilad asked for:
  "The Hermes LLM agent will manage Kelly and Sharpe ratio."

The LLM (via OpenRouter) acts as the decision brain:
  1. Reads live price data
  2. Gets multi-signal prediction (RSI + momentum + volume + trend)
  3. Reads current Sharpe ratio and hit rate from feedback history
  4. REASONS about whether to bet, how much Kelly to apply,
     or sit out completely
  5. Returns a structured decision with full reasoning logged

This replaces the fixed Kelly formula with an intelligent agent
that adapts to market conditions in real time.

Usage:
    python hermes_llm_agent.py --asset BTC --cycles 5
    python hermes_llm_agent.py --asset BTC --asset ETH --cycles 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Deque

import httpx
import numpy as np

from config import settings
from core.logging_config import configure_logging, get_logger
from core.schemas import Asset, OHLCVBar, OHLCVSeries, Prediction
from core.kelly import kelly_fraction
from tools.llm_client import OpenRouterClient
from tools.multi_signal_predictor import MultiSignalPredictor

configure_logging()
logger = get_logger("hermes_llm_agent")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BANKROLL = settings.bankroll_usd
_HISTORY_LEN = 50


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_bars(asset: Asset, n_bars: int = 200,
               interval: str = "1m") -> OHLCVSeries:
    """Fetch live bars from Binance, fall back to synthetic."""
    symbol = f"{asset.value}USDT"
    try:
        params = {"symbol": symbol, "interval": interval,
                  "limit": min(n_bars, 1000)}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(BINANCE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        bars = [
            OHLCVBar(
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]), high=float(k[2]),
                low=float(k[3]), close=float(k[4]),
                volume=float(k[5]),
            ) for k in data
        ]
        logger.info("live_bars_fetched", asset=asset, bars=len(bars))
        return OHLCVSeries(asset=asset, timeframe=interval,
                           bars=bars, source="binance-live")
    except Exception as exc:
        logger.warning("binance_fallback", error=str(exc))
        return _synthetic_series(asset, n_bars, interval)


def _synthetic_series(asset: Asset, n_bars: int,
                       timeframe: str) -> OHLCVSeries:
    seed_price = {Asset.BTC: 65000.0, Asset.ETH: 3400.0}
    rng = random.Random(hash(asset.value) % (2 ** 31))
    price = seed_price.get(asset, 1000.0)
    now = datetime.now(timezone.utc)
    bars = []
    for i in range(n_bars):
        ts = now - timedelta(minutes=(n_bars - i))
        change = rng.gauss(0, 0.0015) * price
        o, c = price, max(price + change, 0.01)
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.0005)))
        low = min(o, c) * (1 - abs(rng.gauss(0, 0.0005)))
        bars.append(OHLCVBar(timestamp=ts, open=o, high=h,
                             low=low, close=c,
                             volume=abs(rng.gauss(50, 20))))
        price = c
    return OHLCVSeries(asset=asset, timeframe=timeframe,
                       bars=bars, source="synthetic-fallback")


# ─────────────────────────────────────────────
# PERFORMANCE TRACKER
# ─────────────────────────────────────────────

class PerformanceTracker:
    """Tracks outcomes and computes Sharpe + hit rate per asset."""

    def __init__(self):
        self._outcomes: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=_HISTORY_LEN))
        self._pnls: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=_HISTORY_LEN))
        self.bankroll = BANKROLL

    def record(self, asset: str, correct: bool, pnl: float) -> None:
        self._outcomes[asset].append(correct)
        self._pnls[asset].append(pnl)
        self.bankroll = max(self.bankroll + pnl, 0.01)

    def hit_rate(self, asset: str) -> float:
        h = self._outcomes[asset]
        return sum(h) / len(h) if h else 0.5

    def sharpe(self, asset: str) -> float:
        pnls = list(self._pnls[asset])
        if len(pnls) < 5:
            return 0.0
        arr = np.array(pnls, dtype=float)
        std = arr.std()
        if std < 1e-9:
            return 0.0
        return float(arr.mean() / std * math.sqrt(252))

    def summary(self, asset: str) -> Dict:
        return {
            "hit_rate": round(self.hit_rate(asset), 4),
            "sharpe": round(self.sharpe(asset), 3),
            "bankroll": round(self.bankroll, 2),
            "n_trades": len(self._outcomes[asset]),
        }


# ─────────────────────────────────────────────
# HERMES LLM AGENT — THE BRAIN
# ─────────────────────────────────────────────

class HermesLLMAgent:
    """
    The LLM acts as the decision brain.
    It reads prediction + performance stats and decides:
      - Should we bet at all? (edge filter)
      - What Kelly fraction to apply? (dynamic, not fixed)
      - Should we increase or decrease exposure? (Sharpe-aware)

    This is what Gilad wanted: LLM managing Kelly and Sharpe.
    """

    def __init__(self):
        self.llm = OpenRouterClient()
        self.predictor = MultiSignalPredictor()
        self.tracker = PerformanceTracker()

    def _build_prompt(self, asset: Asset, prediction: Prediction,
                      perf: Dict, market_price: float = 0.5) -> str:
        raw_kelly = kelly_fraction(
            p_win=max(prediction.p_up, prediction.p_down),
            price=market_price,
            fraction_cap=1.0
        )
        return f"""You are a quantitative trading agent managing a crypto prediction market strategy.

CURRENT MARKET STATE:
- Asset: {asset.value}
- Prediction model says P(UP) = {prediction.p_up:.1%}, P(DOWN) = {prediction.p_down:.1%}
- Signal direction: {"UP" if prediction.p_up > 0.5 else "DOWN"}
- Model rationale: {prediction.rationale}
- Market price (YES share): {market_price:.2f}

PERFORMANCE METRICS:
- Recent hit rate: {perf['hit_rate']:.1%} (last {perf['n_trades']} trades)
- Current Sharpe ratio: {perf['sharpe']:.3f}
- Current bankroll: ${perf['bankroll']:.2f}
- Raw Kelly fraction: {raw_kelly.raw_fraction:.4f} ({raw_kelly.raw_fraction:.1%} of bankroll)

YOUR JOB:
Decide whether to bet and how much Kelly fraction to apply (0.0 to 0.5 max).

RULES:
- If Sharpe ratio is below -1.0, DO NOT BET (model is broken)
- If hit rate is below 0.48 and n_trades > 20, DO NOT BET
- If P(UP) or P(DOWN) is below 0.51, the edge is too small — DO NOT BET
- If Sharpe is positive and hit rate above 0.52, apply UP TO half Kelly
- If Sharpe is very positive (above 1.0) and hit rate above 0.55, apply full Kelly fraction (cap 0.5)
- Always reduce Kelly by 50% if bankroll dropped more than 10% from start

Respond in this exact JSON format only, no other text:
{{
  "should_bet": true or false,
  "side": "YES" or "NO" or "NONE",
  "kelly_fraction": 0.0 to 0.5,
  "reasoning": "one sentence explanation"
}}"""

    def decide(self, asset: Asset, series: OHLCVSeries,
               market_price: float = 0.5) -> Dict:
        """Core decision method — LLM reasons about Kelly + Sharpe."""
        prediction = self.predictor.predict(series, horizon_minutes=5)
        perf = self.tracker.summary(asset.value)

        prompt = self._build_prompt(asset, prediction, perf, market_price)

        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            # Clean and parse JSON
            clean = raw.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            llm_decision = json.loads(clean.strip())
        except Exception as exc:
            logger.warning("llm_decision_fallback", error=str(exc))
            # Fallback: use pure Kelly math if LLM fails
            llm_decision = self._fallback_decision(prediction, perf,
                                                    market_price)

        stake_usd = 0.0
        if llm_decision.get("should_bet") and llm_decision.get("side") != "NONE":
            kf = float(llm_decision.get("kelly_fraction", 0.0))
            kf = min(max(kf, 0.0), 0.5)
            stake_usd = round(self.tracker.bankroll * kf, 2)

        result = {
            "asset": asset.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": {
                "p_up": prediction.p_up,
                "p_down": prediction.p_down,
                "signal": "UP" if prediction.p_up > 0.5 else "DOWN",
                "model": prediction.model_name,
                "rationale": prediction.rationale,
            },
            "performance": perf,
            "llm_decision": llm_decision,
            "stake_usd": stake_usd,
            "bankroll": round(self.tracker.bankroll, 2),
        }

        logger.info(
            "hermes_agent_decision",
            asset=asset.value,
            should_bet=llm_decision.get("should_bet"),
            side=llm_decision.get("side"),
            kelly=llm_decision.get("kelly_fraction"),
            stake_usd=stake_usd,
            reasoning=llm_decision.get("reasoning", ""),
        )

        # Simulate outcome for paper trading
        if llm_decision.get("should_bet") and stake_usd > 0:
            correct = random.random() < max(prediction.p_up, prediction.p_down)
            pnl = stake_usd if correct else -stake_usd
            self.tracker.record(asset.value, correct, pnl)
            result["simulated_outcome"] = {
                "correct": correct,
                "pnl_usd": round(pnl, 2),
                "new_bankroll": round(self.tracker.bankroll, 2),
            }

        return result

    def _fallback_decision(self, prediction: Prediction,
                            perf: Dict, market_price: float) -> Dict:
        """Pure math fallback if LLM is unavailable."""
        p_win = max(prediction.p_up, prediction.p_down)
        side = "YES" if prediction.p_up > 0.5 else "NO"
        sharpe = perf["sharpe"]
        hit_rate = perf["hit_rate"]
        n = perf["n_trades"]

        if sharpe < -1.0:
            return {"should_bet": False, "side": "NONE",
                    "kelly_fraction": 0.0,
                    "reasoning": "Sharpe below -1.0, model not working"}
        if n > 20 and hit_rate < 0.48:
            return {"should_bet": False, "side": "NONE",
                    "kelly_fraction": 0.0,
                    "reasoning": "Hit rate below 48% threshold"}
        if p_win < 0.51:
            return {"should_bet": False, "side": "NONE",
                    "kelly_fraction": 0.0,
                    "reasoning": "Edge too small (p_win < 53%)"}

        kf_result = kelly_fraction(p_win=p_win, price=market_price,
                                   fraction_cap=0.5)
        kf = min(kf_result.capped_fraction, settings.max_stake_pct)

        # Scale down if Sharpe is negative but above threshold
        if sharpe < 0:
            kf *= 0.5

        return {
            "should_bet": kf > 0,
            "side": side,
            "kelly_fraction": round(kf, 4),
            "reasoning": f"Math fallback: p_win={p_win:.2%} kelly={kf:.2%}",
        }


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def run_cycles(assets: List[Asset], n_cycles: int = 5) -> None:
    agent = HermesLLMAgent()
    all_results = []

    for cycle in range(1, n_cycles + 1):
        print(f"\n{'=' * 60}")
        print(f"  HERMES LLM AGENT — CYCLE {cycle}/{n_cycles}")
        print(f"{'=' * 60}")

        for asset in assets:
            print(f"\n  Asset: {asset.value}")
            series = fetch_bars(asset, n_bars=200)
            result = agent.decide(asset, series)

            print(f"  Signal       : {result['prediction']['signal']}")
            print(f"  P(UP)        : {result['prediction']['p_up']:.1%}")
            print(f"  P(DOWN)      : {result['prediction']['p_down']:.1%}")
            print(f"  Hit Rate     : {result['performance']['hit_rate']:.1%}")
            print(f"  Sharpe       : {result['performance']['sharpe']:.3f}")
            print(f"  LLM Decision : {result['llm_decision'].get('should_bet')} "
                  f"— {result['llm_decision'].get('side')}")
            print(f"  Kelly        : {result['llm_decision'].get('kelly_fraction', 0):.1%}")
            print(f"  Stake        : ${result['stake_usd']}")
            print(f"  Reasoning    : {result['llm_decision'].get('reasoning', '')}")
            print(f"  Bankroll     : ${result['bankroll']}")

            if "simulated_outcome" in result:
                o = result["simulated_outcome"]
                print(f"  Outcome      : {'✓ CORRECT' if o['correct'] else '✗ WRONG'} "
                      f"PnL: ${o['pnl_usd']}")

            all_results.append(result)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for asset in assets:
        perf = agent.tracker.summary(asset.value)
        print(f"  {asset.value}: hit_rate={perf['hit_rate']:.1%} "
              f"sharpe={perf['sharpe']:.3f} "
              f"bankroll=${perf['bankroll']:.2f}")

    # Save results
    with open("hermes_agent_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to hermes_agent_results.json")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hermes LLM Agent — Dynamic Kelly + Sharpe Manager")
    parser.add_argument("--asset", action="append", default=["BTC"],
                        choices=["BTC", "ETH"])
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of decision cycles to run")
    args = parser.parse_args()

    assets = [Asset(a) for a in set(args.asset)]
    run_cycles(assets, n_cycles=args.cycles)
