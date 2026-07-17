"""
Backtest module for Hermes Ads Research pipeline.

Downloads real historical BTC/ETH OHLCV data from Binance free API,
replays it bar by bar through the existing Kronos predictor and Kelly
risk manager, and produces a full performance report:
  - Hit rate per asset
  - Sharpe ratio
  - Max drawdown
  - Cumulative PnL
  - Walk-forward validation results

Usage:
    python backtest.py --asset BTC --bars 1000 --horizon 5
    python backtest.py --asset ETH --bars 1000 --walkforward
    python backtest.py --asset BTC --asset ETH --bars 1000 --walkforward
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import httpx
import numpy as np

from core.schemas import Asset, OHLCVBar, OHLCVSeries
from core.kelly import kelly_fraction
from tools.kronos_client import KronosClient
from core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger("backtest")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BANKROLL = 1000.0
KELLY_CAP = 0.5
MAX_STAKE_PCT = 0.05


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def fetch_historical_bars(asset: Asset, n_bars: int = 1000,
                           interval: str = "1m") -> List[OHLCVBar]:
    """Fetch historical OHLCV bars from Binance public API."""
    symbol = f"{asset.value}USDT"
    logger.info("fetching_historical_data", symbol=symbol, n_bars=n_bars)
    params = {"symbol": symbol, "interval": interval, "limit": min(n_bars, 1000)}
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(BINANCE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        bars = [
            OHLCVBar(
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]), high=float(k[2]),
                low=float(k[3]), close=float(k[4]),
                volume=float(k[5]),
            )
            for k in data
        ]
        logger.info("historical_data_fetched", bars=len(bars))
        return bars
    except Exception as exc:
        logger.error("binance_fetch_failed", error=str(exc))
        raise


# ─────────────────────────────────────────────
# SINGLE WINDOW BACKTEST
# ─────────────────────────────────────────────

def run_window(bars: List[OHLCVBar], asset: Asset,
               kronos: KronosClient, horizon: int = 5,
               warmup: int = 50) -> Dict:
    """
    Replay bars through predictor + Kelly.
    Returns dict of metrics for this window.
    """
    bankroll = BANKROLL
    pnl_curve = [0.0]
    results = []

    for i in range(warmup, len(bars) - horizon):
        history = bars[:i]
        future_close = bars[i + horizon - 1].close
        current_close = bars[i - 1].close
        actual_direction = "UP" if future_close > current_close else "DOWN"

        series = OHLCVSeries(asset=asset, timeframe="1m", bars=history)
        prediction = kronos.predict(series, horizon_minutes=horizon)

        # Simulate market price (use prediction as proxy for demo)
        market_price = 0.5
        side = "YES" if prediction.p_up > 0.5 else "NO"
        p_win = prediction.p_up if side == "YES" else prediction.p_down

        kelly = kelly_fraction(p_win=p_win, price=market_price,
                               fraction_cap=KELLY_CAP)
        stake_pct = min(kelly.capped_fraction, MAX_STAKE_PCT)
        stake = bankroll * stake_pct

        predicted_direction = "UP" if side == "YES" else "DOWN"
        correct = predicted_direction == actual_direction

        # Simple binary payout: win stake, lose stake
        pnl = stake if correct else -stake
        bankroll = max(bankroll + pnl, 0.01)
        pnl_curve.append(bankroll - BANKROLL)

        results.append({
            "bar": i,
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": correct,
            "p_up": prediction.p_up,
            "stake": round(stake, 2),
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    return _compute_metrics(results, pnl_curve)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def _compute_metrics(results: List[Dict], pnl_curve: List[float]) -> Dict:
    if not results:
        return {}

    n = len(results)
    hits = sum(1 for r in results if r["correct"])
    hit_rate = hits / n

    pnls = [r["pnl"] for r in results]
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if np.std(pnls) > 0 else 1e-9
    sharpe = (avg_pnl / std_pnl) * math.sqrt(252 * 24 * 60)

    # Max drawdown
    peak = BANKROLL
    max_dd = 0.0
    running = BANKROLL
    for r in results:
        running += r["pnl"]
        if running > peak:
            peak = running
        dd = (peak - running) / peak
        if dd > max_dd:
            max_dd = dd

    total_pnl = sum(pnls)
    final_bankroll = BANKROLL + total_pnl

    return {
        "trades": n,
        "hit_rate": round(hit_rate, 4),
        "hit_rate_pct": f"{hit_rate:.1%}",
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": f"{max_dd:.1%}",
        "total_pnl_usd": round(total_pnl, 2),
        "final_bankroll_usd": round(final_bankroll, 2),
        "return_pct": f"{((final_bankroll - BANKROLL) / BANKROLL):.1%}",
    }


# ─────────────────────────────────────────────
# WALK FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward(bars: List[OHLCVBar], asset: Asset,
                 kronos: KronosClient, horizon: int = 5,
                 train_size: int = 500,
                 test_size: int = 100) -> List[Dict]:
    """
    Walk forward validation:
    Train on `train_size` bars, test on next `test_size`, slide forward.
    Returns list of metrics per window.
    """
    windows = []
    start = 0
    window_num = 1

    while start + train_size + test_size <= len(bars):
        test_bars = bars[start: start + train_size + test_size]
        logger.info("walk_forward_window", window=window_num,
                    start=start, end=start + train_size + test_size)
        metrics = run_window(test_bars, asset, kronos,
                             horizon=horizon, warmup=train_size)
        metrics["window"] = window_num
        metrics["bar_start"] = start
        metrics["bar_end"] = start + train_size + test_size
        windows.append(metrics)
        start += test_size
        window_num += 1

    return windows


# ─────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(asset: Asset, metrics: Dict,
                 wf_windows: List[Dict] = None) -> None:
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {asset.value}/USDT")
    print(f"{'=' * 60}")
    print(f"  Total trades     : {metrics.get('trades', 0)}")
    print(f"  Hit rate         : {metrics.get('hit_rate_pct', 'N/A')}")
    print(f"  Sharpe ratio     : {metrics.get('sharpe_ratio', 'N/A')}")
    print(f"  Max drawdown     : {metrics.get('max_drawdown_pct', 'N/A')}")
    print(f"  Total PnL        : ${metrics.get('total_pnl_usd', 0)}")
    print(f"  Final bankroll   : ${metrics.get('final_bankroll_usd', BANKROLL)}")
    print(f"  Return           : {metrics.get('return_pct', '0%')}")

    if wf_windows:
        print(f"\n  WALK-FORWARD VALIDATION ({len(wf_windows)} windows)")
        print(f"  {'Window':<8} {'Trades':<8} {'Hit Rate':<12} "
              f"{'Sharpe':<10} {'PnL':<10}")
        print(f"  {'-' * 50}")
        for w in wf_windows:
            print(f"  {w['window']:<8} {w['trades']:<8} "
                  f"{w['hit_rate_pct']:<12} {w['sharpe_ratio']:<10} "
                  f"${w['total_pnl_usd']:<10}")

        avg_hit = sum(w["hit_rate"] for w in wf_windows) / len(wf_windows)
        avg_sharpe = sum(w["sharpe_ratio"] for w in wf_windows) / len(wf_windows)
        print(f"\n  Average hit rate  : {avg_hit:.1%}")
        print(f"  Average Sharpe    : {avg_sharpe:.3f}")

    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hermes Ads Research Backtester")
    parser.add_argument("--asset", action="append", default=["BTC"],
                        choices=["BTC", "ETH"],
                        help="Asset to backtest (can use multiple times)")
    parser.add_argument("--bars", type=int, default=1000,
                        help="Number of historical bars to fetch")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Prediction horizon in minutes")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    kronos = KronosClient()
    all_results = {}

    for asset_str in set(args.asset):
        asset = Asset(asset_str)
        print(f"\nFetching {args.bars} bars for {asset.value}...")

        try:
            bars = fetch_historical_bars(asset, n_bars=args.bars)
        except Exception as exc:
            print(f"Could not fetch data for {asset.value}: {exc}")
            print("Make sure you have internet access.")
            continue

        print(f"Running backtest on {len(bars)} bars...")
        metrics = run_window(bars, asset, kronos, horizon=args.horizon)

        wf_windows = None
        if args.walkforward and len(bars) >= 600:
            print("Running walk-forward validation...")
            wf_windows = walk_forward(bars, asset, kronos,
                                      horizon=args.horizon)

        print_report(asset, metrics, wf_windows)
        all_results[asset_str] = {
            "metrics": metrics,
            "walk_forward": wf_windows,
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()
