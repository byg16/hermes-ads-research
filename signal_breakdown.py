"""
signal_breakdown.py — Per-Signal Hourly + Daily Breakdown

Exactly what Gilad asked for:
- Run hourly + daily breakdown for ONLY 5min:n+1
- Run hourly + daily breakdown for ONLY 1min:n+5
- Run hourly + daily breakdown for ONLY 5min:n+2
- Find which hours each signal has edge independently
- Compare the 3 schemas to find overlapping edge windows

This tells us:
  "At 14:00 UTC, does 5min:n+1 alone have edge?"
  "At 14:00 UTC, does 1min:n+5 alone have edge?"
  "Where do all 3 overlap? That's our highest conviction window."

Usage:
    python signal_breakdown.py --asset BTC
    python signal_breakdown.py --asset BTC --asset ETH
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import httpx

from core.logging_config import configure_logging, get_logger
from core.schemas import Asset, OHLCVBar, OHLCVSeries
from tools.multi_signal_predictor import MultiSignalPredictor

configure_logging()
logger = get_logger("signal_breakdown")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BANKROLL = 1000.0
STAKE_PCT = 0.05
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_history(asset: Asset, interval: str,
                  months: int = 3) -> List[OHLCVBar]:
    symbol = f"{asset.value}USDT"
    bars = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (months * 30 * 24 * 60 * 60 * 1000)
    current_start = start_time

    logger.info(f"Fetching {asset.value} {interval} ({months} months)...")
    with httpx.Client(timeout=30.0) as client:
        while current_start < end_time:
            try:
                resp = client.get(BINANCE_URL, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": current_start,
                    "endTime": end_time, "limit": 1000,
                })
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                bars.extend([
                    OHLCVBar(
                        timestamp=datetime.fromtimestamp(
                            k[0]/1000, tz=timezone.utc),
                        open=float(k[1]), high=float(k[2]),
                        low=float(k[3]), close=float(k[4]),
                        volume=float(k[5]),
                    ) for k in data
                ])
                current_start = data[-1][0] + 1
                if len(data) < 1000:
                    break
            except Exception as exc:
                logger.error(f"Fetch error: {exc}")
                break

    logger.info(f"Got {len(bars)} {interval} bars")
    return bars


# ─────────────────────────────────────────────
# PER-SIGNAL BREAKDOWN
# ─────────────────────────────────────────────

def run_signal_breakdown(
        bars_primary: List[OHLCVBar],
        asset: Asset,
        signal_name: str,
        horizon_minutes: int,
        timeframe: str,
        predictor: MultiSignalPredictor,
        actual_horizon_bars: int = 1,
        bars_all: List[OHLCVBar] = None,
        warmup: int = 100) -> Dict:
    """
    Run full hourly + daily breakdown for ONE signal only.
    No cascade — pure single signal analysis.
    """
    hourly = defaultdict(lambda: {
        "correct": 0, "total": 0, "pnl": 0.0})
    daily = defaultdict(lambda: {
        "correct": 0, "total": 0, "pnl": 0.0})

    bankroll = BANKROLL
    total_correct = 0
    total_trades = 0

    bars = bars_primary
    m1_dict = {b.timestamp: b for b in (bars_all or [])}

    for i in range(warmup, len(bars) - actual_horizon_bars - 1):
        current = bars[i]
        t = current.timestamp
        hour = t.hour
        dow = t.weekday()

        history = bars[:i]
        series = OHLCVSeries(asset=asset, timeframe=timeframe,
                              bars=history, source="backtest")

        # Actual future direction
        future_close = bars[i + actual_horizon_bars].close
        actual_up = future_close > current.close

        # For 1min:n+5 use 1min bars and check 5 min ahead
        if timeframe == "1m" and m1_dict:
            t_plus = t + timedelta(minutes=5)
            if t_plus not in m1_dict:
                continue
            actual_up = m1_dict[t_plus].close > current.close

        try:
            pred = predictor.predict(series,
                                     horizon_minutes=horizon_minutes)
        except Exception:
            continue

        predicted_up = pred.p_up > 0.5
        correct = predicted_up == actual_up

        # Only bet when confident
        p_win = pred.p_up if predicted_up else pred.p_down
        if p_win < 0.51:
            continue

        stake = bankroll * STAKE_PCT
        pnl = stake if correct else -stake
        bankroll = max(bankroll + pnl, 0.01)

        total_trades += 1
        if correct:
            total_correct += 1

        hourly[hour]["total"] += 1
        hourly[hour]["pnl"] += pnl
        if correct:
            hourly[hour]["correct"] += 1

        daily[dow]["total"] += 1
        daily[dow]["pnl"] += pnl
        if correct:
            daily[dow]["correct"] += 1

    # Build results
    hourly_results = {}
    for h in sorted(hourly.keys()):
        v = hourly[h]
        rate = v["correct"] / v["total"] if v["total"] > 0 else 0
        hourly_results[f"{h:02d}:00"] = {
            "trades": v["total"],
            "hit_rate": f"{rate:.1%}",
            "hit_rate_raw": round(rate, 4),
            "pnl": round(v["pnl"], 2),
            "edge": rate > 0.53,
        }

    daily_results = {}
    for d in sorted(daily.keys()):
        v = daily[d]
        rate = v["correct"] / v["total"] if v["total"] > 0 else 0
        daily_results[DAY_NAMES[d]] = {
            "trades": v["total"],
            "hit_rate": f"{rate:.1%}",
            "hit_rate_raw": round(rate, 4),
            "pnl": round(v["pnl"], 2),
            "edge": rate > 0.53,
        }

    overall_rate = total_correct / total_trades if total_trades > 0 else 0
    best_hours = [h for h, v in hourly_results.items() if v["edge"]]
    best_days = [d for d, v in daily_results.items() if v["edge"]]

    return {
        "signal": signal_name,
        "overall_hit_rate": f"{overall_rate:.1%}",
        "total_trades": total_trades,
        "total_correct": total_correct,
        "final_bankroll": round(bankroll, 2),
        "total_return": f"{((bankroll-BANKROLL)/BANKROLL):.1%}",
        "best_hours_utc": best_hours,
        "best_days": best_days,
        "hourly_breakdown": hourly_results,
        "daily_breakdown": daily_results,
    }


# ─────────────────────────────────────────────
# PRINT SINGLE SIGNAL REPORT
# ─────────────────────────────────────────────

def print_signal_report(r: Dict) -> None:
    signal = r.get("signal", "?")
    print(f"\n{'='*65}")
    print(f"  SIGNAL: {signal}")
    print(f"  Overall Hit Rate : {r.get('overall_hit_rate')}")
    print(f"  Total Trades     : {r.get('total_trades')}")
    print(f"  Final Bankroll   : ${r.get('final_bankroll')}")
    print(f"  Total Return     : {r.get('total_return')}")
    print(f"  Best Hours UTC   : {r.get('best_hours_utc')}")
    print(f"  Best Days        : {r.get('best_days')}")

    print(f"\n  HOURLY BREAKDOWN:")
    print(f"  {'Hour':<8} {'Hit Rate':<12} {'Trades':<10} "
          f"{'PnL':<12} {'Edge?'}")
    print(f"  {'-'*55}")
    for h, v in sorted(r.get("hourly_breakdown", {}).items(),
                        key=lambda x: x[1]["pnl"], reverse=True):
        edge = "✅ EDGE" if v["edge"] else ""
        print(f"  {h:<8} {v['hit_rate']:<12} "
              f"{v['trades']:<10} ${v['pnl']:<12} {edge}")

    print(f"\n  DAILY BREAKDOWN:")
    print(f"  {'Day':<8} {'Hit Rate':<12} {'Trades':<10} "
          f"{'PnL':<12} {'Edge?'}")
    print(f"  {'-'*50}")
    for day, v in r.get("daily_breakdown", {}).items():
        edge = "✅ EDGE" if v["edge"] else ""
        print(f"  {day:<8} {v['hit_rate']:<12} "
              f"{v['trades']:<10} ${v['pnl']:<12} {edge}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────
# OVERLAP ANALYSIS
# ─────────────────────────────────────────────

def find_overlapping_edge(results: Dict) -> Dict:
    """
    Find hours and days where MULTIPLE signals have edge.
    These are the highest confidence trading windows.
    """
    all_signals = list(results.keys())

    # Collect edge hours per signal
    edge_hours_per_signal = {}
    edge_days_per_signal = {}

    for sig, r in results.items():
        edge_hours_per_signal[sig] = set(r.get("best_hours_utc", []))
        edge_days_per_signal[sig] = set(r.get("best_days", []))

    # Find overlaps
    all_edge_hours = [v for v in edge_hours_per_signal.values()]
    all_edge_days = [v for v in edge_days_per_signal.values()]

    # Hours where ALL signals have edge
    if all_edge_hours:
        triple_overlap_hours = set.intersection(*all_edge_hours) \
            if len(all_edge_hours) == 3 else set()
        double_overlap_hours = set()
        for i in range(len(all_edge_hours)):
            for j in range(i+1, len(all_edge_hours)):
                double_overlap_hours.update(
                    all_edge_hours[i] & all_edge_hours[j])
    else:
        triple_overlap_hours = set()
        double_overlap_hours = set()

    # Days where ALL signals have edge
    if all_edge_days:
        triple_overlap_days = set.intersection(*all_edge_days) \
            if len(all_edge_days) == 3 else set()
    else:
        triple_overlap_days = set()

    overlap = {
        "schema_1_all_3_agree_hours": sorted(triple_overlap_hours),
        "schema_2_any_2_agree_hours": sorted(
            double_overlap_hours - triple_overlap_hours),
        "schema_3_best_days": sorted(triple_overlap_days),
        "per_signal_edge_hours": {
            sig: sorted(hours)
            for sig, hours in edge_hours_per_signal.items()
        },
        "per_signal_edge_days": {
            sig: sorted(days)
            for sig, days in edge_days_per_signal.items()
        },
        "recommendation": (
            f"Trade during {sorted(triple_overlap_hours)} UTC "
            f"when all 3 signals agree"
            if triple_overlap_hours
            else f"Trade during {sorted(double_overlap_hours)} UTC "
                 f"when any 2 signals agree"
        )
    }

    return overlap


def print_overlap_report(overlap: Dict) -> None:
    print(f"\n{'='*65}")
    print(f"  OVERLAP ANALYSIS — Where do signals agree on edge?")
    print(f"{'='*65}")

    schema1 = overlap.get("schema_1_all_3_agree_hours", [])
    schema2 = overlap.get("schema_2_any_2_agree_hours", [])
    schema3 = overlap.get("schema_3_best_days", [])

    print(f"\n  SCHEMA 1 — Hours where ALL 3 signals have edge:")
    print(f"  {schema1 if schema1 else 'None found — signals disagree on best hours'}")

    print(f"\n  SCHEMA 2 — Hours where ANY 2 signals have edge:")
    print(f"  {schema2 if schema2 else 'None found'}")

    print(f"\n  SCHEMA 3 — Days where all signals agree:")
    print(f"  {schema3 if schema3 else 'None found'}")

    print(f"\n  PER SIGNAL EDGE HOURS:")
    for sig, hours in overlap.get("per_signal_edge_hours", {}).items():
        print(f"  {sig}: {hours}")

    print(f"\n  RECOMMENDATION:")
    print(f"  {overlap.get('recommendation', 'Need more data')}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append",
                        default=["BTC"], choices=["BTC", "ETH"])
    parser.add_argument("--months", type=int, default=3)
    args = parser.parse_args()

    predictor = MultiSignalPredictor()
    all_results = {}

    for asset_str in set(args.asset):
        asset = Asset(asset_str)
        print(f"\n{'#'*65}")
        print(f"# SIGNAL BREAKDOWN: {asset_str}/USDT")
        print(f"{'#'*65}")

        bars_5m = fetch_history(asset, "5m", months=args.months)
        bars_1m = fetch_history(asset, "1m", months=args.months)

        if len(bars_5m) < 200 or len(bars_1m) < 500:
            print("Not enough data")
            continue

        m1_dict = {b.timestamp: b for b in bars_1m}
        signal_results = {}

        # Signal 1: 5min:n+1
        print(f"\nRunning 5min:n+1...")
        r1 = run_signal_breakdown(
            bars_5m, asset,
            signal_name="5min:n+1",
            horizon_minutes=5,
            timeframe="5m",
            predictor=predictor,
            actual_horizon_bars=1,
            warmup=100,
        )
        print_signal_report(r1)
        signal_results["5min_n1"] = r1

        # Signal 2: 1min:n+5
        print(f"\nRunning 1min:n+5...")
        r2 = run_signal_breakdown(
            bars_1m, asset,
            signal_name="1min:n+5",
            horizon_minutes=5,
            timeframe="1m",
            predictor=predictor,
            actual_horizon_bars=5,
            bars_all=bars_1m,
            warmup=100,
        )
        print_signal_report(r2)
        signal_results["1min_n5"] = r2

        # Signal 3: 5min:n+2
        print(f"\nRunning 5min:n+2...")
        r3 = run_signal_breakdown(
            bars_5m, asset,
            signal_name="5min:n+2",
            horizon_minutes=10,
            timeframe="5m",
            predictor=predictor,
            actual_horizon_bars=2,
            warmup=100,
        )
        print_signal_report(r3)
        signal_results["5min_n2"] = r3

        # Overlap analysis
        overlap = find_overlapping_edge(signal_results)
        print_overlap_report(overlap)

        all_results[asset_str] = {
            "signals": signal_results,
            "overlap_analysis": overlap,
        }

    # Save results
    output = "signal_breakdown_results.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output}")
    print("NOTE: These are BACKTEST results. Not sent to dashboard.")


if __name__ == "__main__":
    main()
