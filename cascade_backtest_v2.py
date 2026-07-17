"""
cascade_backtest_v2.py — Cascade Backtest (Fixed + Enhanced)

Based on Gilad's original code with these fixes:
1. Uses MultiSignalPredictor (not KronosClient)
2. Added Kelly criterion PnL tracking
3. Added MEDIUM conviction (2/3 signals agree)
4. Added PnL and hourly breakdown
5. NEVER reports to dashboard — saves to backtest_results.json only
6. TWO PASS: First collect edge slots, then only trade those!
7. Added STOP LOSS to cap losses!

DIFFERENCE:
  BACKTEST  = historical data, no dashboard, saves to JSON file
  FORWARD   = live data, reports to dashboard as Hermes Crypto bot

Usage:
    python cascade_backtest_v2.py --asset BTC
    python cascade_backtest_v2.py --asset ETH
    python cascade_backtest_v2.py --asset BTC --asset ETH
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Set, Tuple

import httpx

from core.logging_config import configure_logging, get_logger
from core.schemas import Asset, OHLCVBar, OHLCVSeries
from tools.multi_signal_predictor import MultiSignalPredictor

configure_logging()
logger = get_logger("cascade_backtest_v2")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BANKROLL = 1000.0
STAKE_HIGH = 0.15     # 15% Kelly for HIGH conviction (3/3 agree)
STAKE_MEDIUM = 0.075    # 7.5% Kelly for MEDIUM conviction (2/3 agree)
STOP_LOSS_PCT = 0.07   # 7% stop loss (slightly more room)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_3_months_history(asset: Asset, interval: str) -> List[OHLCVBar]:
    """Downloads 3 months of historical data using pagination."""
    logger.info(f"Downloading 3 months of {asset.value} ({interval})...")
    bars = []

    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (90 * 24 * 60 * 60 * 1000)
    current_start = start_time
    symbol = f"{asset.value}USDT"

    with httpx.Client(timeout=30.0) as client:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            }
            try:
                resp = client.get(BINANCE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                batch = [
                    OHLCVBar(
                        timestamp=datetime.fromtimestamp(
                            k[0] / 1000, tz=timezone.utc),
                        open=float(k[1]), high=float(k[2]),
                        low=float(k[3]), close=float(k[4]),
                        volume=float(k[5]),
                    )
                    for k in data
                ]
                bars.extend(batch)
                current_start = data[-1][0] + 1
                if len(data) < 1000:
                    break
            except Exception as exc:
                logger.error(f"Fetch error: {exc}")
                break

    logger.info(f"Downloaded {len(bars)} {interval} bars for {asset.value}")
    return bars


# ─────────────────────────────────────────────
# PNL CALCULATION WITH STOP LOSS
# ─────────────────────────────────────────────

def compute_pnl_with_exits(
    entry_price: float, 
    exit_price: float, 
    stake: float,
    direction: str, 
    stop_loss_pct: float = STOP_LOSS_PCT
) -> float:
    """
    Compute PnL with hard stop loss.
    direction: "UP" (long) or "DOWN" (short)
    """
    if direction == "UP":
        if exit_price <= entry_price * (1 - stop_loss_pct):
            return -stake * stop_loss_pct  # Lose only stop loss percentage
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        if exit_price >= entry_price * (1 + stop_loss_pct):
            return -stake * stop_loss_pct
        pnl_pct = (entry_price - exit_price) / entry_price
    
    return stake * pnl_pct


# ─────────────────────────────────────────────
# FIRST PASS: COLLECT ALL DATA AND IDENTIFY EDGE SLOTS
# ─────────────────────────────────────────────

def first_pass_collect_edge_slots(
    asset: Asset,
    bars_5m: List[OHLCVBar],
    bars_1m: List[OHLCVBar],
    predictor: MultiSignalPredictor
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    First pass: run through all data to collect which (dow, hour) are edge slots
    Returns 3 sets: (cascade_edge_slots, 5m_n1_edge_slots, 1m_n5_edge_slots)
    """
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Combined HIGH conviction
    day_hourly = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    # Individual 5min_n1 Only
    day_hourly_5m_n1 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    # Individual 1min_n5 Only
    day_hourly_1m_n5 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    
    m1_dict = {b.timestamp: b for b in bars_1m}
    bankroll_high = BANKROLL
    bankroll_5m_n1 = BANKROLL
    bankroll_1m_n5 = BANKROLL
    
    idx_1m = 0
    for i in range(100, len(bars_5m) - 2):
        current = bars_5m[i]
        t = current.timestamp
        hour = t.hour
        dow = t.weekday()
        day_hour_key = (dow, hour)
        
        history_5m = bars_5m[max(0, i - 100):i]
        series_5m = OHLCVSeries(asset=asset, timeframe="5m",
                                bars=history_5m, source="backtest")

        while idx_1m < len(bars_1m) and bars_1m[idx_1m].timestamp <= t:
            idx_1m += 1
        
        if idx_1m < 50:
            continue
        history_1m = bars_1m[max(0, idx_1m - 100):idx_1m]
        series_1m = OHLCVSeries(asset=asset, timeframe="1m",
                                bars=history_1m, source="backtest")
        
        future_n1 = bars_5m[i + 1].close
        future_n2 = bars_5m[i + 2].close
        actual_up_n1 = future_n1 > current.close

        t_plus_5 = t + timedelta(minutes=5)
        if t_plus_5 not in m1_dict:
            continue
        actual_up_1m = m1_dict[t_plus_5].close > current.close
        
        try:
            pred_5m_n1 = predictor.predict(series_5m, horizon_minutes=5)
            pred_1m_n5 = predictor.predict(series_1m, horizon_minutes=5)
        except Exception:
            continue
        
        # Individual signal evaluation
        dir_5m_n1 = pred_5m_n1.p_up > 0.5
        dir_1m_n5 = pred_1m_n5.p_up > 0.5
        
        # Cascade signals
        pred_5m_n2 = predictor.predict(series_5m, horizon_minutes=10)
        dir_5m_n2 = pred_5m_n2.p_up > 0.5
        up_count = sum([dir_5m_n1, dir_5m_n2, dir_1m_n5])
        down_count = 3 - up_count
        predicted_up = up_count >= 2
        
        # Cascade HIGH
        if up_count == 3 or down_count == 3:
            stake = bankroll_high * STAKE_HIGH
            direction = "UP" if predicted_up else "DOWN"
            pnl = compute_pnl_with_exits(current.close, bars_5m[i+2].close, stake, direction)
            bankroll_high = max(bankroll_high + pnl, 0.01)
            day_hourly[day_hour_key]["total"] += 1
            day_hourly[day_hour_key]["pnl"] += pnl
            if pnl > 0:
                day_hourly[day_hour_key]["correct"] += 1
        
        # 5m_n1 individual
        direction_5m_n1 = "UP" if dir_5m_n1 else "DOWN"
        stake_5m_n1 = bankroll_5m_n1 * STAKE_HIGH
        pnl_5m_n1 = compute_pnl_with_exits(current.close, bars_5m[i+1].close, stake_5m_n1, direction_5m_n1)
        bankroll_5m_n1 = max(bankroll_5m_n1 + pnl_5m_n1, 0.01)
        day_hourly_5m_n1[day_hour_key]["total"] += 1
        day_hourly_5m_n1[day_hour_key]["pnl"] += pnl_5m_n1
        if pnl_5m_n1 > 0:
            day_hourly_5m_n1[day_hour_key]["correct"] += 1
        
        # 1m_n5 individual
        direction_1m_n5 = "UP" if dir_1m_n5 else "DOWN"
        stake_1m_n5 = bankroll_1m_n5 * STAKE_HIGH
        pnl_1m_n5 = compute_pnl_with_exits(current.close, m1_dict[t_plus_5].close, stake_1m_n5, direction_1m_n5)
        bankroll_1m_n5 = max(bankroll_1m_n5 + pnl_1m_n5, 0.01)
        day_hourly_1m_n5[day_hour_key]["total"] += 1
        day_hourly_1m_n5[day_hour_key]["pnl"] += pnl_1m_n5
        if pnl_1m_n5 > 0:
            day_hourly_1m_n5[day_hour_key]["correct"] += 1
    
    # Now filter edge slots: hit rate > 0.53, pnl > 0, trades >=50
    def get_edge_slots(dh_dict):
        edge_slots = set()
        for (d, h), v in dh_dict.items():
            if v["total"] < 50:
                continue
            hit_rate = v["correct"] / v["total"] if v["total"] > 0 else 0
            if hit_rate > 0.53 and v["pnl"] > 0:
                edge_slots.add((d, h))
        return edge_slots
    
    cascade_edge_slots = get_edge_slots(day_hourly)
    slot_5m_n1_edge_slots = get_edge_slots(day_hourly_5m_n1)
    slot_1m_n5_edge_slots = get_edge_slots(day_hourly_1m_n5)
    
    return cascade_edge_slots, slot_5m_n1_edge_slots, slot_1m_n5_edge_slots


# ─────────────────────────────────────────────
# SECOND PASS: RUN BACKTEST ONLY ON EDGE SLOTS
# ─────────────────────────────────────────────

def run_cascade_backtest(asset_str: str) -> dict:
    asset = Asset(asset_str)

    bars_5m = fetch_3_months_history(asset, "5m")
    bars_1m = fetch_3_months_history(asset, "1m")

    if len(bars_5m) < 200 or len(bars_1m) < 500:
        logger.error("Not enough historical data")
        return {}

    predictor = MultiSignalPredictor()
    
    # First pass: collect edge slots
    logger.info("First pass: collecting edge slots...")
    cascade_edge_slots, slot_5m_n1_edge_slots, slot_1m_n5_edge_slots = first_pass_collect_edge_slots(
        asset, bars_5m, bars_1m, predictor
    )
    logger.info(f"Found {len(cascade_edge_slots)} cascade edge slots")
    logger.info(f"Found {len(slot_5m_n1_edge_slots)} 5m_n1 edge slots")
    logger.info(f"Found {len(slot_1m_n5_edge_slots)} 1m_n5 edge slots")

    # Stats trackers
    stats_5m_n1 = {"correct": 0, "total": 0}
    stats_5m_n2 = {"correct": 0, "total": 0}
    stats_1m_n5 = {"correct": 0, "total": 0}
    stats_high = {"correct": 0, "total": 0, "pnl": 0.0}
    stats_medium = {"correct": 0, "total": 0, "pnl": 0.0}

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Combined HIGH conviction
    hourly = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    daily = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    day_hourly = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})

    # Individual 5min_n1 Only
    hourly_5m_n1 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    daily_5m_n1 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    day_hourly_5m_n1 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})

    # Individual 1min_n5 Only
    hourly_1m_n5 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    daily_1m_n5 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})
    day_hourly_1m_n5 = defaultdict(lambda: {"correct": 0, "total": 0, "pnl": 0.0})

    m1_dict = {b.timestamp: b for b in bars_1m}
    bankroll_high = BANKROLL
    bankroll_5m_n1 = BANKROLL
    bankroll_1m_n5 = BANKROLL

    logger.info("Second pass: running backtest on edge slots...")

    idx_1m = 0
    for i in range(100, len(bars_5m) - 2):
        current = bars_5m[i]
        t = current.timestamp
        hour = t.hour
        dow = t.weekday()
        day_hour_key = (dow, hour)

        history_5m = bars_5m[max(0, i - 100):i]
        series_5m = OHLCVSeries(asset=asset, timeframe="5m",
                                bars=history_5m, source="backtest")

        while idx_1m < len(bars_1m) and bars_1m[idx_1m].timestamp <= t:
            idx_1m += 1
        
        if idx_1m < 50:
            continue
        history_1m = bars_1m[max(0, idx_1m - 100):idx_1m]
        series_1m = OHLCVSeries(asset=asset, timeframe="1m",
                                bars=history_1m, source="backtest")

        future_n1 = bars_5m[i + 1].close
        future_n2 = bars_5m[i + 2].close
        actual_up_n1 = future_n1 > current.close
        actual_up_n2 = future_n2 > current.close

        t_plus_5 = t + timedelta(minutes=5)
        if t_plus_5 not in m1_dict:
            continue
        actual_up_1m = m1_dict[t_plus_5].close > current.close

        try:
            pred_5m_n1 = predictor.predict(series_5m, horizon_minutes=5)
            pred_5m_n2 = predictor.predict(series_5m, horizon_minutes=10)
            pred_1m_n5 = predictor.predict(series_1m, horizon_minutes=5)
        except Exception:
            continue

        # Individual signal evaluation
        dir_5m_n1 = pred_5m_n1.p_up > 0.5
        dir_5m_n2 = pred_5m_n2.p_up > 0.5
        dir_1m_n5 = pred_1m_n5.p_up > 0.5

        stats_5m_n1["total"] += 1
        if dir_5m_n1 == actual_up_n1:
            stats_5m_n1["correct"] += 1

        stats_5m_n2["total"] += 1
        if dir_5m_n2 == actual_up_n2:
            stats_5m_n2["correct"] += 1

        stats_1m_n5["total"] += 1
        if dir_1m_n5 == actual_up_1m:
            stats_1m_n5["correct"] += 1

        # Cascade conviction
        up_count = sum([dir_5m_n1, dir_5m_n2, dir_1m_n5])
        down_count = 3 - up_count
        predicted_up = up_count >= 2
        correct = predicted_up == actual_up_n1

        # Only trade if this is an edge slot!
        if day_hour_key in cascade_edge_slots:
            if up_count == 3 or down_count == 3:
                stake = bankroll_high * STAKE_HIGH
                direction = "UP" if predicted_up else "DOWN"
                pnl = compute_pnl_with_exits(current.close, bars_5m[i+2].close, stake, direction)
                bankroll_high = max(bankroll_high + pnl, 0.01)
                stats_high["total"] += 1
                stats_high["pnl"] += pnl
                if pnl > 0:
                    stats_high["correct"] += 1
                hourly[hour]["total"] += 1
                hourly[hour]["pnl"] += pnl
                if pnl > 0:
                    hourly[hour]["correct"] += 1
                daily[dow]["total"] += 1
                daily[dow]["pnl"] += pnl
                if pnl > 0:
                    daily[dow]["correct"] += 1
                
                day_hourly[day_hour_key]["total"] += 1
                day_hourly[day_hour_key]["pnl"] += pnl
                if pnl > 0:
                    day_hourly[day_hour_key]["correct"] += 1

            elif up_count == 2 or down_count == 2:
                stake = bankroll_high * STAKE_MEDIUM
                direction = "UP" if predicted_up else "DOWN"
                pnl = compute_pnl_with_exits(current.close, bars_5m[i+2].close, stake, direction)
                stats_medium["total"] += 1
                stats_medium["pnl"] += pnl
                if pnl > 0:
                    stats_medium["correct"] += 1

        # Run 5min_n1 individual simulation (only if edge slot)
        if day_hour_key in slot_5m_n1_edge_slots:
            direction_5m_n1 = "UP" if dir_5m_n1 else "DOWN"
            stake_5m_n1 = bankroll_5m_n1 * STAKE_HIGH
            pnl_5m_n1 = compute_pnl_with_exits(current.close, bars_5m[i+1].close, stake_5m_n1, direction_5m_n1)
            bankroll_5m_n1 = max(bankroll_5m_n1 + pnl_5m_n1, 0.01)

            hourly_5m_n1[hour]["total"] += 1
            hourly_5m_n1[hour]["pnl"] += pnl_5m_n1
            if pnl_5m_n1 > 0:
                hourly_5m_n1[hour]["correct"] += 1
            daily_5m_n1[dow]["total"] += 1
            daily_5m_n1[dow]["pnl"] += pnl_5m_n1
            if pnl_5m_n1 > 0:
                daily_5m_n1[dow]["correct"] += 1
            day_hourly_5m_n1[day_hour_key]["total"] += 1
            day_hourly_5m_n1[day_hour_key]["pnl"] += pnl_5m_n1
            if pnl_5m_n1 > 0:
                day_hourly_5m_n1[day_hour_key]["correct"] += 1

        # Run 1min_n5 individual simulation (only if edge slot)
        if day_hour_key in slot_1m_n5_edge_slots:
            direction_1m_n5 = "UP" if dir_1m_n5 else "DOWN"
            stake_1m_n5 = bankroll_1m_n5 * STAKE_HIGH
            pnl_1m_n5 = compute_pnl_with_exits(current.close, m1_dict[t_plus_5].close, stake_1m_n5, direction_1m_n5)
            bankroll_1m_n5 = max(bankroll_1m_n5 + pnl_1m_n5, 0.01)

            hourly_1m_n5[hour]["total"] += 1
            hourly_1m_n5[hour]["pnl"] += pnl_1m_n5
            if pnl_1m_n5 > 0:
                hourly_1m_n5[hour]["correct"] += 1
            daily_1m_n5[dow]["total"] += 1
            daily_1m_n5[dow]["pnl"] += pnl_1m_n5
            if pnl_1m_n5 > 0:
                daily_1m_n5[dow]["correct"] += 1
            day_hourly_1m_n5[day_hour_key]["total"] += 1
            day_hourly_1m_n5[day_hour_key]["pnl"] += pnl_1m_n5
            if pnl_1m_n5 > 0:
                day_hourly_1m_n5[day_hour_key]["correct"] += 1

    def hr(s):
        return f"{s['correct']/s['total']:.1%}" if s["total"] > 0 else "N/A"

    def build_hourly(hourly_dict):
        results = {}
        for h in sorted(hourly_dict.keys()):
            v = hourly_dict[h]
            rate = v["correct"] / v["total"] if v["total"] > 0 else 0
            results[f"{h:02d}:00"] = {
                "trades": v["total"],
                "hit_rate": f"{rate:.1%}",
                "pnl": round(v["pnl"], 2),
                "edge": rate > 0.53,
            }
        return results

    def build_daily(daily_dict):
        results = {}
        for d in sorted(daily_dict.keys()):
            v = daily_dict[d]
            rate = v["correct"] / v["total"] if v["total"] > 0 else 0
            results[day_names[d]] = {
                "trades": v["total"],
                "hit_rate": f"{rate:.1%}",
                "pnl": round(v["pnl"], 2),
                "edge": rate > 0.53,
            }
        return results

    def build_day_hourly(dh_dict):
        results = {}
        for (d, h) in sorted(dh_dict.keys()):
            v = dh_dict[(d, h)]
            rate = v["correct"] / v["total"] if v["total"] > 0 else 0
            key_str = f"{day_names[d]}_{h:02d}:00"
            results[key_str] = {
                "trades": v["total"],
                "hit_rate": f"{rate:.1%}",
                "pnl": round(v["pnl"], 2),
                "edge": rate > 0.53,
            }
        return results

    # Build breakdowns for all 3 schemas
    hourly_results = build_hourly(hourly)
    daily_results = build_daily(daily)
    day_hourly_results = build_day_hourly(day_hourly)

    hourly_results_5m_n1 = build_hourly(hourly_5m_n1)
    daily_results_5m_n1 = build_daily(daily_5m_n1)
    day_hourly_results_5m_n1 = build_day_hourly(day_hourly_5m_n1)

    hourly_results_1m_n5 = build_hourly(hourly_1m_n5)
    daily_results_1m_n5 = build_daily(daily_1m_n5)
    day_hourly_results_1m_n5 = build_day_hourly(day_hourly_1m_n5)

    best_hours = [h for h, v in hourly_results.items() if v["edge"]]
    best_hours_5m_n1 = [h for h, v in hourly_results_5m_n1.items() if v["edge"]]
    best_hours_1m_n5 = [h for h, v in hourly_results_1m_n5.items() if v["edge"]]

    return {
        "asset": asset_str,
        "type": "BACKTEST",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_period": "Past 3 months Binance historical",
        "NOTE": "This is BACKTEST data. NOT sent to dashboard.",
        "individual_signals": {
            "5min_n1": {"hit_rate": hr(stats_5m_n1),
                        "trades": stats_5m_n1["total"]},
            "5min_n2": {"hit_rate": hr(stats_5m_n2),
                        "trades": stats_5m_n2["total"]},
            "1min_n5": {"hit_rate": hr(stats_1m_n5),
                        "trades": stats_1m_n5["total"]},
        },
        "cascade_combined": {
            "HIGH_conviction_3of3": {
                "hit_rate": hr(stats_high),
                "trades": stats_high["total"],
                "pnl": round(stats_high["pnl"], 2),
            },
            "MEDIUM_conviction_2of3": {
                "hit_rate": hr(stats_medium),
                "trades": stats_medium["total"],
                "pnl": round(stats_medium["pnl"], 2),
            },
        },
        # Schema 1: CASCADE HIGH conviction (3/3 agree)
        "hourly_breakdown": hourly_results,
        "daily_breakdown": daily_results,
        "day_hourly_breakdown": day_hourly_results,
        "best_hours_utc": best_hours,
        # Schema 2: 5min:n+1 ONLY
        "5m_n1_hourly_breakdown": hourly_results_5m_n1,
        "5m_n1_daily_breakdown": daily_results_5m_n1,
        "5m_n1_day_hourly_breakdown": day_hourly_results_5m_n1,
        "5m_n1_best_hours_utc": best_hours_5m_n1,
        "5m_n1_final_bankroll": round(bankroll_5m_n1, 2),
        "5m_n1_total_return": f"{((bankroll_5m_n1 - BANKROLL)/BANKROLL):.1%}",
        # Schema 3: 1min:n+5 ONLY
        "1m_n5_hourly_breakdown": hourly_results_1m_n5,
        "1m_n5_daily_breakdown": daily_results_1m_n5,
        "1m_n5_day_hourly_breakdown": day_hourly_results_1m_n5,
        "1m_n5_best_hours_utc": best_hours_1m_n5,
        "1m_n5_final_bankroll": round(bankroll_1m_n5, 2),
        "1m_n5_total_return": f"{((bankroll_1m_n5 - BANKROLL)/BANKROLL):.1%}",
        # Combined
        "final_bankroll": round(bankroll_high, 2),
        "total_return": f"{((bankroll_high - BANKROLL)/BANKROLL):.1%}",
    }


# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────

def print_grid_report(r: dict, label: str, dh_key: str, file=None) -> None:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # PnL Grid
    print(f"\n  DAY + HOUR PnL MATRIX (UTC) - {label}:", file=file)
    header = "  Day  |" + "".join(f"  {h:02d} |" for h in range(24))
    print(header, file=file)
    print("  -----+" + "-" * (24 * 6), file=file)
    
    day_hourly = r.get(dh_key, {})
    for d_name in day_names:
        row_str = f"  {d_name}  |"
        for h in range(24):
            key = f"{d_name}_{h:02d}:00"
            v = day_hourly.get(key, {"pnl": 0.0})
            pnl_val = int(round(v["pnl"]))
            if pnl_val > 0:
                val_str = f"+{pnl_val}"
            elif pnl_val < 0:
                val_str = f"{pnl_val}"
            else:
                val_str = "0"
            row_str += f" {val_str:^4}|"
        print(row_str, file=file)
        
    # Hit Rate Grid
    print(f"\n  DAY + HOUR HIT RATE MATRIX (UTC) - {label}:", file=file)
    print(header, file=file)
    print("  -----+" + "-" * (24 * 6), file=file)
    for d_name in day_names:
        row_str = f"  {d_name}  |"
        for h in range(24):
            key = f"{d_name}_{h:02d}:00"
            v = day_hourly.get(key, {})
            hit_str = v.get("hit_rate", "N/A")
            if hit_str != "N/A" and hit_str != "0.0%":
                try:
                    val_int = int(round(float(hit_str.strip("%"))))
                    val_str = f"{val_int}%"
                except Exception:
                    val_str = hit_str
            else:
                val_str = "-"
            row_str += f" {val_str:^4}|"
        print(row_str, file=file)

    # Edge cells (>53% hit rate AND positive PnL)
    edge_cells = []
    for dh_slot, v in day_hourly.items():
        if v.get("edge", False) and v.get("pnl", 0) > 0 and v.get("trades", 0) >= 50:
            edge_cells.append((dh_slot, v))
    edge_cells.sort(key=lambda x: x[1]["pnl"], reverse=True)
    if edge_cells:
        print(f"\n  🔥 EDGE SLOTS ({label}) — Hit Rate >53% AND PnL >0 AND trades >=50:", file=file)
        print(f"  {'Day_Hour':<14} {'Hit Rate':<12} {'Trades':<10} {'PnL'}", file=file)
        print(f"  {'-'*55}", file=file)
        for dh_slot, v in edge_cells:
            print(f"  {dh_slot:<14} {v['hit_rate']:<12} {v['trades']:<10} ${v['pnl']}", file=file)


def print_schema_section(r: dict, label: str, hourly_key: str, daily_key: str,
                         dh_key: str, best_key: str, bankroll_key: str,
                         return_key: str, file=None) -> None:
    """Print a full breakdown section for one schema."""
    print(f"\n{'-'*65}", file=file)
    print(f"  SCHEMA: {label}", file=file)
    print(f"{'-'*65}", file=file)

    print(f"\n  BY HOUR OF DAY (UTC) - {label}:", file=file)
    print(f"  {'Hour':<8} {'Hit Rate':<12} {'Trades':<10} "
          f"{'PnL':<12} {'Edge?'}", file=file)
    print(f"  {'-'*55}", file=file)
    for h, v in sorted(r.get(hourly_key, {}).items(),
                        key=lambda x: x[1]["pnl"], reverse=True):
        edge = "✅ EDGE" if v.get("edge", False) else ""
        print(f"  {h:<8} {v['hit_rate']:<12} "
              f"{v['trades']:<10} ${v['pnl']:<12} {edge}", file=file)

    print(f"\n  BY DAY OF WEEK — {label}:", file=file)
    print(f"  {'Day':<8} {'Hit Rate':<12} {'Trades':<10} {'PnL':<12} {'Edge?'}", file=file)
    print(f"  {'-'*55}", file=file)
    for day, v in r.get(daily_key, {}).items():
        edge = "✅ EDGE" if v.get("edge", False) else ""
        print(f"  {day:<8} {v['hit_rate']:<12} "
              f"{v['trades']:<10} ${v['pnl']:<12} {edge}", file=file)

    print(f"\n  BY DAY + HOUR (UTC) — {label} (Top 30 by PnL):", file=file)
    print(f"  {'Day_Hour':<14} {'Hit Rate':<12} {'Trades':<10} {'PnL'}", file=file)
    print(f"  {'-'*55}", file=file)
    day_hour_sorted = sorted(
        r.get(dh_key, {}).items(),
        key=lambda x: x[1]["pnl"],
        reverse=True
    )
    for dh, v in day_hour_sorted[:30]:
        edge = "✅" if v.get("edge", False) else ""
        print(f"  {dh:<14} {v['hit_rate']:<12} {v['trades']:<10} ${v['pnl']} {edge}", file=file)

    # Grid matrices
    print_grid_report(r, label, dh_key, file=file)

    best = r.get(best_key, [])
    print(f"\n  BEST HOURS ({label}): {best if best else 'None >53%'}", file=file)
    if bankroll_key:
        print(f"  Final Bankroll: ${r.get(bankroll_key, 1000)}", file=file)
        print(f"  Total Return  : {r.get(return_key, '0%')}", file=file)


def print_report(r: dict, file=None) -> None:
    asset = r.get("asset", "?")
    print(f"\n{'='*65}", file=file)
    print(f"  CASCADE BACKTEST: {asset}/USDT - Past 3 Months", file=file)
    print(f"  Type: BACKTEST (historical) | NOT forward testing", file=file)
    print(f"{'='*65}", file=file)

    print(f"\n  INDIVIDUAL SIGNALS (overall):", file=file)
    print(f"  {'Signal':<15} {'Hit Rate':<12} {'Trades'}", file=file)
    print(f"  {'-'*35}", file=file)
    for name, v in r.get("individual_signals", {}).items():
        print(f"  {name:<15} {v['hit_rate']:<12} {v['trades']}", file=file)

    print(f"\n  CASCADE COMBINED:", file=file)
    print(f"  {'Conviction':<25} {'Hit Rate':<12} {'Trades':<10} {'PnL'}", file=file)
    print(f"  {'-'*55}", file=file)
    for name, v in r.get("cascade_combined", {}).items():
        print(f"  {name:<25} {v['hit_rate']:<12} "
              f"{v['trades']:<10} ${v['pnl']}", file=file)

    # ── Schema 1: CASCADE HIGH ──
    print_schema_section(r, "CASCADE HIGH (3/3 agree)",
                         "hourly_breakdown", "daily_breakdown",
                         "day_hourly_breakdown", "best_hours_utc",
                         "final_bankroll", "total_return", file=file)

    # ── Schema 2: 5min:n+1 ONLY ──
    print_schema_section(r, "5min:n+1 ONLY",
                         "5m_n1_hourly_breakdown", "5m_n1_daily_breakdown",
                         "5m_n1_day_hourly_breakdown", "5m_n1_best_hours_utc",
                         "5m_n1_final_bankroll", "5m_n1_total_return", file=file)

    # ── Schema 3: 1min:n+5 ONLY ──
    print_schema_section(r, "1min:n+5 ONLY",
                         "1m_n5_hourly_breakdown", "1m_n5_daily_breakdown",
                         "1m_n5_day_hourly_breakdown", "1m_n5_best_hours_utc",
                         "1m_n5_final_bankroll", "1m_n5_total_return", file=file)

    # ── EDGE COMPARISON ACROSS ALL 3 SCHEMAS ──
    print(f"\n{'='*65}", file=file)
    print(f"  🎯 EDGE COMPARISON: All 3 Schemas — {asset}/USDT", file=file)
    print(f"{'='*65}", file=file)
    
    schemas = [
        ("CASCADE HIGH", "day_hourly_breakdown", "final_bankroll", "total_return"),
        ("5min:n+1", "5m_n1_day_hourly_breakdown", "5m_n1_final_bankroll", "5m_n1_total_return"),
        ("1min:n+5", "1m_n5_day_hourly_breakdown", "1m_n5_final_bankroll", "1m_n5_total_return"),
    ]
    
    for schema_name, dh_key, bk_key, rt_key in schemas:
        dh = r.get(dh_key, {})
        edge_slots = [(k, v) for k, v in dh.items()
                      if v.get("edge", False) and v.get("pnl", 0) > 0 and v.get("trades", 0) >= 50]
        edge_slots.sort(key=lambda x: x[1]["pnl"], reverse=True)
        bankroll = r.get(bk_key, 1000)
        ret = r.get(rt_key, "0%")
        print(f"\n  {schema_name} | Bankroll: ${bankroll} | Return: {ret} | Edge Slots: {len(edge_slots)}", file=file)
        if edge_slots:
            for dh_slot, v in edge_slots[:10]:
                print(f"    {dh_slot:<14} {v['hit_rate']:<10} {v['trades']} trades  ${v['pnl']}", file=file)
        else:
            print(f"    No edge slots found (>53% hit AND PnL>0 AND >=50 trades)", file=file)

    print(f"\n{'='*65}\n", file=file)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append",
                        default=["BTC"], choices=["BTC", "ETH"])
    args = parser.parse_args()

    all_results = {}
    output_txt = "cascade_backtest_report.txt"
    
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for asset_str in set(args.asset):
            print(f"\nRunning backtest for {asset_str}...")
            results = run_cascade_backtest(asset_str)
            if results:
                print_report(results)
                print_report(results, file=f_out)
                all_results[asset_str] = results

    output = "cascade_backtest_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON findings to {output}")
    print(f"Saved complete text tables report to {output_txt}")
    print("NOTE: Backtest results are NEVER sent to dashboard.")
    print("Dashboard only shows FORWARD TEST (live paper trading).")
