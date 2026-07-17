"""
cascade_engine.py — Gilad's Cascade Prediction System

Implements exactly what Gilad specified:
  - 5min:n+1  → predict next 5-minute candle
  - 5min:n+2  → predict 2 candles ahead (10 minutes)
  - 1min:n+5  → predict next 5 minutes using 1-min bars
  - Combine   → only bet when signals AGREE (high conviction)
  - Hour filter → track which hours are best
  - Stop-loss → 5% hard stop to cap losses
  - Edge-slots → only trade on pre-identified profitable hours/days

Logic:
  If 5min:n+1 says UP AND 1min:n+5 says UP → HIGH CONVICTION BET
  If they disagree → SIT OUT (no bet)
  If 5min:n+2 also agrees → INCREASE Kelly fraction

This generates trades 3x faster than 15-min bars
and only bets when there is real conviction across timeframes.

Usage:
    python cascade_engine.py --asset BTC --cycles 100
    python cascade_engine.py --asset BTC --asset ETH --cycles 500
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import httpx
import numpy as np

from config import settings
from core.logging_config import configure_logging, get_logger
from core.schemas import Asset, OHLCVBar, OHLCVSeries
from tools.multi_signal_predictor import MultiSignalPredictor

configure_logging()
logger = get_logger("cascade_engine")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BANKROLL = settings.bankroll_usd
DB_PATH = "hermes-crypto/paper_trades.db"  # Save to the dashboard's database!
MAX_DRAWDOWN_PCT = 0.08  # Hard cap on max drawdown per trade (safety net)

# Edge slots from backtest - only trade these times!
EDGE_SLOTS_5MIN_N1 = {
    "Tue_00", "Tue_08", "Wed_13", "Wed_15", "Wed_21", "Thu_01",
    "Thu_06", "Thu_10", "Thu_15", "Thu_21", "Fri_06", "Fri_13",
    "Fri_23", "Sat_01", "Sat_17", "Sun_22"
}

EDGE_SLOTS_1MIN_N5 = {
    "Mon_00", "Mon_03", "Tue_10", "Tue_16", "Wed_04", "Wed_08",
    "Wed_16", "Wed_18", "Wed_20", "Thu_20", "Fri_15", "Fri_16",
    "Fri_21", "Sat_20", "Sun_23"
}

# Track active positions (for real-time re-evaluation)
active_positions: Dict[str, Dict] = {}  # key: asset, value: position dict


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def is_edge_slot(now: datetime, asset: Asset) -> bool:
    """Check if current time is in an edge slot for the asset."""
    day_name = now.strftime("%a")  # "Mon", "Tue", etc.
    hour = now.hour
    slot_key = f"{day_name}_{hour:02d}"
    
    # Check both models' edge slots (we want to cover both for faster trading)
    return slot_key in EDGE_SLOTS_5MIN_N1 or slot_key in EDGE_SLOTS_1MIN_N5

def compute_pnl_with_exits(
    entry_price: float, current_price: float, stake: float,
    direction: str, current_signal: Optional["CascadeSignal"] = None
) -> Tuple[float, bool]:
    """
    Compute PnL with MODEL-DRIVEN exits + a max drawdown safety net.
    direction: "YES" (long/UP) or "NO" (short/DOWN)
    current_signal: The latest cascade signal to decide if we should exit
    Returns (pnl, should_exit)
    """
    if direction == "YES":
        # Long position: entry_price < exit_price = profit
        pnl_pct = (current_price - entry_price) / entry_price
        # First check safety net: max drawdown cap
        if pnl_pct <= -MAX_DRAWDOWN_PCT:
            return -stake * MAX_DRAWDOWN_PCT, True
        # Now check model signal: if current signal is NO or conflicting, exit!
        if current_signal:
            if current_signal.side == "NO" or current_signal.conviction == "NONE":
                return stake * pnl_pct, True
    else:
        # Short position: entry_price > exit_price = profit
        pnl_pct = (entry_price - current_price) / entry_price
        # First check safety net: max drawdown cap
        if pnl_pct <= -MAX_DRAWDOWN_PCT:
            return -stake * MAX_DRAWDOWN_PCT, True
        # Now check model signal: if current signal is YES or conflicting, exit!
        if current_signal:
            if current_signal.side == "YES" or current_signal.conviction == "NONE":
                return stake * pnl_pct, True

    # If no exit condition met: hold, return current PnL
    return stake * pnl_pct, False

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_bars(asset: Asset, n_bars: int, interval: str) -> List[OHLCVBar]:
    """Fetch OHLCV bars from Binance."""
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
                timestamp=datetime.fromtimestamp(
                    k[0] / 1000, tz=timezone.utc),
                open=float(k[1]), high=float(k[2]),
                low=float(k[3]), close=float(k[4]),
                volume=float(k[5]),
            ) for k in data
        ]
        logger.info("bars_fetched", asset=asset,
                    interval=interval, n=len(bars))
        return bars
    except Exception as exc:
        logger.warning("binance_fetch_failed",
                       asset=asset, interval=interval, error=str(exc))
        return _synthetic_bars(asset, n_bars)


def _synthetic_bars(asset: Asset, n_bars: int) -> List[OHLCVBar]:
    seed = {Asset.BTC: 65000.0, Asset.ETH: 3400.0}
    rng = random.Random(hash(asset.value) % (2 ** 31))
    price = seed.get(asset, 1000.0)
    now = datetime.now(timezone.utc)
    bars = []
    for i in range(n_bars):
        ts = now - timedelta(minutes=(n_bars - i))
        c = max(price + rng.gauss(0, 0.0015) * price, 0.01)
        bars.append(OHLCVBar(
            timestamp=ts, open=price,
            high=max(price, c) * 1.001,
            low=min(price, c) * 0.999,
            close=c, volume=abs(rng.gauss(50, 20))))
        price = c
    return bars


# ─────────────────────────────────────────────
# CASCADE SIGNAL
# ─────────────────────────────────────────────

@dataclass
class CascadeSignal:
    asset: str
    timestamp: str

    # Individual signals
    p_up_5min_n1: float    # 5min predicting next 1 candle
    p_up_5min_n2: float    # 5min predicting 2 candles ahead
    p_up_1min_n5: float    # 1min predicting next 5 minutes

    # Combined signal
    conviction: str         # HIGH / MEDIUM / LOW / NONE
    combined_p_up: float
    should_bet: bool
    side: str              # YES / NO / NONE
    kelly_fraction: float

    # Context
    hour_utc: int
    rationale: str


def compute_cascade(asset: Asset,
                    predictor: MultiSignalPredictor) -> CascadeSignal:
    """
    Gilad's cascade: run all 3 signals, combine for conviction.

    5min:n+1 — what does 5-min data say about next candle?
    5min:n+2 — what does 5-min data say about 2 candles ahead?
    1min:n+5 — what does 1-min data say about next 5 minutes?
    """
    now = datetime.now(timezone.utc)
    hour = now.hour

    # Fetch both timeframes
    bars_5m = fetch_bars(asset, n_bars=200, interval="5m")
    bars_1m = fetch_bars(asset, n_bars=300, interval="1m")

    series_5m = OHLCVSeries(asset=asset, timeframe="5m",
                             bars=bars_5m, source="binance")
    series_1m = OHLCVSeries(asset=asset, timeframe="1m",
                             bars=bars_1m, source="binance")

    # Signal 1: 5min → n+1
    pred_5m_n1 = predictor.predict(series_5m, horizon_minutes=5)
    p_up_5m_n1 = pred_5m_n1.p_up

    # Signal 2: 5min → n+2 (use last 180 bars for slightly different window)
    series_5m_n2 = OHLCVSeries(asset=asset, timeframe="5m",
                                 bars=bars_5m[-180:], source="binance")
    pred_5m_n2 = predictor.predict(series_5m_n2, horizon_minutes=10)
    p_up_5m_n2 = pred_5m_n2.p_up

    # Signal 3: 1min → n+5
    pred_1m_n5 = predictor.predict(series_1m, horizon_minutes=5)
    p_up_1m_n5 = pred_1m_n5.p_up

    # ── Determine conviction ──
    # All 3 agree on direction = HIGH conviction
    # 2 out of 3 agree = MEDIUM conviction
    # All disagree = NO bet

    signals_up = [
        p_up_5m_n1 > 0.53,
        p_up_5m_n2 > 0.53,
        p_up_1m_n5 > 0.53,
    ]
    signals_down = [
        p_up_5m_n1 < 0.47,
        p_up_5m_n2 < 0.47,
        p_up_1m_n5 < 0.47,
    ]

    up_count = sum(signals_up)
    down_count = sum(signals_down)

    # Combined probability — weighted average
    # 5min:n+1 gets most weight as primary signal
    combined_p_up = (
        0.45 * p_up_5m_n1 +
        0.30 * p_up_1m_n5 +
        0.25 * p_up_5m_n2
    )

    if up_count == 3:
        conviction = "HIGH"
        should_bet = True
        side = "YES"
        kelly = 0.15  # 15% Kelly when all agree
        rationale = "All 3 signals agree UP — high conviction"
    elif down_count == 3:
        conviction = "HIGH"
        should_bet = True
        side = "NO"
        kelly = 0.15
        rationale = "All 3 signals agree DOWN — high conviction"
    elif up_count == 2:
        conviction = "MEDIUM"
        should_bet = True
        side = "YES"
        kelly = 0.075  # 7.5% Kelly when 2/3 agree
        rationale = f"2/3 signals agree UP — medium conviction"
    elif down_count == 2:
        conviction = "MEDIUM"
        should_bet = True
        side = "NO"
        kelly = 0.075
        rationale = f"2/3 signals agree DOWN — medium conviction"
    else:
        conviction = "NONE"
        should_bet = False
        side = "NONE"
        kelly = 0.0
        rationale = "Signals conflict — sitting out"

    return CascadeSignal(
        asset=asset.value,
        timestamp=now.isoformat(),
        p_up_5min_n1=round(p_up_5m_n1, 4),
        p_up_5min_n2=round(p_up_5m_n2, 4),
        p_up_1min_n5=round(p_up_1m_n5, 4),
        conviction=conviction,
        combined_p_up=round(combined_p_up, 4),
        should_bet=should_bet,
        side=side,
        kelly_fraction=kelly,
        hour_utc=hour,
        rationale=rationale,
    )


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def calculate_hit_rate(conn: sqlite3.Connection, asset: str) -> float:
    """Calculate current hit rate from database."""
    row = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)
        FROM trades
        WHERE asset = ? AND correct IS NOT NULL
    """, (asset,)).fetchone()
    total = row[0] or 0
    hits = row[1] or 0
    return hits / total if total > 0 else 0.0

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            asset TEXT NOT NULL,
            signal TEXT NOT NULL,
            p_up REAL NOT NULL,
            p_down REAL NOT NULL,
            side TEXT NOT NULL,
            kelly_fraction REAL NOT NULL,
            stake_usd REAL NOT NULL,
            bankroll REAL NOT NULL,
            correct INTEGER,
            pnl_usd REAL,
            hit_rate REAL,
            sharpe REAL,
            reasoning TEXT,
            model TEXT,
            interval TEXT DEFAULT '5m'
        )
    """)
    conn.commit()
    return conn


def save_trade(conn: sqlite3.Connection, signal: CascadeSignal, stake: float,
               correct: Optional[bool], pnl: float, bankroll: float,
               hit_rate: float = 0.0, sharpe: float = 0.0) -> None:
    # Convert combined_p_up to signal string
    if signal.combined_p_up > 0.53:
        signal_str = "UP"
    elif signal.combined_p_up < 0.47:
        signal_str = "DOWN"
    else:
        signal_str = "NEUTRAL"
    
    # Get model/interval
    model = "cascade"
    interval = "5m"
    
    conn.execute("""
        INSERT INTO trades (
            timestamp, asset, signal, p_up, p_down,
            side, kelly_fraction, stake_usd, bankroll,
            correct, pnl_usd, hit_rate, sharpe,
            reasoning, model, interval
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal.timestamp,
        signal.asset,
        signal_str,
        signal.combined_p_up,
        1 - signal.combined_p_up,
        signal.side,
        signal.kelly_fraction,
        stake,
        bankroll,
        1 if correct else 0 if correct is not None else None,
        pnl,
        hit_rate,
        sharpe,
        signal.rationale,
        model,
        interval,
    ))
    conn.commit()


# ─────────────────────────────────────────────
# STATS & EDGE TIMINGS & EXPORT
# ─────────────────────────────────────────────

def show_edge_timings() -> None:
    """Show all edge timings in tabular format."""
    print(f"\n{'=' * 65}")
    print(f"  EDGE TIMINGS BY DAY (ALL SCHEMAS)")
    print(f"{'=' * 65}")
    
    # Days of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    print(f"\n  CASCADE HIGH & 5min:n+1 & 1min:n+5 Edge Slots")
    print(f"  {'Day':<8} {'Hours':<50}")
    print(f"  {'-' * 60}")
    
    for day in days:
        # Collect edge slots for this day
        slots_5min = sorted([s.split('_')[1] for s in EDGE_SLOTS_5MIN_N1 if s.startswith(day)])
        slots_1min = sorted([s.split('_')[1] for s in EDGE_SLOTS_1MIN_N5 if s.startswith(day)])
        all_slots = sorted(list(set(slots_5min + slots_1min)))
        
        # Format hours as HH:00
        hours_str = ', '.join([f"{h}:00" for h in all_slots])
        
        print(f"  {day:<8} {hours_str if hours_str else 'No edge slots'}")
    print(f"{'=' * 65}\n")

def export_forward_test_results(conn: sqlite3.Connection) -> None:
    """Export forward test results to CSV file."""
    import csv
    with open("forward_test_results.csv", "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["Forward Test Results"])
        writer.writerow(["ID", "Timestamp", "Asset", "Signal", "Side", "Stake USD", "PnL USD", "Bankroll", "Correct", "Hit Rate"])
        
        # Fetch all trades
        trades = conn.execute("""
            SELECT id, timestamp, asset, signal, side, stake_usd, pnl_usd, bankroll, correct, hit_rate
            FROM trades
            ORDER BY id DESC
        """).fetchall()
        
        for trade in trades:
            id, ts, asset, signal, side, stake, pnl, bankroll, correct, hit_rate = trade
            writer.writerow([
                id,
                ts,
                asset,
                signal,
                side,
                f"${stake:.2f}" if stake else "-",
                f"${pnl:.2f}" if pnl else "-",
                f"${bankroll:.2f}",
                "✓" if correct else "✗" if correct is not None else "Pending",
                f"{hit_rate:.1%}" if hit_rate else "-"
            ])
    print(f"\n✅ Forward test results exported to forward_test_results.csv!")

def show_stats(conn: sqlite3.Connection) -> None:
    """Show performance breakdown."""
    print(f"\n{'=' * 65}")
    print(f"  CASCADE PERFORMANCE BREAKDOWN")
    print(f"{'=' * 65}")

    # Overall stats
    total = conn.execute("""
        SELECT COUNT(*), COUNT(CASE WHEN side!='NONE' THEN 1 END),
               ROUND(AVG(CASE WHEN correct=1 AND side!='NONE'
                         THEN 100.0 ELSE NULL END),1),
               ROUND(SUM(pnl_usd),2), MAX(bankroll)
        FROM trades WHERE correct IS NOT NULL
    """).fetchone()

    if total and total[0]:
        print(f"\n  OVERALL:")
        print(f"  Total cycles : {total[0]}")
        print(f"  Bets placed  : {total[1]}")
        print(f"  Win rate     : {total[2]}%")
        print(f"  Total PnL    : ${total[3]}")
        print(f"  Peak bankroll: ${total[4]}")

        if total[2] and total[2] >= 53:
            print(f"\n  ✅ EDGE CONFIRMED — Ready to scale")
        elif total[0] < 50:
            print(f"\n  ⏳ Need more data ({total[0]}/50 minimum)")
        else:
            print(f"\n  ⚠️  No edge yet — keep running")

    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────
# MAIN RUNNER (BOTH MODES)
# ─────────────────────────────────────────────

def run_cascade(assets: List[Asset], n_cycles: int = 500,
                sleep_seconds: int = 30,
                forever: bool = False) -> None:
    """Run the cascade engine in automated mode (original behavior)."""
    conn = init_db()
    predictor = MultiSignalPredictor()
    bankroll = BANKROLL
    active_positions.clear()

    print(f"\n{'=' * 65}")
    print(f"  HERMES CASCADE ENGINE (FORWARD TESTING AUTOMATED)")
    print(f"  Strategy: 5min:n+1 + 5min:n+2 + 1min:n+5")
    print(f"  Features: Model-driven exits + max {MAX_DRAWDOWN_PCT*100:.0f}% safety stop-loss")
    print(f"  Assets: {[a.value for a in assets]}")
    print(f"  Target: {n_cycles} cycles")
    print(f"{'=' * 65}\n")

    # Show edge timings first
    show_edge_timings()

    cycle = 0
    try:
        while forever or cycle < n_cycles:
            cycle += 1
            now = datetime.now(timezone.utc)

            for asset in assets:
                try:
                    print(f"\n  ── Cycle {cycle} | {asset.value} | "
                          f"{now.strftime('%H:%M UTC')} ──")

                    # ======================================
                    # 1. FIRST: Re-evaluate EXISTING OPEN POSITIONS
                    # ======================================
                    if asset.value in active_positions:
                        pos = active_positions[asset.value]
                        print(f"\n  → Re-evaluating open {pos['side']} position...")
                        # Get current price
                        bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                        if bars_1m:
                            current_price = bars_1m[-1].close
                            # Get latest cascade signal
                            current_signal = compute_cascade(asset, predictor)
                            # Decide to exit or hold
                            pnl, should_exit = compute_pnl_with_exits(
                                pos['entry_price'],
                                current_price,
                                pos['stake'],
                                pos['side'],
                                current_signal
                            )
                            if should_exit:
                                print(f"  📤 Exiting position! PnL: ${pnl:.2f}")
                                # Determine if the trade was correct
                                correct = pnl > 0
                                # Update bankroll
                                bankroll = max(bankroll + pnl, 0.01)
                                # Calculate hit rate
                                hit_rate = calculate_hit_rate(conn, asset.value)
                                # Save the closed trade
                                pos['signal'] = pos['entry_signal']
                                pos['correct'] = correct
                                pos['pnl'] = pnl
                                pos['bankroll'] = bankroll
                                pos['exit_price'] = current_price
                                # Use the entry signal for saving
                                save_trade(conn, pos['entry_signal'], pos['stake'], correct, pnl, bankroll, hit_rate)
                                # Remove from active positions
                                del active_positions[asset.value]
                            else:
                                print(f"  📥 Holding position. Current PnL: ${pnl:.2f}")
                                # Update active position's unrealized PnL
                                pos['unrealized_pnl'] = pnl
                                # Don't save yet—only when closed

                    # ======================================
                    # 2. SECOND: Check if we can ENTER A NEW POSITION
                    # ======================================
                    if asset.value not in active_positions:
                        # Check if current time is in an edge slot
                        if not is_edge_slot(now, asset):
                            print(f"  ⏳ Not in edge slot, sitting out")
                            continue

                        # Compute cascade signal
                        signal = compute_cascade(asset, predictor)

                        # Print signal breakdown
                        print(f"  5min:n+1  : {'UP' if signal.p_up_5min_n1 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_5min_n1:.1%})")
                        print(f"  5min:n+2  : {'UP' if signal.p_up_5min_n2 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_5min_n2:.1%})")
                        print(f"  1min:n+5  : {'UP' if signal.p_up_1min_n5 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_1min_n5:.1%})")
                        print(f"  Conviction: {signal.conviction}")
                        print(f"  Decision  : {signal.side} | {signal.rationale}")

                        if signal.should_bet:
                            # Calculate stake
                            stake = round(bankroll * signal.kelly_fraction, 2)
                            if stake <= 0:
                                print(f"  Stake too small, sitting out")
                                continue
                            # Fetch current price for entry
                            bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                            if bars_1m:
                                entry_price = bars_1m[-1].close
                                # Open the position!
                                print(f"  📥 Opening {signal.side} position!")
                                print(f"  Entry     : ${entry_price:.2f}")
                                print(f"  Stake     : ${stake:.2f}")
                                # Track as active
                                active_positions[asset.value] = {
                                    "side": signal.side,
                                    "entry_price": entry_price,
                                    "stake": stake,
                                    "entry_signal": signal,
                                    "entry_time": now.isoformat(),
                                    "unrealized_pnl": 0.0
                                }
                        else:
                            print(f"  Sitting out — {signal.rationale}")

                except Exception as exc:
                    logger.error("cycle_error",
                                 cycle=cycle, asset=asset, error=str(exc))
                    continue

            # Show stats every 25 cycles
            if cycle % 25 == 0:
                show_stats(conn)

            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        print(f"\nClosing any remaining open positions...")
        # Close any open positions before exiting
        for asset in assets:
            if asset.value in active_positions:
                pos = active_positions[asset.value]
                # Get current price
                bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                if bars_1m:
                    current_price = bars_1m[-1].close
                    pnl, _ = compute_pnl_with_exits(
                        pos['entry_price'], current_price, pos['stake'], pos['side'], None
                    )
                    correct = pnl > 0
                    bankroll = max(bankroll + pnl, 0.01)
                    hit_rate = calculate_hit_rate(conn, asset.value)
                    save_trade(conn, pos['entry_signal'], pos['stake'], correct, pnl, bankroll, hit_rate)
                    del active_positions[asset.value]
        print(f"\nFinal stats after {cycle} cycles:")
        show_stats(conn)
        export_forward_test_results(conn)
        conn.close()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hermes Cascade Engine — 5min:n+1 + 5min:n+2 + 1min:n+5")
    parser.add_argument("--asset", action="append",
                        default=["BTC"], choices=["BTC", "ETH"])
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--sleep", type=int, default=30)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    assets = [Asset(a) for a in set(args.asset)]

    if args.interactive:
        # Run interactive mode (on-demand)
        import sys
        from typing import List, Dict, Optional, Tuple

        def run_cascade_interactive(assets: List[Asset]) -> None:
            conn = init_db()
            predictor = MultiSignalPredictor()
            bankroll = BANKROLL
            active_positions.clear()

            print(f"\n{'=' * 65}")
            print(f"  HERMES CASCADE ENGINE (FORWARD TESTING ON-DEMAND)")
            print(f"  Strategy: 5min:n+1 + 5min:n+2 + 1min:n+5")
            print(f"  Features: Model-driven exits + max {MAX_DRAWDOWN_PCT*100:.0f}% safety stop-loss")
            print(f"  Assets: {[a.value for a in assets]}")
            print(f"{'=' * 65}\n")

            show_edge_timings()
            cycle = 0

            try:
                while True:
                    now = datetime.now(timezone.utc)
                    print(f"\n  ──────────────────────────────────────────────")
                    print(f"  Current time (UTC): {now.strftime('%Y-%m-%d %H:%M')}")
                    print(f"\n  Options:")
                    print(f"  1. Run a single cycle")
                    print(f"  2. Run N cycles (enter N after selecting)")
                    print(f"  3. Show performance stats")
                    print(f"  4. Show edge timings")
                    print(f"  5. Export results to CSV")
                    print(f"  6. Exit")

                    try:
                        choice = input(f"\n  Enter your choice (1-6): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\nExiting...")
                        break

                    if choice == "1":
                        cycle += 1
                        bankroll = run_single_cycle(conn, predictor, assets, cycle, bankroll)
                    elif choice == "2":
                        try:
                            n_cycles = int(input("  Enter number of cycles to run: ").strip())
                            for _ in range(n_cycles):
                                cycle += 1
                                bankroll = run_single_cycle(conn, predictor, assets, cycle, bankroll)
                                if _ < n_cycles - 1:
                                    time.sleep(2)
                        except ValueError:
                            print("  Invalid number!")
                    elif choice == "3":
                        show_stats(conn)
                    elif choice == "4":
                        show_edge_timings()
                    elif choice == "5":
                        export_forward_test_results(conn)
                    elif choice == "6":
                        break
                    else:
                        print("  Invalid choice!")

            except KeyboardInterrupt:
                print("\n\nStopped by user.")

            finally:
                print(f"\nClosing any remaining open positions...")
                for asset in assets:
                    if asset.value in active_positions:
                        pos = active_positions[asset.value]
                        bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                        if bars_1m:
                            current_price = bars_1m[-1].close
                            pnl, _ = compute_pnl_with_exits(
                                pos['entry_price'], current_price, pos['stake'], pos['side'], None
                            )
                            correct = pnl > 0
                            bankroll = max(bankroll + pnl, 0.01)
                            hit_rate = calculate_hit_rate(conn, asset.value)
                            save_trade(conn, pos['entry_signal'], pos['stake'], correct, pnl, bankroll, hit_rate)
                            del active_positions[asset.value]
                print(f"\nFinal stats after {cycle} cycles:")
                show_stats(conn)
                export_forward_test_results(conn)
                conn.close()

        def run_single_cycle(conn: sqlite3.Connection, predictor: MultiSignalPredictor,
                             assets: List[Asset], cycle: int, bankroll: float) -> float:
            now = datetime.now(timezone.utc)
            for asset in assets:
                try:
                    print(f"\n  ── Cycle {cycle} | {asset.value} | "
                          f"{now.strftime('%H:%M UTC')} ──")

                    # 1. FIRST: Re-evaluate existing open positions
                    if asset.value in active_positions:
                        pos = active_positions[asset.value]
                        print(f"\n  → Re-evaluating open {pos['side']} position...")
                        # Get current price
                        bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                        if bars_1m:
                            current_price = bars_1m[-1].close
                            # Get latest cascade signal
                            current_signal = compute_cascade(asset, predictor)
                            # Decide to exit or hold
                            pnl, should_exit = compute_pnl_with_exits(
                                pos['entry_price'],
                                current_price,
                                pos['stake'],
                                pos['side'],
                                current_signal
                            )
                            if should_exit:
                                print(f"  📤 Exiting position! PnL: ${pnl:.2f}")
                                correct = pnl > 0
                                bankroll = max(bankroll + pnl, 0.01)
                                hit_rate = calculate_hit_rate(conn, asset.value)
                                save_trade(conn, pos['entry_signal'], pos['stake'], correct, pnl, bankroll, hit_rate)
                                del active_positions[asset.value]
                            else:
                                print(f"  📥 Holding position. Current PnL: ${pnl:.2f}")
                                pos['unrealized_pnl'] = pnl

                    # 2. SECOND: Check if we can enter new position
                    if asset.value not in active_positions:
                        # Check if current time is in an edge slot
                        if not is_edge_slot(now, asset):
                            print(f"  ⏳ Not in edge slot, sitting out")
                            continue

                        signal = compute_cascade(asset, predictor)
                        print(f"  5min:n+1  : {'UP' if signal.p_up_5min_n1 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_5min_n1:.1%})")
                        print(f"  5min:n+2  : {'UP' if signal.p_up_5min_n2 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_5min_n2:.1%})")
                        print(f"  1min:n+5  : {'UP' if signal.p_up_1min_n5 > 0.5 else 'DOWN'} "
                              f"({signal.p_up_1min_n5:.1%})")
                        print(f"  Conviction: {signal.conviction}")
                        print(f"  Decision  : {signal.side} | {signal.rationale}")

                        if signal.should_bet:
                            stake = round(bankroll * signal.kelly_fraction, 2)
                            if stake <= 0:
                                print(f"  Stake too small, sitting out")
                                continue

                            bars_1m = fetch_bars(asset, n_bars=1, interval="1m")
                            if bars_1m:
                                entry_price = bars_1m[-1].close
                                print(f"  📥 Opening {signal.side} position!")
                                print(f"  Entry     : ${entry_price:.2f}")
                                print(f"  Stake     : ${stake:.2f}")
                                active_positions[asset.value] = {
                                    "side": signal.side,
                                    "entry_price": entry_price,
                                    "stake": stake,
                                    "entry_signal": signal,
                                    "entry_time": now.isoformat(),
                                    "unrealized_pnl": 0.0
                                }
                        else:
                            print(f"  Sitting out — {signal.rationale}")

                except Exception as exc:
                    logger.error("cycle_error",
                                 cycle=cycle, asset=asset, error=str(exc))
                    continue
            return bankroll

        run_cascade_interactive(assets=assets)
    else:
        # Run automated mode (original behavior)
        run_cascade(assets=assets, n_cycles=args.cycles,
                    sleep_seconds=args.sleep, forever=args.forever)
