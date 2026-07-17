"""
run_paper.py — Automated paper trading cycle runner.

Runs the Hermes LLM agent continuously for N cycles,
saves EVERY decision to a SQLite database permanently,
and shows live stats after each cycle.

Usage:
    python run_paper.py --cycles 500 --asset BTC --asset ETH
    python run_paper.py --cycles 100 --asset BTC --interval 5m
    python run_paper.py --forever --asset BTC --asset ETH

Nothing in the original project is touched.
All data saved to paper_trades.db in your project folder.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import time
from datetime import datetime, timezone

from config import settings
from core.logging_config import configure_logging, get_logger
from core.schemas import Asset
from hermes_llm_agent import HermesLLMAgent, fetch_bars

configure_logging()
logger = get_logger("run_paper")

DB_PATH = "paper_trades.db"


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_cycles INTEGER DEFAULT 0,
            assets TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    return conn


def save_trade(conn: sqlite3.Connection, result: dict,
               interval: str = "5m") -> None:
    """Save one trade decision to the database."""
    decision = result.get("llm_decision", {})
    outcome = result.get("simulated_outcome", {})
    perf = result.get("performance", {})

    conn.execute("""
        INSERT INTO trades (
            timestamp, asset, signal, p_up, p_down,
            side, kelly_fraction, stake_usd, bankroll,
            correct, pnl_usd, hit_rate, sharpe,
            reasoning, model, interval
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("timestamp"),
        result.get("asset"),
        result["prediction"].get("signal"),
        result["prediction"].get("p_up"),
        result["prediction"].get("p_down"),
        decision.get("side", "NONE"),
        decision.get("kelly_fraction", 0.0),
        result.get("stake_usd", 0.0),
        result.get("bankroll", settings.bankroll_usd),
        1 if outcome.get("correct") else 0 if outcome else None,
        outcome.get("pnl_usd"),
        perf.get("hit_rate"),
        perf.get("sharpe"),
        decision.get("reasoning", ""),
        result["prediction"].get("model"),
        interval,
    ))
    conn.commit()


# ─────────────────────────────────────────────
# STATS DISPLAY
# ─────────────────────────────────────────────

def show_live_stats(conn: sqlite3.Connection,
                    asset: str, cycle: int, total: int) -> None:
    """Print live stats from database after each cycle."""
    rows = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as hits,
            SUM(pnl_usd) as total_pnl,
            AVG(sharpe) as avg_sharpe,
            MAX(bankroll) as peak,
            MIN(bankroll) as trough,
            bankroll as current_bankroll
        FROM trades
        WHERE asset = ? AND correct IS NOT NULL
    """, (asset,)).fetchone()

    if not rows or rows[0] == 0:
        return

    total_trades = rows[0] or 0
    hits = rows[1] or 0
    total_pnl = rows[2] or 0
    avg_sharpe = rows[3] or 0
    peak = rows[4] or settings.bankroll_usd
    trough = rows[5] or settings.bankroll_usd
    current = rows[6] or settings.bankroll_usd
    hit_rate = hits / total_trades if total_trades > 0 else 0
    max_dd = ((peak - trough) / peak * 100) if peak > 0 else 0

    print(f"\n  ── LIVE STATS: {asset} "
          f"(Cycle {cycle}/{total if total > 0 else '∞'}) ──")
    print(f"  Total trades  : {total_trades}")
    print(f"  Hit rate      : {hit_rate:.1%}")
    print(f"  Total PnL     : ${total_pnl:.2f}")
    print(f"  Bankroll      : ${current:.2f}")
    print(f"  Avg Sharpe    : {avg_sharpe:.3f}")
    print(f"  Max Drawdown  : {max_dd:.1f}%")

    # Progress bar
    if hit_rate >= 0.53:
        status = "✅ EDGE CONFIRMED"
    elif hit_rate >= 0.51:
        status = "⚠️  BORDERLINE"
    else:
        status = "❌ NO EDGE YET"
    print(f"  Edge Status   : {status}")


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_paper_cycles(assets: list, n_cycles: int = 500,
                     interval: str = "5m", forever: bool = False,
                     sleep_seconds: int = 10) -> None:
    """Run paper trading cycles and save everything to SQLite."""
    conn = init_db()
    agent = HermesLLMAgent()

    # Record session
    session_id = conn.execute("""
        INSERT INTO sessions (started_at, assets, notes)
        VALUES (?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        ",".join(a.value for a in assets),
        f"paper trading {n_cycles} cycles on {interval} bars"
    )).lastrowid
    conn.commit()

    print(f"\n{'=' * 60}")
    print(f"  PAPER TRADING RUNNER")
    print(f"  Assets  : {[a.value for a in assets]}")
    print(f"  Cycles  : {'∞ (forever)' if forever else n_cycles}")
    print(f"  Interval: {interval} bars")
    print(f"  Database: {DB_PATH}")
    print(f"{'=' * 60}\n")

    cycle = 0
    try:
        while forever or cycle < n_cycles:
            cycle += 1

            for asset in assets:
                try:
                    # Fetch live bars
                    series = fetch_bars(asset, n_bars=1000,
                                        interval=interval)

                    # Get LLM decision
                    result = agent.decide(asset, series)

                    # Save to database
                    save_trade(conn, result, interval=interval)

                    # Print cycle summary
                    decision = result.get("llm_decision", {})
                    outcome = result.get("simulated_outcome", {})

                    print(f"\n  Cycle {cycle} | {asset.value} | "
                          f"{datetime.now().strftime('%H:%M:%S')}")
                    print(f"  Signal : {result['prediction']['signal']} "
                          f"P(UP)={result['prediction']['p_up']:.1%}")
                    print(f"  Decision: {decision.get('side', 'NONE')} "
                          f"Kelly={decision.get('kelly_fraction', 0):.1%} "
                          f"Stake=${result.get('stake_usd', 0)}")
                    if outcome:
                        icon = "✓" if outcome.get("correct") else "✗"
                        print(f"  Outcome : {icon} "
                              f"PnL=${outcome.get('pnl_usd', 0):.2f} "
                              f"Bankroll=${outcome.get('new_bankroll', 0):.2f}")

                    # Show live stats every 10 cycles
                    if cycle % 10 == 0:
                        show_live_stats(conn, asset.value,
                                        cycle, n_cycles)

                except Exception as exc:
                    logger.error("cycle_failed",
                                 cycle=cycle, asset=asset, error=str(exc))
                    continue

            # Small delay between cycles
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        # Update session record
        conn.execute("""
            UPDATE sessions SET ended_at = ?, total_cycles = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), cycle, session_id))
        conn.commit()

        # Final stats
        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULTS")
        print(f"{'=' * 60}")
        for asset in assets:
            show_live_stats(conn, asset.value, cycle, n_cycles)
        print(f"\n  All data saved to: {DB_PATH}")
        print(f"  Run: python view_results.py to see full history")
        print(f"{'=' * 60}\n")
        conn.close()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated paper trading cycle runner")
    parser.add_argument("--asset", action="append",
                        default=["BTC"], choices=["BTC", "ETH"])
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--interval", default="5m",
                        choices=["1m", "5m", "15m"])
    parser.add_argument("--forever", action="store_true",
                        help="Run continuously until stopped")
    parser.add_argument("--sleep", type=int, default=10,
                        help="Seconds between cycles")
    args = parser.parse_args()

    assets = [Asset(a) for a in set(args.asset)]
    run_paper_cycles(
        assets=assets,
        n_cycles=args.cycles,
        interval=args.interval,
        forever=args.forever,
        sleep_seconds=args.sleep,
    )
