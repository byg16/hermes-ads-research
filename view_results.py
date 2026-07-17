"""
view_results.py — View all paper trading results from database.

Shows complete history, stats per asset, and best/worst trades.

Usage:
    python view_results.py
    python view_results.py --asset BTC
    python view_results.py --last 50
    python view_results.py --export
"""
from __future__ import annotations

import argparse
import sqlite3
import json
from datetime import datetime

DB_PATH = "paper_trades.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def show_summary(conn: sqlite3.Connection,
                 asset: str = None) -> None:
    where = f"WHERE asset = '{asset}'" if asset else ""
    row = conn.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as hits,
            SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as misses,
            SUM(CASE WHEN side != 'NONE' THEN 1 ELSE 0 END) as bets,
            SUM(CASE WHEN side = 'NONE' THEN 1 ELSE 0 END) as sat_out,
            SUM(pnl_usd) as total_pnl,
            AVG(pnl_usd) as avg_pnl,
            MAX(bankroll) as peak_bankroll,
            MIN(bankroll) as trough_bankroll
        FROM trades
        {where}
        AND correct IS NOT NULL
    """).fetchone()

    if not row or row["total"] == 0:
        print("No trades found yet. Run: python run_paper.py")
        return

    n = row["total"]
    hits = row["hits"] or 0
    hit_rate = hits / n if n > 0 else 0
    peak = row["peak_bankroll"] or 1000
    trough = row["trough_bankroll"] or 1000
    max_dd = ((peak - trough) / peak * 100) if peak > 0 else 0
    total_pnl = row["total_pnl"] or 0
    final_bankroll = 1000 + total_pnl

    title = f"ASSET: {asset}" if asset else "ALL ASSETS"
    print(f"\n{'=' * 60}")
    print(f"  PAPER TRADING RESULTS — {title}")
    print(f"{'=' * 60}")
    print(f"  Total decisions : {n}")
    print(f"  Bets placed     : {row['bets']}")
    print(f"  Sat out         : {row['sat_out']}")
    print(f"  Correct         : {hits}")
    print(f"  Wrong           : {row['misses']}")
    print(f"  Hit rate        : {hit_rate:.1%}")
    print(f"  Total PnL       : ${total_pnl:.2f}")
    print(f"  Final bankroll  : ${final_bankroll:.2f}")
    print(f"  Peak bankroll   : ${peak:.2f}")
    print(f"  Max drawdown    : {max_dd:.1f}%")

    # Edge status
    if n < 50:
        status = f"⏳ NEED MORE DATA ({n}/50 minimum)"
    elif n < 200:
        status = f"⏳ BUILDING SAMPLE ({n}/200)"
    elif hit_rate >= 0.54:
        status = "✅ STRONG EDGE — Consider live deployment"
    elif hit_rate >= 0.53:
        status = "✅ EDGE CONFIRMED — Keep validating"
    elif hit_rate >= 0.51:
        status = "⚠️  BORDERLINE — Need more cycles"
    else:
        status = "❌ NO EDGE — Model needs improvement"

    print(f"\n  Edge Status     : {status}")
    print(f"{'=' * 60}")


def show_recent(conn: sqlite3.Connection,
                n: int = 20, asset: str = None) -> None:
    where = f"AND asset = '{asset}'" if asset else ""
    rows = conn.execute(f"""
        SELECT timestamp, asset, signal, side,
               kelly_fraction, stake_usd, pnl_usd,
               bankroll, hit_rate, reasoning
        FROM trades
        WHERE correct IS NOT NULL {where}
        ORDER BY id DESC
        LIMIT {n}
    """).fetchall()

    if not rows:
        print("No completed trades yet.")
        return

    print(f"\n  LAST {n} TRADES")
    print(f"  {'Time':<10} {'Asset':<6} {'Signal':<8} "
          f"{'Side':<6} {'Stake':<8} {'PnL':<10} {'Bankroll':<12} {'Hit%'}")
    print(f"  {'-' * 70}")
    for r in reversed(rows):
        ts = r["timestamp"][:16].replace("T", " ") if r["timestamp"] else "?"
        pnl = r["pnl_usd"] or 0
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        hr = f"{r['hit_rate']:.1%}" if r["hit_rate"] else "?"
        print(f"  {ts:<10} {r['asset']:<6} {r['signal']:<8} "
              f"{r['side']:<6} ${r['stake_usd']:<7.2f} "
              f"{pnl_str:<10} ${r['bankroll']:<11.2f} {hr}")


def show_by_asset(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT asset,
               COUNT(*) as total,
               SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as hits,
               SUM(pnl_usd) as pnl,
               MAX(bankroll) as peak
        FROM trades
        WHERE correct IS NOT NULL
        GROUP BY asset
    """).fetchall()

    if not rows:
        return

    print(f"\n  BREAKDOWN BY ASSET")
    print(f"  {'Asset':<8} {'Trades':<8} {'Hit Rate':<12} {'PnL':<12} {'Peak'}")
    print(f"  {'-' * 50}")
    for r in rows:
        n = r["total"]
        hr = r["hits"] / n if n > 0 else 0
        print(f"  {r['asset']:<8} {n:<8} {hr:.1%}{'':4} "
              f"${r['pnl']:.2f}{'':4} ${r['peak']:.2f}")


def export_json(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT * FROM trades ORDER BY id
    """).fetchall()
    data = [dict(r) for r in rows]
    filename = f"paper_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Exported {len(data)} trades to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View paper trading results")
    parser.add_argument("--asset", default=None, choices=["BTC", "ETH"])
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    try:
        conn = get_conn()
    except Exception:
        print(f"Database not found at {DB_PATH}")
        print("Run: python run_paper.py --cycles 10 first")
        exit(1)

    show_summary(conn, asset=args.asset)
    show_by_asset(conn)
    show_recent(conn, n=args.last, asset=args.asset)

    if args.export:
        export_json(conn)

    conn.close()
