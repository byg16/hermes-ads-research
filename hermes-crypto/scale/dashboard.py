"""Minimal read-only FastAPI dashboard for user visibility (scaling idea 6.3).

Exposes the latest decisions, ad briefs, and aggregate stats. Intended to be
polled by a frontend or piped into a Slack bot. Run with:

    uvicorn scale.dashboard:app --reload --port 8000
"""
from __future__ import annotations

import sqlite3
from typing import List, Dict, Any

from fastapi import FastAPI

from core.orchestrator import LATEST_DECISIONS, LATEST_AD_BRIEFS, Orchestrator

app = FastAPI(title="Hermes Ads Research Dashboard", version="0.1.0")
_orchestrator = Orchestrator()
CASCADE_DB_PATH = "paper_trades.db"  # Since we're in hermes-crypto directory


def get_cascade_db_connection():
    conn = sqlite3.connect(CASCADE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def predictions():
    return [d.prediction.model_dump(mode="json") for d in LATEST_DECISIONS]


@app.get("/stakes")
def stakes():
    return [
        {
            "asset": d.market.asset,
            "venue": d.market.venue,
            "market_id": d.market.market_id,
            "side": d.side,
            "stake_usd": d.stake_usd,
            "kelly_fraction": d.kelly_fraction,
            "edge": d.edge,
        }
        for d in LATEST_DECISIONS
    ]


@app.get("/adbriefs")
def ad_briefs():
    return [b.model_dump(mode="json") for b in LATEST_AD_BRIEFS[-20:]]


@app.get("/stats")
def stats():
    by_asset = {}
    for asset, history in _orchestrator.feedback._history.items():
        if not history:
            continue
        n = len(history)
        hits = sum(1 for o in history if o.correct)
        pnl = sum(o.pnl_usd for o in history)
        by_asset[asset.value] = {
            "sample_size": n,
            "hit_rate": round(hits / n, 4),
            "cumulative_pnl_usd": round(pnl, 2),
        }
    return by_asset


@app.get("/cascade/stats")
def cascade_stats():
    conn = get_cascade_db_connection()
    try:
        total = conn.execute("""
            SELECT COUNT(*), 
                   COUNT(CASE WHEN correct = 1 THEN 1 END),
                   COUNT(CASE WHEN correct = 0 THEN 1 END),
                   ROUND(SUM(pnl_usd), 2),
                   MAX(bankroll),
                   ROUND(AVG(CASE WHEN correct = 1 THEN 100.0 ELSE NULL END), 1)
            FROM trades
            WHERE correct IS NOT NULL
        """).fetchone()
        
        total_trades = total[0] or 0
        winning_trades = total[1] or 0
        losing_trades = total[2] or 0
        total_pnl = total[3] or 0.0
        peak_bankroll = total[4] or 1000.0
        hit_rate = total[5] or 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "hit_rate": hit_rate,
            "total_pnl": total_pnl,
            "peak_bankroll": peak_bankroll
        }
    finally:
        conn.close()


@app.get("/cascade/trades", response_model=List[Dict[str, Any]])
def cascade_trades(limit: int = 20):
    conn = get_cascade_db_connection()
    try:
        rows = conn.execute("""
            SELECT id, timestamp, asset, signal, p_up, p_down, side,
                   kelly_fraction, stake_usd, bankroll, correct, pnl_usd,
                   hit_rate, model, interval
            FROM trades
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    finally:
        conn.close()


@app.post("/run-cycle")
def run_cycle():
    """Manually trigger one research cycle (useful for demoing the dashboard)."""
    decisions = _orchestrator.run_cycle()
    return {"decisions_made": len(decisions)}
