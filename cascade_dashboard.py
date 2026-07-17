"""FastAPI dashboard for cascade engine forward testing results."""
from __future__ import annotations

import sqlite3
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cascade Engine Dashboard", version="1.0.0")
DB_PATH = "cascade_trades.db"

class TradeStats(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    peak_bankroll: float
    current_bankroll: float

class Trade(BaseModel):
    id: int
    timestamp: str
    asset: str
    conviction: str
    side: str
    stake_usd: float
    correct: bool | None
    pnl_usd: float
    bankroll: float
    hour_utc: int
    rationale: str

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats", response_model=TradeStats)
def get_stats():
    conn = get_db_connection()
    try:
        # Get overall stats
        total = conn.execute("""
            SELECT COUNT(*), 
                   COUNT(CASE WHEN correct=1 THEN 1 END),
                   COUNT(CASE WHEN correct=0 THEN 1 END),
                   ROUND(SUM(pnl_usd), 2),
                   MAX(bankroll),
                   (SELECT bankroll FROM cascade_trades ORDER BY id DESC LIMIT 1)
            FROM cascade_trades
            WHERE side != 'NONE'
        """).fetchone()
        
        total_trades = total[0] if total[0] else 0
        winning_trades = total[1] if total[1] else 0
        losing_trades = total[2] if total[2] else 0
        total_pnl = total[3] if total[3] else 0.0
        peak_bankroll = total[4] if total[4] else 0.0
        current_bankroll = total[5] if total[5] else 1000.0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl=total_pnl,
            peak_bankroll=peak_bankroll,
            current_bankroll=current_bankroll
        )
    finally:
        conn.close()

@app.get("/trades", response_model=List[Trade])
def get_trades(limit: int = 50):
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT id, timestamp, asset, conviction, side, stake_usd, 
                   correct, pnl_usd, bankroll, hour_utc, rationale
            FROM cascade_trades
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        trades = []
        for row in rows:
            trades.append(Trade(
                id=row["id"],
                timestamp=row["timestamp"],
                asset=row["asset"],
                conviction=row["conviction"],
                side=row["side"],
                stake_usd=row["stake_usd"],
                correct=bool(row["correct"]) if row["correct"] is not None else None,
                pnl_usd=row["pnl_usd"],
                bankroll=row["bankroll"],
                hour_utc=row["hour_utc"],
                rationale=row["rationale"]
            ))
        return trades
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
