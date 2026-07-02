"""Minimal read-only FastAPI dashboard for user visibility (scaling idea 6.3).

Exposes the latest decisions, ad briefs, and aggregate stats. Intended to be
polled by a frontend or piped into a Slack bot. Run with:

    uvicorn scale.dashboard:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from core.orchestrator import LATEST_DECISIONS, LATEST_AD_BRIEFS, Orchestrator

app = FastAPI(title="Hermes Ads Research Dashboard", version="0.1.0")
_orchestrator = Orchestrator()


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


@app.post("/run-cycle")
def run_cycle():
    """Manually trigger one research cycle (useful for demoing the dashboard)."""
    decisions = _orchestrator.run_cycle()
    return {"decisions_made": len(decisions)}
