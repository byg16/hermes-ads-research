"""Demo script — runs all 5 agents with synthetic data so output is always visible."""
from datetime import datetime, timezone, timedelta
from core.logging_config import configure_logging
from core.schemas import Asset, MarketCandidate, Venue
from agents.data_fetcher import DataFetcherAgent
from agents.predictor import PredictorAgent
from agents.risk_manager import RiskManagerAgent
from agents.feedback_loop import FeedbackLoopAgent

configure_logging()

print("\n" + "="*60)
print("  HERMES ADS RESEARCH — LIVE DEMO")
print("="*60 + "\n")

# Synthetic markets (mimicking real Polymarket/Kalshi 5-min contracts)
DEMO_MARKETS = [
    MarketCandidate(
        venue=Venue.POLYMARKET, asset=Asset.BTC,
        market_id="btc-5min-up-001",
        title="Will BTC be higher in 5 minutes?",
        yes_price=0.47, no_price=0.53,
        close_time=datetime.now(timezone.utc) + timedelta(minutes=5),
        horizon_minutes=5,
        url="https://polymarket.com/event/btc-5min"
    ),
    MarketCandidate(
        venue=Venue.KALSHI, asset=Asset.ETH,
        market_id="KXETH-5MIN-UP-001",
        title="Will ETH be higher in 5 minutes?",
        yes_price=0.52, no_price=0.48,
        close_time=datetime.now(timezone.utc) + timedelta(minutes=5),
        horizon_minutes=5,
        url="https://kalshi.com/markets/KXETH-5MIN"
    ),
    MarketCandidate(
        venue=Venue.POLYMARKET, asset=Asset.ETH,
        market_id="eth-5min-up-002",
        title="Will ETH price rise next 5 minutes?",
        yes_price=0.44, no_price=0.56,
        close_time=datetime.now(timezone.utc) + timedelta(minutes=5),
        horizon_minutes=5,
        url="https://polymarket.com/event/eth-5min"
    ),
]

fetcher = DataFetcherAgent()
predictor = PredictorAgent()
risk_manager = RiskManagerAgent()
feedback = FeedbackLoopAgent()

for market in DEMO_MARKETS:
    print(f"\n{'─'*60}")
    print(f"  AGENT 1 — Market Found")
    print(f"  Venue   : {market.venue.value.upper()}")
    print(f"  Asset   : {market.asset.value}")
    print(f"  Market  : {market.title}")
    print(f"  YES price: {market.yes_price} | NO price: {market.no_price}")

    print(f"\n  AGENT 2 — Fetching OHLCV bars via Apify...")
    series = fetcher.run(market.asset, n_bars=200)
    print(f"  Fetched {len(series.bars)} bars | Source: {series.source}")

    print(f"\n  AGENT 3 — Kronos Prediction Model")
    context = feedback.context_for_predictor(market.asset)
    prediction = predictor.run(series, horizon_minutes=5, context=context)
    direction = "UP" if prediction.p_up > 0.5 else "DOWN"
    print(f"  P(UP)  : {prediction.p_up:.2%}")
    print(f"  P(DOWN): {prediction.p_down:.2%}")
    print(f"  Signal : {direction}")
    print(f"  Model  : {prediction.model_name}")

    print(f"\n  AGENT 4 — Risk Manager (Kelly Criterion)")
    decision = risk_manager.run(market, prediction)
    print(f"  Side         : {decision.side}")
    print(f"  Edge         : {decision.edge:.2%}")
    print(f"  Kelly fraction: {decision.kelly_fraction:.2%} of bankroll")
    print(f"  Stake        : ${decision.stake_usd}")
    print(f"  Paper trading: ON (no real money)")

    print(f"\n  AGENT 5 — Feedback Loop + Ad Brief")
    brief = feedback.build_ad_brief(market.asset, decision)
    print(f"  Headline: {brief.headline}")
    print(f"  Copy    : {brief.body}")

print(f"\n{'='*60}")
print("  PIPELINE COMPLETE — all 5 agents ran successfully")
print(f"{'='*60}\n")