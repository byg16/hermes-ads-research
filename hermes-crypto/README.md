# Hermes Ads Research — Crypto Prediction Market Agent Pipeline

A backend agent pipeline built on the **Hermes Agents** framework that researches
short-horizon crypto prediction markets (Polymarket + Kalshi) for BTC/ETH, pulls
recent price action via **Apify**, forecasts the next up/down move with a
**Kronos**-style time-series model, sizes positions with the **Kelly criterion**,
and feeds outcomes back into the loop — producing structured "ad copy"-ready
research briefs that a downstream ads/marketing pipeline can consume
(e.g. "ETH 5-min up-probability 68% — feature in promo").

> This is intentionally a *research/paper-trading* pipeline. No real funds are
> moved. `risk_manager.py` only ever computes a recommended stake; execution is
> a stub you can wire to real Polymarket/Kalshi order placement once you've
> validated the strategy.

## Why "Ads"?

The brief asked for a "search and research tool for creating Ads." The pipeline
here treats each market prediction as a piece of **content**: the
`feedback_loop` agent turns every cycle's prediction + outcome + Kelly stake
into a short structured "ad brief" (`AdBrief` schema) — e.g. promotional copy
ideas like *"Our model called ETH's next 5 minutes correctly 7 times in a
row — try the bot"* — which is what a marketing/ads agent downstream would
consume. See `core/schemas.py::AdBrief`.

## Architecture

```
                 ┌─────────────────────┐
                 │ 1. MarketScanner     │  Polymarket + Kalshi search
                 │    (agents/market_   │  for BTC/ETH 5-min markets
                 │    scanner.py)       │
                 └─────────┬────────────┘
                           │ MarketCandidate[]
                 ┌─────────▼────────────┐
                 │ 2. DataFetcher        │  Apify actor → last N OHLCV bars
                 │    (agents/data_      │
                 │    fetcher.py)        │
                 └─────────┬────────────┘
                           │ OHLCVSeries
                 ┌─────────▼────────────┐
                 │ 3. Predictor          │  Kronos forecasting model
                 │    (agents/predictor  │  → P(up) next bar
                 │    .py)               │
                 └─────────┬────────────┘
                           │ Prediction
                 ┌─────────▼────────────┐
                 │ 4. RiskManager        │  Kelly criterion stake sizing
                 │    (agents/risk_      │
                 │    manager.py)        │
                 └─────────┬────────────┘
                           │ Decision
                 ┌─────────▼────────────┐
                 │ 5. FeedbackLoop       │  Hermes agent loop: scores past
                 │    (agents/feedback_  │  predictions, updates running
                 │    loop.py)           │  stats, emits AdBrief
                 └─────────┬────────────┘
                           │
                 ┌─────────▼────────────┐
                 │ core/orchestrator.py  │  Wires all agents into a Hermes
                 │                       │  Agent loop, runs on a schedule
                 └───────────────────────┘
```

All LLM-backed agents (scanner reasoning, brief writing) call **OpenRouter**
via `tools/llm_client.py`, defaulting to a free model
(`meta-llama/llama-3.1-8b-instruct:free`) — swap via `.env`.

## Project layout

```
hermes_ads_research/
├── main.py                  # entrypoint, runs orchestrator loop
├── config.py                 # env-driven settings (pydantic-settings)
├── agents/
│   ├── market_scanner.py     # Agent 1
│   ├── data_fetcher.py       # Agent 2
│   ├── predictor.py          # Agent 3
│   ├── risk_manager.py       # Agent 4
│   └── feedback_loop.py      # Agent 5
├── tools/
│   ├── polymarket_client.py  # Gamma/CLOB REST wrapper
│   ├── kalshi_client.py      # Kalshi REST wrapper
│   ├── apify_client.py       # Apify actor runner for OHLCV scraping
│   ├── kronos_client.py      # Kronos model wrapper (local or HF endpoint)
│   └── llm_client.py         # OpenRouter chat-completions wrapper
├── core/
│   ├── orchestrator.py       # Hermes Agent graph + run loop
│   ├── schemas.py            # pydantic data contracts between agents
│   ├── kelly.py              # Kelly criterion math, unit-tested
│   └── logging_config.py     # structured logging setup
├── scale/
│   ├── multi_timeframe.py    # 1-min→5-min cascade, 15-min arbitrage idea
│   └── dashboard.py          # FastAPI read-only dashboard for visibility
├── tests/
│   └── test_kelly.py
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY, APIFY_API_TOKEN, etc.
```

## Run

```bash
# one research cycle for BTC + ETH across both venues
python main.py --once

# continuous loop, polling every 60s, paper-trading
python main.py --loop --interval 60

# visibility dashboard (read-only, shows live predictions/Kelly stakes/AdBriefs)
uvicorn scale.dashboard:app --reload --port 8000
```

## Required keys (.env)

| Variable | Purpose | Free tier? |
|---|---|---|
| `OPENROUTER_API_KEY` | LLM calls via OpenRouter | Yes — use `:free` model suffixes |
| `OPENROUTER_MODEL` | e.g. `meta-llama/llama-3.1-8b-instruct:free` | — |
| `APIFY_API_TOKEN` | Run Apify actors to scrape OHLCV bars | Yes, free tier |
| `POLYMARKET_GAMMA_URL` | Defaults to public Gamma API, no key needed | — |
| `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` | Optional — only needed for authenticated endpoints; public markets are read without auth | — |
| `KRONOS_MODE` | `local` (runs the open-source Kronos checkpoint) or `stub` (lightweight statistical fallback so the project runs with zero GPU/model download) | — |

## Kelly criterion (agents/risk_manager.py + core/kelly.py)

Implements the standard binary-outcome Kelly fraction:

```
f* = p - (1 - p) / b
```

where `p` = model's predicted probability of "up" (or "down"), and `b` =
net odds offered by the market price (`(1 - price) / price` for a binary
share priced in [0,1], as on Polymarket/Kalshi). We clip to
`[0, KELLY_FRACTION_CAP]` (default half-Kelly, cap 0.5×) to control variance,
and to a max bankroll fraction (`MAX_STAKE_PCT`, default 5%) as a hard
risk limit independent of Kelly's raw output.

## Scaling ideas implemented (see `scale/`)

1. **Timeframe cascade** (`multi_timeframe.py::cascade_predict`): runs Kronos
   on 1-min bars to predict bar *n+5*, then re-derives a 5-min-bar up/down
   call for *n+1* by aggregating the five 1-min predictions — cheaper and
   higher-frequency than calling Kronos directly on 5-min bars, and lets you
   cross-validate the two horizons against each other.
2. **Internal arbitrage scan** (`multi_timeframe.py::scan_arbitrage`):
   compares the model's *implied* 15-min up-probability (compounding three
   consecutive 5-min predictions) against the *market's* quoted 15-min
   contract price, flagging mispriced spread opportunities between the
   15-min market and the chain of three 5-min markets.
3. **User visibility**: `scale/dashboard.py` is a minimal FastAPI app
   exposing `/predictions`, `/stakes`, `/adbriefs`, and `/stats` (running
   hit-rate, Brier score, cumulative Kelly PnL) for a frontend or Slack bot
   to consume.

## Notes on the Hermes Agents framework integration

`core/orchestrator.py` wraps each of the 5 agents as a Hermes `Agent` node
with a typed input/output contract (pydantic models in `core/schemas.py`),
and chains them with `HermesLoop`, which is what gives us point (5):
the loop re-invokes `FeedbackLoop` after each market resolves, updates a
running scoreboard, and *that scoreboard is fed back as additional context*
into `Predictor`'s next system prompt — closing the feedback loop described
in the task. If the `hermes-agent` package isn't installed/available in your
environment (it's an early-stage framework), `core/orchestrator.py` falls
back to a minimal compatible shim (`core/_hermes_shim.py`) implementing the
same `Agent`/`Loop` interface, so the project still runs end-to-end.

## Testing

```bash
pytest tests/ -v
```

## Disclaimer

Prediction markets and crypto are high-risk. This is a research/engineering
exercise — nothing here is financial advice, and the risk manager's Kelly
sizing is for paper-trading simulation by default (`PAPER_TRADING=true`).
