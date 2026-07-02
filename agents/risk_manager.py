"""Agent 4: sizes positions using the Kelly criterion against a chosen market."""
from __future__ import annotations

from core.kelly import kelly_fraction
from core.logging_config import get_logger
from core.schemas import Decision, MarketCandidate, Prediction
from config import settings

logger = get_logger(__name__)


class RiskManagerAgent:
    name = "risk_manager"

    def run(self, market: MarketCandidate, prediction: Prediction) -> Decision:
        # Decide side: bet YES if our P(up) exceeds the YES price (positive
        # edge buying "up" shares), otherwise consider NO (betting "down").
        edge_yes = prediction.p_up - market.yes_price
        edge_no = prediction.p_down - market.no_price

        if edge_yes >= edge_no and edge_yes > 0:
            side, p_win, price, edge = "YES", prediction.p_up, market.yes_price, edge_yes
        elif edge_no > 0:
            side, p_win, price, edge = "NO", prediction.p_down, market.no_price, edge_no
        else:
            side, p_win, price, edge = "NONE", 0.0, 0.5, 0.0

        if side == "NONE":
            decision = Decision(
                market=market, prediction=prediction, kelly_fraction=0.0, stake_usd=0.0,
                side=side, edge=0.0, notes="No positive edge found; sitting out.",
            )
            logger.info("risk_decision_no_bet", market_id=market.market_id, asset=market.asset)
            return decision

        result = kelly_fraction(p_win=p_win, price=price, fraction_cap=settings.kelly_fraction_cap)
        capped_pct = min(result.capped_fraction, settings.max_stake_pct)
        stake = round(settings.bankroll_usd * capped_pct, 2)

        decision = Decision(
            market=market,
            prediction=prediction,
            kelly_fraction=round(capped_pct, 4),
            stake_usd=stake,
            side=side,
            edge=round(edge, 4),
            notes=(
                f"raw_kelly={result.raw_fraction:.4f} fraction_cap={settings.kelly_fraction_cap} "
                f"max_stake_pct={settings.max_stake_pct} paper_trading={settings.paper_trading}"
            ),
        )
        logger.info(
            "risk_decision",
            market_id=market.market_id,
            side=side,
            stake_usd=stake,
            kelly_fraction=capped_pct,
            edge=edge,
        )
        return decision
