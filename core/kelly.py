"""Kelly criterion math for binary-outcome prediction markets.

For a binary share priced at `price` in (0, 1) that pays out $1 if correct
and $0 otherwise, the net odds received on a winning bet are:

    b = (1 - price) / price

Standard Kelly fraction for a binary bet with model probability `p` of
winning:

    f* = p - (1 - p) / b   =   (p * b - (1 - p)) / b

We additionally support fractional Kelly (e.g. half-Kelly) via a cap
multiplier, and clip to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyResult:
    raw_fraction: float
    capped_fraction: float
    edge: float
    odds_b: float


def kelly_fraction(p_win: float, price: float, fraction_cap: float = 1.0) -> KellyResult:
    """Compute the Kelly-optimal fraction of bankroll to stake.

    Args:
        p_win: model-estimated probability the bet wins (0..1).
        price: current market price of the share being bought (0..1),
            i.e. cost to win $1.
        fraction_cap: multiplier applied to raw Kelly (e.g. 0.5 = half-Kelly)
            to reduce variance. Final result is also clipped to [0, 1].

    Returns:
        KellyResult with raw and capped fractions plus diagnostic edge/odds.
    """
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0,1), got {price}")
    if not (0.0 <= p_win <= 1.0):
        raise ValueError(f"p_win must be in [0,1], got {p_win}")

    b = (1.0 - price) / price
    raw = p_win - (1.0 - p_win) / b
    raw = max(raw, 0.0)  # never bet negative Kelly; just sit out
    edge = p_win - price  # simple probability-vs-price edge, for reporting

    capped = min(raw * fraction_cap, 1.0)
    return KellyResult(raw_fraction=raw, capped_fraction=capped, edge=edge, odds_b=b)


def stake_amount(p_win: float, price: float, bankroll: float, fraction_cap: float = 0.5,
                  max_stake_pct: float = 0.05) -> float:
    """Convenience wrapper: dollar stake after applying both Kelly fraction
    cap and an absolute max-bankroll-percent risk limit."""
    result = kelly_fraction(p_win, price, fraction_cap)
    pct = min(result.capped_fraction, max_stake_pct)
    return round(bankroll * pct, 2)
