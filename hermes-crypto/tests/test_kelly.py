import pytest

from core.kelly import kelly_fraction, stake_amount


def test_no_edge_gives_zero_kelly():
    # p_win == price -> no edge -> Kelly fraction should be ~0
    res = kelly_fraction(p_win=0.5, price=0.5, fraction_cap=1.0)
    assert res.raw_fraction == pytest.approx(0.0, abs=1e-9)


def test_positive_edge_gives_positive_kelly():
    res = kelly_fraction(p_win=0.65, price=0.5, fraction_cap=1.0)
    assert res.raw_fraction > 0
    assert res.edge == pytest.approx(0.15)


def test_negative_edge_clips_to_zero():
    res = kelly_fraction(p_win=0.3, price=0.5, fraction_cap=1.0)
    assert res.raw_fraction == 0.0


def test_fraction_cap_halves_stake():
    full = kelly_fraction(p_win=0.7, price=0.5, fraction_cap=1.0)
    half = kelly_fraction(p_win=0.7, price=0.5, fraction_cap=0.5)
    assert half.capped_fraction == pytest.approx(full.raw_fraction * 0.5)


def test_stake_amount_respects_max_stake_pct():
    stake = stake_amount(p_win=0.99, price=0.1, bankroll=1000, fraction_cap=1.0, max_stake_pct=0.05)
    assert stake == pytest.approx(50.0)


def test_invalid_price_raises():
    with pytest.raises(ValueError):
        kelly_fraction(p_win=0.5, price=0.0)
    with pytest.raises(ValueError):
        kelly_fraction(p_win=0.5, price=1.0)


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        kelly_fraction(p_win=1.5, price=0.5)
