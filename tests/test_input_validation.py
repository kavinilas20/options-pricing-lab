"""Public APIs reject invalid numerical inputs with clear errors."""

import pytest

from src import binomial, black_scholes, hedging, monte_carlo


@pytest.mark.parametrize("args", [
    (0, 100, 0.05, 0.2, 1.0),
    (100, -1, 0.05, 0.2, 1.0),
    (100, 100, 0.05, 0.0, 1.0),
    (100, 100, 0.05, 0.2, 0.0),
])
def test_black_scholes_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        black_scholes.price(*args)


def test_implied_vol_rejects_invalid_search_bounds():
    with pytest.raises(ValueError):
        black_scholes.implied_vol(10, 100, 100, 0.05, 1, lo=0, hi=10)


def test_binomial_rejects_invalid_steps_and_market_inputs():
    with pytest.raises(ValueError):
        binomial.binomial_price(100, 100, 0.05, 0.2, 1, 0)
    with pytest.raises(ValueError):
        binomial.binomial_price(100, 100, 0.05, 0, 1, 100)


def test_monte_carlo_rejects_invalid_sample_budget():
    with pytest.raises(ValueError):
        monte_carlo.mc_european(100, 100, 0.05, 0.2, 1, n=1)
    with pytest.raises(ValueError):
        monte_carlo.mc_european(100, 100, 0.05, 0.2, 1, n=101, antithetic=True)


def test_monte_carlo_delta_rejects_invalid_kind():
    with pytest.raises(ValueError):
        monte_carlo.mc_delta_pathwise(100, 100, 0.05, 0.2, 1, kind="straddle")


def test_hedging_rejects_invalid_path_count():
    with pytest.raises(ValueError):
        hedging.simulate_delta_hedge(100, 100, 0.05, 0.2, 0.2, 1, 10, 1)
