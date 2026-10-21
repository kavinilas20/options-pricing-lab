"""Binomial tree and Monte Carlo against Black-Scholes and against each other."""

import math

import numpy as np
import pytest

from src import binomial as bi
from src import black_scholes as bs
from src import monte_carlo as mc

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [85, 100, 120])
def test_binomial_converges_to_black_scholes(kind, strike):
    exact = float(bs.price(S, strike, R, SIG, T, kind))
    errs = [abs(bi.binomial_price(S, strike, R, SIG, T, n, kind) - exact) for n in (50, 400, 3000)]
    assert errs[2] < 0.01
    assert errs[2] < errs[0]


def test_one_step_tree_by_hand():
    # S=100, u=e^{0.2}, d=1/u, r=0, T=1, K=100: p=(1-d)/(u-d)
    u = math.exp(0.2); d = 1 / u; p = (1 - d) / (u - d)
    expected = p * (100 * u - 100)
    assert bi.binomial_price(100, 100, 0.0, 0.2, 1.0, 1, "call") == pytest.approx(expected)


def test_american_call_without_dividends_equals_european():
    eu = bi.binomial_price(S, K, R, SIG, T, 800, "call")
    am = bi.binomial_price(S, K, R, SIG, T, 800, "call", american=True)
    assert am == pytest.approx(eu, abs=1e-10)


@pytest.mark.parametrize("spot", [60, 80, 100, 120])
def test_american_put_bounds(spot):
    eu = bi.binomial_price(spot, K, R, SIG, T, 800, "put")
    am = bi.binomial_price(spot, K, R, SIG, T, 800, "put", american=True)
    assert am >= eu - 1e-12                     # early exercise right has non-negative value
    assert am >= max(K - spot, 0) - 1e-12       # never worth less than exercising now


def test_american_put_known_value_and_deep_itm_is_intrinsic():
    assert bi.binomial_price(100, 100, 0.05, 0.2, 1.0, 2000, "put", american=True) == pytest.approx(6.09, abs=0.01)
    # very deep in the money, high rates: exercise immediately
    assert bi.binomial_price(20, 100, 0.08, 0.2, 1.0, 800, "put", american=True) == pytest.approx(80.0, abs=1e-9)


def test_binomial_rejects_unstable_tree():
    with pytest.raises(ValueError):
        bi.binomial_price(100, 100, 0.5, 0.01, 5.0, 2, "call")   # p outside (0,1)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
def test_mc_within_four_standard_errors_of_black_scholes(kind, flags):
    exact = float(bs.price(S, K, R, SIG, T, kind))
    est, se = mc.mc_european(S, K, R, SIG, T, kind, n=200_000, seed=11,
                             antithetic=flags[0], control_variate=flags[1])
    assert abs(est - exact) < 4 * se


def test_mc_is_deterministic_for_a_seed_and_differs_across_seeds():
    a = mc.mc_european(S, K, R, SIG, T, n=5000, seed=3)
    assert a == mc.mc_european(S, K, R, SIG, T, n=5000, seed=3)
    assert a != mc.mc_european(S, K, R, SIG, T, n=5000, seed=4)


def test_variance_reduction_actually_reduces_standard_error():
    n = 100_000
    plain = mc.mc_european(S, K, R, SIG, T, n=n, seed=5)[1]
    anti = mc.mc_european(S, K, R, SIG, T, n=n, seed=5, antithetic=True)[1]
    cv = mc.mc_european(S, K, R, SIG, T, n=n, seed=5, control_variate=True)[1]
    both = mc.mc_european(S, K, R, SIG, T, n=n, seed=5, antithetic=True, control_variate=True)[1]
    assert anti < plain and cv < plain and both < min(anti, cv)


def test_mc_confidence_interval_coverage_is_close_to_nominal():
    exact = float(bs.price(S, K, R, SIG, T, "call"))
    hits, reps = 0, 400
    for i in range(reps):
        est, se = mc.mc_european(S, K, R, SIG, T, n=2000, seed=1000 + i)
        hits += abs(est - exact) < 1.96 * se
    assert 0.92 < hits / reps < 0.98            # nominal 0.95; sd of the estimate ~ 0.011


@pytest.mark.parametrize("kind", ["call", "put"])
def test_pathwise_delta_matches_black_scholes_delta(kind):
    exact = float(bs.greeks(S, K, R, SIG, T, kind)["delta"])
    est, se = mc.mc_delta_pathwise(S, K, R, SIG, T, kind, n=400_000, seed=2)
    assert abs(est - exact) < 4 * se
