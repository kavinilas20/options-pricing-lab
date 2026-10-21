"""Black-Scholes: textbook values, identities, Greeks vs finite differences, implied-vol round trips."""

import math

import numpy as np
import pytest

from src import black_scholes as bs

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_textbook_values():
    # Standard reference values for S=K=100, r=5%, sigma=20%, T=1
    assert bs.price(S, K, R, SIG, T, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs.price(S, K, R, SIG, T, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [70, 90, 100, 120, 150])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_put_call_parity(strike, sigma):
    c = bs.price(S, strike, R, sigma, T, "call")
    p = bs.price(S, strike, R, sigma, T, "put")
    assert c - p == pytest.approx(S - strike * math.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80, 100, 125])
def test_greeks_match_finite_differences(kind, strike):
    g = bs.greeks(S, strike, R, SIG, T, kind)
    f = lambda s=S, sig=SIG, t=T, r=R: float(bs.price(s, strike, r, sig, t, kind))
    h = 1e-4
    assert g["delta"] == pytest.approx((f(s=S + h) - f(s=S - h)) / (2 * h), rel=1e-6)
    hg_ = 1e-2    # a second difference divides rounding noise by h^2, so it needs a larger step
    assert g["gamma"] == pytest.approx((f(s=S + hg_) - 2 * f() + f(s=S - hg_)) / hg_ ** 2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sig=SIG + h) - f(sig=SIG - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(t=T + h) - f(t=T - h)) / (2 * h), rel=1e-6)   # theta = -dV/dT


def test_call_delta_minus_put_delta_is_one_and_gamma_vega_agree():
    gc, gp = bs.greeks(S, K, R, SIG, T, "call"), bs.greeks(S, K, R, SIG, T, "put")
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0)
    assert gc["gamma"] == pytest.approx(gp["gamma"]) and gc["vega"] == pytest.approx(gp["vega"])


def test_price_bounds_and_monotonicity():
    spots = np.linspace(50, 150, 41)
    c = bs.price(spots, K, R, SIG, T, "call")
    assert np.all(np.diff(c) > 0)                                   # call increases with spot
    assert np.all(c >= np.maximum(spots - K * math.exp(-R * T), 0) - 1e-12)
    assert np.all(c <= spots)
    vols = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(bs.price(S, K, R, vols, T, "call")) > 0)  # vega > 0


def test_expiry_limit_is_intrinsic_value():
    assert bs.price(110.0, 100.0, R, SIG, 1e-9, "call") == pytest.approx(10.0, abs=1e-6)
    assert bs.price(90.0, 100.0, R, SIG, 1e-9, "call") == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [60, 90, 100, 115, 160])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.7, 2.0])
@pytest.mark.parametrize("t", [0.05, 1.0, 3.0])
def test_implied_vol_round_trip(kind, strike, sigma, t):
    p = float(bs.price(S, strike, R, sigma, t, kind))
    lower = max(S - strike * math.exp(-R * t), 0) if kind == "call" else max(strike * math.exp(-R * t) - S, 0)
    if p - lower < 1e-8:            # price is numerically indistinguishable from intrinsic: vol not recoverable
        pytest.skip("price equals intrinsic value to machine precision")
    assert bs.implied_vol(p, S, strike, R, t, kind) == pytest.approx(sigma, rel=1e-5)


@pytest.mark.parametrize("bad_price", [-1.0, 0.0, 100.0, 150.0, 4.0])
def test_implied_vol_rejects_arbitrage_prices(bad_price):
    # 4.0 is below the call's intrinsic value S - K e^{-rT} = 4.88
    with pytest.raises(ValueError):
        bs.implied_vol(bad_price, S, K, R, T, "call")


def test_invalid_kind():
    with pytest.raises(ValueError):
        bs.price(S, K, R, SIG, T, "straddle")
