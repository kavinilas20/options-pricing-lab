r"""
black_scholes.py
----------------
Closed-form Black-Scholes prices, Greeks and implied volatility for European
options on a non-dividend-paying asset.

    d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
    call = S N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S N(-d1)

All functions accept scalars or numpy arrays (broadcasting), except
`implied_vol`, which works on one option at a time.
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def _check_market_inputs(S, K, r, sigma, T):
    for name, value in (("S", S), ("K", K), ("r", r), ("sigma", sigma), ("T", T)):
        if np.any(~np.isfinite(np.asarray(value))):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
        raise ValueError("S and K must be positive")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(T) <= 0):
        raise ValueError("sigma and T must be positive")


def d1_d2(S, K, r, sigma, T):
    _check_market_inputs(S, K, r, sigma, T)
    sig_rt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sig_rt
    return d1, d1 - sig_rt


def norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def price(S, K, r, sigma, T, kind="call"):
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    disc = np.exp(-r * T)
    if kind == "call":
        return S * ndtr(d1) - K * disc * ndtr(d2)
    return K * disc * ndtr(-d2) - S * ndtr(-d1)


def greeks(S, K, r, sigma, T, kind="call"):
    """Delta, gamma, vega (per 1.00 of vol), theta (per year), rho (per 1.00 of rate)."""
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    disc = np.exp(-r * T)
    pdf = norm_pdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    if kind == "call":
        delta = ndtr(d1)
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) - r * K * disc * ndtr(d2)
        rho = K * T * disc * ndtr(d2)
    else:
        delta = ndtr(d1) - 1.0
        theta = -S * pdf * sigma / (2 * np.sqrt(T)) + r * K * disc * ndtr(-d2)
        rho = -K * T * disc * ndtr(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(option_price, S, K, r, T, kind="call", lo=1e-6, hi=10.0):
    """
    Volatility that makes the Black-Scholes price equal `option_price`.
    Raises ValueError when no volatility can do so (price outside the
    no-arbitrage bounds), instead of returning a meaningless number.
    """
    _check_kind(kind)
    if not all(math.isfinite(x) for x in (option_price, S, K, r, T, lo, hi)):
        raise ValueError("option inputs and search bounds must be finite")
    _check_market_inputs(S, K, r, 1.0, T)
    if lo <= 0 or hi <= lo:
        raise ValueError("search bounds must satisfy 0 < lo < hi")
    disc = math.exp(-r * T)
    if kind == "call":
        lower, upper = max(S - K * disc, 0.0), S
    else:
        lower, upper = max(K * disc - S, 0.0), K * disc
    if not (lower < option_price < upper):
        raise ValueError(
            f"price {option_price:.6g} is outside the no-arbitrage range ({lower:.6g}, {upper:.6g})")

    def f(v):
        return float(price(S, K, r, v, T, kind)) - option_price

    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("no implied volatility in the search range [%g, %g]" % (lo, hi))
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=500)
