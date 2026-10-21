r"""
monte_carlo.py
--------------
Monte Carlo pricing of European options under geometric Brownian motion,
with two classic variance-reduction methods.

    S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0,1)
    price = e^{-rT} E[payoff(S_T)]

`n` is always the TOTAL number of payoff evaluations, so methods can be
compared at equal cost.

Antithetic variates: pair every draw Z with -Z and average the two payoffs.
Control variate: the discounted terminal price e^{-rT} S_T has known mean S0;
subtract beta * (C - S0), with beta the sample regression slope. Estimating
beta from the same sample adds a bias of order 1/n, negligible here.
"""

import math

import numpy as np


def _payoff(S_T, K, kind):
    return np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)


def mc_european(S, K, r, sigma, T, kind="call", n=100_000, seed=0,
                antithetic=False, control_variate=False):
    """Returns (price estimate, standard error)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if not all(math.isfinite(x) for x in (S, K, r, sigma, T)):
        raise ValueError("S, K, r, sigma and T must be finite")
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma and T must be positive")
    if isinstance(n, bool) or int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    n = int(n)
    if antithetic and n % 2:
        raise ValueError("n must be even when antithetic=True")
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)

    if antithetic:
        m = n // 2                                   # pairs -> n payoff evaluations
        z = rng.standard_normal(m)
        s_a, s_b = S * np.exp(drift + vol * z), S * np.exp(drift - vol * z)
        y = 0.5 * disc * (_payoff(s_a, K, kind) + _payoff(s_b, K, kind))
        c = 0.5 * disc * (s_a + s_b)
    else:
        m = n
        z = rng.standard_normal(m)
        s_t = S * np.exp(drift + vol * z)
        y = disc * _payoff(s_t, K, kind)
        c = disc * s_t

    if control_variate:
        beta = np.cov(y, c, ddof=1)[0, 1] / np.var(c, ddof=1)
        y = y - beta * (c - S)                       # E[c] = S exactly
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(m))


def mc_delta_pathwise(S, K, r, sigma, T, kind="call", n=200_000, seed=0):
    """
    Pathwise delta estimator: d/dS0 of e^{-rT} payoff(S_T) = e^{-rT} 1{exercised} S_T / S0.
    Returns (delta estimate, standard error).
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if not all(math.isfinite(x) for x in (S, K, r, sigma, T)):
        raise ValueError("S, K, r, sigma and T must be finite")
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma and T must be positive")
    if isinstance(n, bool) or int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    n = int(n)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    s_t = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    disc = math.exp(-r * T)
    if kind == "call":
        d = disc * (s_t > K) * s_t / S
    else:
        d = -disc * (s_t < K) * s_t / S
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
