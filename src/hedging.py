r"""
hedging.py
----------
Discrete delta-hedging of a SOLD European option, simulated path by path.

The seller
    * sells the option at its Black-Scholes value V0 computed with `sigma_imp`
      (the volatility priced in),
    * holds Delta_k shares, with Delta_k the Black-Scholes delta at `sigma_imp`,
      re-set at n equally spaced times,
    * keeps the rest in a cash account earning the risk-free rate r.

The stock actually moves with volatility `sigma_real` (and drift mu).
Final P&L (in time-T dollars) = cash + shares * S_T - payoff.

Facts this module is used to check
    1. sigma_real == sigma_imp: mean P&L is ~0 and its standard deviation
       falls like 1/sqrt(n) as the hedge frequency n grows.
    2. sigma_real != sigma_imp (continuous-hedging limit, drift mu = r):
           E[P&L] = (1/2)(sigma_imp^2 - sigma_real^2) *
                    Integral_0^T e^{r(T-t)} E[ S_t^2 Gamma_imp(t, S_t) ] dt
       The seller loses when realised volatility exceeds the volatility sold.
       `theory_mean_pnl` evaluates this numerically.
"""

import math

import numpy as np

from src import black_scholes as bs


def simulate_delta_hedge(S0, K, r, sigma_imp, sigma_real, T, n_steps, n_paths, seed=0,
                         kind="call", mu=None, hedge=True):
    """Returns the array of final P&L, one entry per path (seller's point of view)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if not all(math.isfinite(x) for x in (S0, K, r, sigma_imp, sigma_real, T)):
        raise ValueError("S0, K, r, volatilities and T must be finite")
    if S0 <= 0 or K <= 0 or sigma_imp <= 0 or sigma_real <= 0 or T <= 0:
        raise ValueError("S0, K, volatilities and T must be positive")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(n_paths, bool) or int(n_paths) != n_paths or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    n_steps, n_paths = int(n_steps), int(n_paths)
    mu = r if mu is None else mu
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    grow = math.exp(r * dt)
    v0 = float(bs.price(S0, K, r, sigma_imp, T, kind))

    S = np.full(n_paths, S0, dtype=float)
    if hedge:
        delta = bs.greeks(S, K, r, sigma_imp, T, kind)["delta"]
        cash = v0 - delta * S
    else:
        delta = np.zeros(n_paths)
        cash = np.full(n_paths, v0)

    drift = (mu - 0.5 * sigma_real ** 2) * dt
    vol = sigma_real * math.sqrt(dt)
    for k in range(1, n_steps + 1):
        S = S * np.exp(drift + vol * rng.standard_normal(n_paths))
        cash = cash * grow
        if hedge and k < n_steps:                    # re-hedge at t_k (no trade at expiry)
            new_delta = bs.greeks(S, K, r, sigma_imp, T - k * dt, kind)["delta"]
            cash = cash - (new_delta - delta) * S
            delta = new_delta
    payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    return cash + delta * S - payoff


def theory_mean_pnl(S0, K, r, sigma_imp, sigma_real, T, kind="call", n_tau=80, n_z=6001):
    """Continuous-hedging expected P&L when the stock drifts at r (see module docstring)."""
    z = np.linspace(-9.0, 9.0, n_z)
    wz = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    dz = z[1] - z[0]
    # integrate over s = sqrt(T - t):  dt = 2 s ds  removes the 1/sqrt(tau) singularity
    nodes, weights = np.polynomial.legendre.leggauss(n_tau)
    s_nodes = 0.5 * math.sqrt(T) * (nodes + 1.0)
    s_weights = 0.5 * math.sqrt(T) * weights
    total = 0.0
    for s, w in zip(s_nodes, s_weights):
        tau = s * s
        t = T - tau
        S_t = S0 * np.exp((r - 0.5 * sigma_real ** 2) * t + sigma_real * math.sqrt(t) * z)
        gamma = bs.greeks(S_t, K, r, sigma_imp, tau, kind)["gamma"]
        e_s2g = float(np.sum(wz * S_t ** 2 * gamma) * dz)
        total += w * 2.0 * s * math.exp(r * tau) * e_s2g
    return 0.5 * (sigma_imp ** 2 - sigma_real ** 2) * total
