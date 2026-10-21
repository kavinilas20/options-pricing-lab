r"""
binomial.py
-----------
Cox-Ross-Rubinstein binomial tree for European and American options.

    u = e^{sigma sqrt(dt)},  d = 1/u,  p = (e^{r dt} - d) / (u - d)

The option value at each node is the discounted risk-neutral expectation of
the two children. For an American option the value is also floored at the
immediate-exercise value at every node.

Vectorised with numpy: one backward step is one array operation, so the cost
is O(N^2) arithmetic but only N Python-level iterations.
"""

import math

import numpy as np


def binomial_price(S, K, r, sigma, T, n_steps, kind="call", american=False):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if not all(math.isfinite(x) for x in (S, K, r, sigma, T)):
        raise ValueError("S, K, r, sigma and T must be finite")
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma and T must be positive")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0,1): use more steps")
    disc = math.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(n_steps + 1)
    prices = S * u ** (2 * j - n_steps)                  # S u^j d^(n-j), j = number of up moves
    values = np.maximum(sign * (prices - K), 0.0)
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            jj = np.arange(step + 1)
            node_prices = S * u ** (2 * jj - step)
            values = np.maximum(values, sign * (node_prices - K))
    return float(values[0])
