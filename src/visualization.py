"""
visualization.py -- figures for the options pricing lab.

One colour per entity across figures (blue = main series, orange = comparison,
aqua = third series), neutral grey for theory/reference lines, thin lines,
recessive grid, a legend whenever there are two or more series.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DPI = 160

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2.0,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_binomial_convergence(n_steps, errors_by_label, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [BLUE, ORANGE]
    for (label, errs), c in zip(errors_by_label.items(), colors):
        ax.plot(n_steps, errs, marker="o", markersize=6, color=c, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ref = np.asarray(n_steps, dtype=float)
    ax.plot(ref, errors_by_label[next(iter(errors_by_label))][0] * ref[0] / ref, color=GREY,
            linestyle=(0, (4, 3)), linewidth=1.4, label="Reference slope $1/N$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of tree steps $N$")
    ax.set_ylabel("Absolute error vs Black-Scholes")
    ax.set_title("Binomial price converges to Black-Scholes, with oscillation")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, out_dir, "01_binomial_convergence.png")


def plot_variance_reduction(labels, ses, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, ses, color=BLUE, height=0.55)
    for yi, s in zip(y, ses):
        ax.text(s + max(ses) * 0.015, yi, f"{s:.4f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, max(ses) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standard error of the price estimate (100,000 payoff evaluations)")
    ax.set_title("Variance reduction: same cost, smaller error")
    return _save(fig, out_dir, "02_mc_variance_reduction.png")


def plot_greeks(spots, greeks, K, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.4), sharex=True)
    names = [("delta", "Delta"), ("gamma", "Gamma"), ("vega", "Vega"), ("theta", "Theta (per year)")]
    for ax, (key, title) in zip(axes.ravel(), names):
        ax.plot(spots, greeks[key], color=BLUE)
        ax.axvline(K, color=GREY, linewidth=1, linestyle=":")
        ax.set_title(title, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Spot price $S$")
    fig.suptitle(f"Greeks of a European call (strike {K:g}, dotted line)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "03_greeks.png")


def plot_american_put(spots, american, european, K, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(spots, np.maximum(K - spots, 0), color=GREY, linestyle=(0, (4, 3)), linewidth=1.4,
            label="Exercise now: $\\max(K-S,0)$")
    ax.plot(spots, european, color=ORANGE, label="European put")
    ax.plot(spots, american, color=BLUE, label="American put")
    ax.set_xlabel("Spot price $S$"); ax.set_ylabel("Option value")
    ax.set_title("The right to exercise early is worth most when deep in the money")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, out_dir, "04_american_put.png")


def plot_hedge_error(n_steps, sds, ses, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    n = np.asarray(n_steps, dtype=float)
    ref = sds[0] * np.sqrt(n[0] / n)
    ax.plot(n, ref, color=GREY, linestyle=(0, (4, 3)), linewidth=1.4, label="Reference slope $1/\\sqrt{n}$")
    ax.plot(n, sds, marker="o", markersize=7, color=BLUE, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="Simulated SD of hedging P&L")
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [0.3, 0.5, 1, 2, 3]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xticks([5, 10, 21, 63, 126, 252, 504]); ax.set_xticklabels(["5", "10", "21", "63", "126", "252", "504"])
    ax.set_xlabel("Rebalances over the life of the option $n$")
    ax.set_ylabel("Standard deviation of final P&L")
    ax.set_title("Hedging error falls like $1/\\sqrt{n}$ as you rebalance more often")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, out_dir, "05_hedge_error_vs_frequency.png")


def plot_vol_mismatch(pnl_by_vol, theory_by_vol, sigma_imp, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [AQUA, BLUE, ORANGE]
    lo = min(np.percentile(v, 0.2) for v in pnl_by_vol.values())
    hi = max(np.percentile(v, 99.8) for v in pnl_by_vol.values())
    bins = np.linspace(lo, hi, 70)
    for (sig, pnl), c in zip(pnl_by_vol.items(), colors):
        ax.hist(pnl, bins=bins, histtype="stepfilled", alpha=0.3, color=c)
        ax.hist(pnl, bins=bins, histtype="step", color=c, linewidth=1.8,
                label=f"Realised vol {sig:.0%} (mean {pnl.mean():+.2f}, theory {theory_by_vol[sig]:+.2f})")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Seller's final P&L")
    ax.set_ylabel("Number of simulated paths")
    ax.set_title(f"Selling volatility at {sigma_imp:.0%}: you win only if realised volatility is lower")
    ax.legend(loc="upper left", fontsize=8.5)
    return _save(fig, out_dir, "06_vol_mismatch_pnl.png")
