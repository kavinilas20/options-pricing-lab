"""
main.py -- runs every experiment, writes tables, figures and run_summary.json.

    python main.py            # about 1-2 minutes
    python main.py --seed 7

Experiments
    A. Black-Scholes sanity: textbook values, put-call parity, implied-vol round trip.
    B. Binomial convergence to Black-Scholes, and the American put premium.
    C. Monte Carlo: agreement with Black-Scholes, confidence-interval coverage,
       variance reduction at equal cost, pathwise delta vs analytic delta.
    D. Delta hedging: error vs rebalancing frequency, and P&L when the volatility
       you sold differs from the volatility that is realised.

Every number the README quotes comes from results/tables/run_summary.json.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src import binomial as bi
from src import black_scholes as bs
from src import hedging as hg
from src import monte_carlo as mc
from src import visualization as viz

FIG_DIR, TABLE_DIR = "results/figures", "results/tables"
S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def save(df, name):
    df.to_csv(os.path.join(TABLE_DIR, name), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    summary = {"seed": seed, "setup": {"S0": S0, "K": K, "r": R, "sigma": SIG, "T": T}}

    # ------------------------------------------------------------------
    section("A. Black-Scholes reference values")
    call, put = float(bs.price(S0, K, R, SIG, T, "call")), float(bs.price(S0, K, R, SIG, T, "put"))
    parity_gap = call - put - (S0 - K * math.exp(-R * T))
    iv = bs.implied_vol(call, S0, K, R, T, "call")
    g = {k: float(v) for k, v in bs.greeks(S0, K, R, SIG, T, "call").items()}
    print(f"call {call:.4f}  put {put:.4f}  parity gap {parity_gap:.2e}  implied vol recovered {iv:.10f}")
    print("call greeks:", {k: round(v, 4) for k, v in g.items()})
    summary["black_scholes"] = {"call": call, "put": put, "parity_gap": parity_gap,
                                "implied_vol_round_trip": iv, "call_greeks": g}

    spots = np.linspace(60, 140, 161)
    viz.plot_greeks(spots, bs.greeks(spots, K, R, SIG, T, "call"), K, FIG_DIR)

    # ------------------------------------------------------------------
    section("B. Binomial tree")
    ns = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    errs = {}
    rows = []
    for label, strike in (("At the money (K=100)", 100.0), ("Out of the money (K=120)", 120.0)):
        exact = float(bs.price(S0, strike, R, SIG, T, "call"))
        errs[label] = [abs(bi.binomial_price(S0, strike, R, SIG, T, n, "call") - exact) for n in ns]
        for n, e in zip(ns, errs[label]):
            rows.append({"strike": strike, "n_steps": n, "abs_error": e})
    conv = pd.DataFrame(rows)
    print(conv.pivot(index="n_steps", columns="strike", values="abs_error").map(lambda v: f"{v:.2e}").to_string())
    save(conv, "binomial_convergence.csv")
    viz.plot_binomial_convergence(ns, errs, FIG_DIR)

    am_rows = []
    spots_am = np.linspace(50, 130, 41)
    eu_curve, am_curve = [], []
    for s in spots_am:
        eu_curve.append(bi.binomial_price(s, K, R, SIG, T, 1000, "put"))
        am_curve.append(bi.binomial_price(s, K, R, SIG, T, 1000, "put", american=True))
    viz.plot_american_put(spots_am, np.array(am_curve), np.array(eu_curve), K, FIG_DIR)
    for s in (80.0, 90.0, 100.0, 110.0):
        eu = bi.binomial_price(s, K, R, SIG, T, 2000, "put")
        am = bi.binomial_price(s, K, R, SIG, T, 2000, "put", american=True)
        am_rows.append({"spot": s, "european_put": eu, "american_put": am, "early_exercise_premium": am - eu})
    am_tab = pd.DataFrame(am_rows)
    print("\nAmerican vs European put (2000 steps):\n" + am_tab.round(4).to_string(index=False))
    save(am_tab, "american_put.csv")
    summary["binomial"] = {"abs_error_atm_n2560": errs["At the money (K=100)"][-1],
                           "american_put_atm": float(am_tab.loc[am_tab.spot == 100.0, "american_put"].iloc[0]),
                           "european_put_atm_binomial": float(am_tab.loc[am_tab.spot == 100.0, "european_put"].iloc[0])}

    # ------------------------------------------------------------------
    section("C. Monte Carlo")
    methods = [("Plain", False, False), ("Antithetic", True, False),
               ("Control variate", False, True), ("Antithetic + control variate", True, True)]
    n_big = 100_000
    vr_rows, ses = [], []
    for label, anti, cv in methods:
        se_runs, est_runs = [], []
        for i in range(20):                               # average SE over 20 independent runs
            est, se = mc.mc_european(S0, K, R, SIG, T, "call", n_big, seed=seed * 100 + i, antithetic=anti, control_variate=cv)
            est_runs.append(est); se_runs.append(se)
        vr_rows.append({"method": label, "mean_estimate": float(np.mean(est_runs)), "mean_se": float(np.mean(se_runs))})
        ses.append(float(np.mean(se_runs)))
    vr = pd.DataFrame(vr_rows)
    vr["variance_ratio_vs_plain"] = (vr["mean_se"] / vr.loc[0, "mean_se"]) ** 2
    print(vr.round(5).to_string(index=False))
    save(vr, "mc_variance_reduction.csv")
    viz.plot_variance_reduction([m[0] for m in methods], ses, FIG_DIR)

    cov_rows = []
    reps = 2000
    for label, anti, cv in methods:
        hits = 0
        for i in range(reps):
            est, se = mc.mc_european(S0, K, R, SIG, T, "call", 2000, seed=seed * 10_000 + i, antithetic=anti, control_variate=cv)
            hits += abs(est - call) < 1.96 * se
        cov_rows.append({"method": label, "coverage_of_95pct_ci": hits / reps, "n_repetitions": reps,
                         "coverage_sd_if_nominal": math.sqrt(0.95 * 0.05 / reps)})
    cov = pd.DataFrame(cov_rows)
    print("\nConfidence-interval coverage (n = 2000 evaluations per run):\n" + cov.round(4).to_string(index=False))
    save(cov, "mc_coverage.csv")

    dl_rows = []
    for strike in (90.0, 100.0, 110.0):
        for kind in ("call", "put"):
            exact = float(bs.greeks(S0, strike, R, SIG, T, kind)["delta"])
            est, se = mc.mc_delta_pathwise(S0, strike, R, SIG, T, kind, 400_000, seed=seed)
            dl_rows.append({"strike": strike, "kind": kind, "analytic_delta": exact, "mc_delta": est, "se": se,
                            "z": (est - exact) / se})
    dl = pd.DataFrame(dl_rows)
    print("\nPathwise Monte Carlo delta vs analytic delta:\n" + dl.round(4).to_string(index=False))
    save(dl, "mc_delta.csv")
    summary["monte_carlo"] = {
        "variance_ratio_vs_plain": dict(zip(vr["method"], vr["variance_ratio_vs_plain"])),
        "ci_coverage": dict(zip(cov["method"], cov["coverage_of_95pct_ci"])),
        "max_abs_z_delta": float(dl["z"].abs().max()),
    }

    # ------------------------------------------------------------------
    section("D. Delta hedging")
    n_paths = 20_000
    steps = [5, 10, 21, 42, 63, 126, 252, 504]
    h_rows = []
    for n in steps:
        pnl = hg.simulate_delta_hedge(S0, K, R, SIG, SIG, T, n, n_paths, seed=seed + n)
        h_rows.append({"n_rebalances": n, "mean_pnl": pnl.mean(), "se_mean": pnl.std(ddof=1) / math.sqrt(n_paths),
                       "sd_pnl": pnl.std(ddof=1)})
    hd = pd.DataFrame(h_rows)
    slope, intercept = np.polyfit(np.log(hd["n_rebalances"]), np.log(hd["sd_pnl"]), 1)
    naked = hg.simulate_delta_hedge(S0, K, R, SIG, SIG, T, 252, n_paths, seed=seed, hedge=False)
    print(hd.round(4).to_string(index=False))
    print(f"log-log slope of SD vs n: {slope:.3f} (theory: -0.5);  unhedged SD: {naked.std(ddof=1):.2f}")
    save(hd, "hedge_error_vs_frequency.csv")
    viz.plot_hedge_error(hd["n_rebalances"], hd["sd_pnl"].to_numpy(), hd["se_mean"].to_numpy(), FIG_DIR)

    mm_rows, pnl_by_vol, theory_by_vol = [], {}, {}
    for sig_real in (0.10, 0.15, 0.20, 0.25, 0.30):
        pnl = hg.simulate_delta_hedge(S0, K, R, SIG, sig_real, T, 252, n_paths, seed=seed + 1)
        th = hg.theory_mean_pnl(S0, K, R, SIG, sig_real, T)
        se = pnl.std(ddof=1) / math.sqrt(n_paths)
        mm_rows.append({"sigma_real": sig_real, "mean_pnl": pnl.mean(), "se": se, "theory_continuous": th,
                        "z_vs_theory": (pnl.mean() - th) / se, "sd_pnl": pnl.std(ddof=1)})
        if sig_real in (0.15, 0.20, 0.30):
            pnl_by_vol[sig_real], theory_by_vol[sig_real] = pnl, th
    mm = pd.DataFrame(mm_rows)
    print("\nSold at 20% vol, daily hedged (252 rebalances), different realised volatility:")
    print(mm.round(4).to_string(index=False))
    save(mm, "hedge_vol_mismatch.csv")
    viz.plot_vol_mismatch(pnl_by_vol, theory_by_vol, SIG, FIG_DIR)
    summary["hedging"] = {
        "loglog_slope_sd_vs_n": float(slope), "unhedged_sd_252": float(naked.std(ddof=1)),
        "sd_by_n": dict(zip(hd["n_rebalances"].astype(int).astype(str), hd["sd_pnl"])),
        "mismatch": {f"{r.sigma_real:.2f}": {"mean": r.mean_pnl, "theory": r.theory_continuous, "z": r.z_vs_theory}
                     for r in mm.itertuples()},
    }

    with open(os.path.join(TABLE_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    section("DONE")
    print(f"figures -> {FIG_DIR}/   tables -> {TABLE_DIR}/")


if __name__ == "__main__":
    main()
