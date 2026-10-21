"""Delta-hedging simulation: the facts documented in src/hedging.py."""

import numpy as np
import pytest

from src import hedging as hg

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_matched_vol_hedge_has_mean_zero_and_error_shrinks_with_frequency():
    sds = []
    for n in (10, 40, 160):
        pnl = hg.simulate_delta_hedge(S, K, R, SIG, SIG, T, n, 20_000, seed=1)
        assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl))
        sds.append(pnl.std())
    assert sds[0] > sds[1] > sds[2]
    # four times the rebalances -> error roughly halves (1/sqrt(n))
    assert sds[0] / sds[1] == pytest.approx(2.0, rel=0.15)
    assert sds[1] / sds[2] == pytest.approx(2.0, rel=0.15)


def test_unhedged_seller_has_far_larger_risk_than_hedged_seller():
    hedged = hg.simulate_delta_hedge(S, K, R, SIG, SIG, T, 100, 10_000, seed=2)
    naked = hg.simulate_delta_hedge(S, K, R, SIG, SIG, T, 100, 10_000, seed=2, hedge=False)
    assert naked.std() > 8 * hedged.std()


@pytest.mark.parametrize("sigma_real", [0.15, 0.3])
def test_mismatched_vol_mean_pnl_matches_continuous_hedging_theory(sigma_real):
    pnl = hg.simulate_delta_hedge(S, K, R, SIG, sigma_real, T, 252, 20_000, seed=4)
    theory = hg.theory_mean_pnl(S, K, R, SIG, sigma_real, T)
    se = pnl.std() / np.sqrt(len(pnl))
    # discrete (daily) hedging adds a small bias on top of the continuous theory
    assert abs(pnl.mean() - theory) < 4 * se + 0.05
    assert np.sign(theory) == np.sign(SIG - sigma_real)         # seller wins if realised vol < sold vol


def test_theory_is_zero_when_vols_match_and_put_call_agree():
    assert hg.theory_mean_pnl(S, K, R, SIG, SIG, T) == pytest.approx(0.0, abs=1e-12)
    c = hg.theory_mean_pnl(S, K, R, SIG, 0.3, T, kind="call")
    p = hg.theory_mean_pnl(S, K, R, SIG, 0.3, T, kind="put")
    assert c == pytest.approx(p, rel=1e-9)                       # same gamma -> same P&L


def test_same_seed_same_result():
    a = hg.simulate_delta_hedge(S, K, R, SIG, SIG, T, 20, 500, seed=9)
    assert np.array_equal(a, hg.simulate_delta_hedge(S, K, R, SIG, SIG, T, 20, 500, seed=9))
