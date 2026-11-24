"""Unit tests on synthetic data (no Bloomberg data needed): python -m pytest tests/"""

import numpy as np
import pandas as pd
import pytest

from mts import baseline as BL
from mts import data, optimize, risk, signals

rng = np.random.default_rng(0)


def random_cov(n, seed=1):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n)) * 0.1
    return A @ A.T + np.diag(r.uniform(0.02, 0.08, n))


# -- Black-Litterman ----------------------------------------------------------------

def test_implied_returns_price_the_market_at_the_risk_premium():
    Sigma = random_cov(6)
    w = np.full(6, 1 / 6)
    pi = BL.implied_returns(Sigma, w, mrp=0.05)
    assert w @ pi == pytest.approx(0.05)


def test_posterior_equals_prior_when_views_agree_with_it():
    Sigma = random_cov(5)
    pi = BL.implied_returns(Sigma, np.full(5, 0.2))
    mu, _ = BL.bl_posterior(Sigma, pi, Q=pi.copy(), tau=0.05)
    np.testing.assert_allclose(mu, pi)


def test_tau_cancels_from_the_posterior_mean_with_he_litterman_omega():
    # Omega = diag(tau Sigma) scales with tau, so the posterior mean does not depend on tau
    Sigma = random_cov(5)
    pi = BL.implied_returns(Sigma, np.full(5, 0.2))
    Q = pi + rng.normal(0, 0.05, 5)
    mu1, _ = BL.bl_posterior(Sigma, pi, Q, tau=0.01)
    mu2, _ = BL.bl_posterior(Sigma, pi, Q, tau=0.5)
    np.testing.assert_allclose(mu1, mu2)


def test_posterior_covariance_matches_the_bayesian_formula():
    Sigma, tau = random_cov(4), 0.05
    pi = BL.implied_returns(Sigma, np.full(4, 0.25))
    _, Sigma_bl = BL.bl_posterior(Sigma, pi, pi + 0.01, tau)
    Omega = np.diag(np.diag(tau * Sigma))
    M = np.linalg.inv(np.linalg.inv(tau * Sigma) + np.linalg.inv(Omega))
    np.testing.assert_allclose(Sigma_bl, Sigma + M)


def test_max_sharpe_matches_the_tangency_portfolio_when_unconstrained():
    Sigma = random_cov(4)
    mu = np.array([0.05, 0.07, 0.06, 0.08])
    w = BL.max_sharpe(mu, Sigma, max_weight=1.0)
    tangency = np.linalg.solve(Sigma, mu)
    tangency /= tangency.sum()
    assert (tangency > 0).all()
    sharpe = lambda x: x @ mu / np.sqrt(x @ Sigma @ x)
    assert sharpe(w) == pytest.approx(sharpe(tangency), rel=1e-6)   # SLSQP's default tolerance
    np.testing.assert_allclose(w, tangency, atol=1e-3)


def test_max_sharpe_respects_the_weight_cap():
    Sigma = random_cov(8)
    w = BL.max_sharpe(rng.uniform(0.02, 0.1, 8), Sigma, max_weight=0.2)
    assert w.sum() == pytest.approx(1) and w.max() <= 0.2 + 1e-6 and w.min() >= 0


def test_arima_view_mean_exceeds_the_median_by_the_variance_term():
    m, s = 0.02, 0.1
    lo, hi = m - 1.959964 * s, m + 1.959964 * s
    assert BL.arima_view_mean(m, lo, hi) == pytest.approx(np.exp(4 * m + 2 * s ** 2) - 1)
    assert BL.arima_view_mean(m, lo, hi) > np.exp(4 * m) - 1


# -- Signals and alphas ---------------------------------------------------------------

def test_winsor_z_is_standardised():
    z = signals.winsor_z(pd.Series(rng.normal(size=500)))
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)


def test_grinold_alpha_scale_and_zero_ic():
    scores = pd.Series(rng.normal(size=200), index=[f"s{i}" for i in range(200)])
    vol = pd.Series(0.3, index=scores.index)
    a = signals.grinold_alpha(scores, vol, ic=0.05)
    assert a.std() == pytest.approx(2 * 0.05 * 0.3, rel=1e-9)
    assert (signals.grinold_alpha(scores, vol, ic=0.0) == 0).all()


def test_point_in_time_ic_uses_only_realised_formations():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    ic = pd.Series(np.arange(40, dtype=float), index=idx)
    est = signals.PointInTimeIC(ic, horizon_months=3, min_obs=5)
    f = idx[20]
    # formations up to idx[17] (3 months before f) are realised: mean of 0..17
    assert est(f) == pytest.approx(np.mean(np.arange(18)))
    assert np.isnan(signals.PointInTimeIC(ic, 3, min_obs=30)(f))


def test_transfer_coefficient_is_one_for_the_unconstrained_optimum():
    t = [f"s{i}" for i in range(30)]
    vol = pd.Series(rng.uniform(0.2, 0.4, 30), index=t)
    alpha = pd.Series(rng.normal(0, 0.02, 30), index=t)
    bench = pd.Series(1 / 30, index=t)
    weights = bench + 0.5 * alpha / vol ** 2          # diagonal-risk optimum
    assert signals.transfer_coefficient(weights, bench, alpha, vol) == pytest.approx(1)


# -- Risk model ------------------------------------------------------------------------

def simulated_returns(n=40, T=2000, k=2, seed=3):
    r = np.random.default_rng(seed)
    B = r.normal(1, 0.3, size=(n, k)) * [1, 0.5][:k]
    f = r.normal(0, 0.01, size=(T, k))
    spec_sd = r.uniform(0.005, 0.015, n)
    e = r.normal(size=(T, n)) * spec_sd
    true_cov = (B @ np.diag([0.01 ** 2] * k) @ B.T + np.diag(spec_sd ** 2)) * 252
    rets = pd.DataFrame(f @ B.T + e, columns=[f"s{i}" for i in range(n)])
    rets.attrs["true_cov"] = true_cov
    return rets, spec_sd


def test_pca_risk_model_recovers_specific_risk():
    rets, spec_sd = simulated_returns()
    rm = risk.fit_pca(rets, n_factors=2)
    est = rm.specific_vol.values / np.sqrt(252)
    assert np.median(np.abs(est / spec_sd - 1)) < 0.1


def test_risk_model_covariance_is_close_to_the_true_covariance():
    rets, _ = simulated_returns()
    rm = risk.fit_pca(rets, n_factors=2)
    true = rets.attrs["true_cov"]
    assert np.linalg.norm(rm.cov - true) / np.linalg.norm(true) < 0.08


def test_factor_model_beats_the_sample_covariance_when_history_is_short():
    # 80 stocks, 120 days: the sample covariance is noisy, the factor structure is not
    rets, _ = simulated_returns(n=80, T=120, seed=9)
    true = rets.attrs["true_cov"]
    err = lambda S: np.linalg.norm(S - true) / np.linalg.norm(true)
    assert err(risk.fit_pca(rets, n_factors=2).cov) < err(rets.cov().values * 252)


def test_decomposition_adds_up_to_tracking_error():
    rets, _ = simulated_returns()
    rm = risk.fit_pca(rets, n_factors=2)
    act = pd.Series(rng.normal(0, 0.02, 40), index=rets.columns)
    d = rm.decompose(act)
    assert d["factor variance"] + d["specific variance"] == pytest.approx(rm.tracking_error(act) ** 2)


# -- Optimiser ----------------------------------------------------------------------------

def setup_opt(n=20, seed=4):
    t = [f"s{i}" for i in range(n)]
    Sigma = random_cov(n, seed)
    bench = pd.Series(np.random.default_rng(seed).uniform(1, 3, n), index=t)
    return t, Sigma, bench / bench.sum()


def test_zero_alpha_holds_the_benchmark():
    t, Sigma, b = setup_opt()
    w = optimize.optimize_active(pd.Series(0.0, index=t), Sigma, b, risk_aversion=5)
    np.testing.assert_allclose(w.values, b.values, atol=1e-4)


def test_tracking_error_cap_is_respected():
    t, Sigma, b = setup_opt()
    alpha = pd.Series(np.random.default_rng(5).normal(0, 0.05, len(t)), index=t)
    w = optimize.optimize_active(alpha, Sigma, b, te_max=0.03, max_weight=1.0)
    act = (w - b).values
    assert np.sqrt(act @ Sigma @ act) <= 0.03 + 1e-4
    assert w.sum() == pytest.approx(1) and w.min() >= -1e-9


def test_high_trading_cost_keeps_current_holdings():
    t, Sigma, b = setup_opt()
    alpha = pd.Series(np.random.default_rng(6).normal(0, 0.01, len(t)), index=t)
    current = (b * pd.Series(np.random.default_rng(7).uniform(0.8, 1.2, len(t)), index=t))
    current /= current.sum()
    w = optimize.optimize_active(alpha, Sigma, b, w_current=current, risk_aversion=1, cost=0.05)
    np.testing.assert_allclose(w.values, current.values, atol=1e-4)


def test_risk_aversion_shrinks_active_risk():
    t, Sigma, b = setup_opt()
    alpha = pd.Series(np.random.default_rng(8).normal(0, 0.03, len(t)), index=t)
    te = lambda w: np.sqrt((w - b).values @ Sigma @ (w - b).values)
    low = optimize.optimize_active(alpha, Sigma, b, risk_aversion=2, max_weight=1.0)
    high = optimize.optimize_active(alpha, Sigma, b, risk_aversion=20, max_weight=1.0)
    assert te(high) < te(low)


# -- Data helpers ----------------------------------------------------------------------------

def test_snapshot_filenames_with_copy_suffix_parse():
    assert data.parse_snapshot_date("SPX as of Sep 30 20201.xlsx") == pd.Timestamp("2020-09-30")
    with pytest.raises(ValueError):
        data.parse_snapshot_date("constituents.xlsx")


# -- Phase 7 -------------------------------------------------------------------------------

def test_ewma_weights_halve_every_half_life_and_sum_to_one():
    w = risk.ewma_weights(100, half_life=10)
    assert w.sum() == pytest.approx(1)
    assert w[-11] / w[-1] == pytest.approx(0.5)
    np.testing.assert_allclose(risk.ewma_weights(5, None), np.full(5, 0.2))


def test_weighted_risk_model_approaches_the_equal_weighted_one_for_long_half_lives():
    rets, _ = simulated_returns()
    eq = risk.fit_pca(rets, n_factors=2)
    long_hl = risk.fit_pca(rets, n_factors=2, half_life=1e9)
    np.testing.assert_allclose(long_hl.cov, eq.cov, rtol=1e-4)


def test_ewma_risk_model_tracks_a_volatility_increase():
    r = np.random.default_rng(10)
    calm = r.normal(0, 0.01, size=(700, 20))
    wild = r.normal(0, 0.03, size=(56, 20))
    rets = pd.DataFrame(np.vstack([calm, wild]))
    eq, ew = risk.fit_pca(rets, 2), risk.fit_pca(rets, 2, half_life=21)
    assert ew.total_vol.mean() > 2 * eq.total_vol.mean()


def test_residual_momentum_ignores_market_moves_and_rewards_consistent_outperformance():
    r = np.random.default_rng(11)
    T = 36
    m = pd.Series(r.normal(0.01, 0.05, T))
    noise = r.normal(0, 0.02, (T, 3))
    beta_only = 1.2 * m + noise[:, 0]
    winner = 1.0 * m + noise[:, 1] + np.r_[np.zeros(T - 12), np.full(11, 0.03), 0]   # beats beta in months t-11..t-1
    loser = 0.8 * m + noise[:, 2] - np.r_[np.zeros(T - 12), np.full(11, 0.03), 0]
    R = pd.DataFrame({"beta_only": beta_only, "winner": winner, "loser": loser})
    s = signals.residual_momentum(R, m)
    assert s["winner"] > s["beta_only"] > s["loser"]


def test_residual_momentum_is_centred_on_zero_for_pure_market_exposure():
    # A strongly rising market in the formation window: raw momentum would favour high betas,
    # residual momentum should not. Each score is sum/sd of 11 residuals, so its sd is about sqrt(11).
    r = np.random.default_rng(12)
    T, n = 36, 500
    m = pd.Series(np.r_[r.normal(0, 0.04, T - 12), r.normal(0.05, 0.04, 12)])
    betas = r.uniform(0.5, 1.5, n)
    R = pd.DataFrame(np.outer(m, betas) + r.normal(0, 0.03, (T, n)))
    s = signals.residual_momentum(R, m)
    assert abs(s.mean()) < 3 * np.sqrt(11) / np.sqrt(n)
    assert abs(np.corrcoef(s, betas)[0, 1]) < 0.15


def test_vol_scale_targets_volatility_and_respects_the_cap():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    calm = pd.Series(0.06 / np.sqrt(252), index=idx)             # constant |return| -> 6% realised
    assert signals.vol_scale(calm, idx[-1], target=0.12, cap=5) == pytest.approx(2.0)
    assert signals.vol_scale(calm, idx[-1], target=0.12, cap=1.5) == pytest.approx(1.5)


def test_residual_momentum_skips_months_without_a_market_return():
    r = np.random.default_rng(13)
    m = pd.Series(r.normal(0.01, 0.05, 36))
    R = pd.DataFrame(np.outer(m, [1.0, 1.2]) + r.normal(0, 0.02, (36, 2)), columns=["a", "b"])
    full = signals.residual_momentum(R, m)
    m_gap = m.copy()
    m_gap.iloc[0] = np.nan                       # e.g. the first month of the data
    gap = signals.residual_momentum(R, m_gap)
    assert gap.notna().all()
    np.testing.assert_allclose(gap, signals.residual_momentum(R.iloc[1:], m.iloc[1:]))
    assert full.notna().all()


def test_alpha_factor_captures_a_hidden_common_factor():
    # Returns have a factor along x that a 1-factor model misses; after adding x as a factor, the
    # forecast variance of a portfolio along x matches the sample variance.
    r = np.random.default_rng(14)
    n, T = 30, 1500
    mkt = r.normal(0, 0.01, T)
    x = pd.Series(r.normal(size=n), index=[f"s{i}" for i in range(n)])
    hidden = r.normal(0, 0.004, T)
    R = np.outer(mkt, np.ones(n)) + np.outer(hidden, x.values) + r.normal(0, 0.01, (T, n))
    rets = pd.DataFrame(R, columns=x.index)
    rm = risk.fit_pca(rets, n_factors=1)
    aug = risk.add_alpha_factor(rm, rets, x)
    xp = x.values - rm.B @ (rm.B.T @ x.values)
    u = xp / np.linalg.norm(xp)
    sample = np.var(R @ u, ddof=1) * 252
    assert u @ aug.cov @ u == pytest.approx(sample, rel=0.02)
    assert u @ rm.cov @ u < 0.8 * sample             # the unaligned model underestimates it
    assert aug.B.shape[1] == 2


def test_bias_statistic_is_one_when_forecasts_are_right():
    from mts import backtest as bt
    r = np.random.default_rng(15)
    dates = list(pd.date_range("2000-03-31", periods=400, freq="QE"))
    te = 0.04
    active = r.normal(0, te * np.sqrt(0.25), len(dates))
    bench = pd.Series(1.0, index=dates + [dates[-1] + pd.offsets.QuarterEnd()])
    curve = pd.Series(np.r_[1.0, np.cumprod(1 + active)], index=bench.index)
    # curve returns relative to a flat benchmark: per-period return = active (compounded levels)
    b = bt.bias_statistic(curve, bench, {d: te for d in dates}, dates)
    assert b["standardised"].std() == pytest.approx(1, abs=0.1)
