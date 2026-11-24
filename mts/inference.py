"""Statistical inference for backtests (Phase 8).

* factor_regression: alpha and factor loadings with Newey-West standard errors
* sharpe: annualised Sharpe ratio from daily returns and a daily risk-free series
* sharpe_difference_bootstrap: circular block bootstrap of the difference in two Sharpe ratios
* probabilistic_sharpe / expected_max_sharpe / deflated_sharpe: Bailey & Lopez de Prado (2012, 2014)
* pbo_cscv: probability of backtest overfitting by combinatorially symmetric cross-validation
  (Bailey, Borwein, Lopez de Prado & Zhu 2017)

Sharpe ratios inside the Bailey & Lopez de Prado formulas are per period (daily, not annualised).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config as C

EULER_GAMMA = 0.5772156649015329


def factor_regression(excess: pd.Series, factors: pd.DataFrame, lags: int = 5) -> dict:
    """OLS of daily excess returns on factor returns, Newey-West (HAC) standard errors.
    Returns annualised alpha, its t-statistic, each loading with its t-statistic, R-squared, days."""
    df = pd.concat([excess.rename("y"), factors], axis=1, join="inner").dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df[factors.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha (annual)": fit.params["const"] * C.TRADING_DAYS, "alpha t": fit.tvalues["const"]}
    for f in factors.columns:
        out[f], out[f"{f} t"] = fit.params[f], fit.tvalues[f]
    out["R2"], out["days"] = fit.rsquared, int(fit.nobs)
    return out


def sharpe(returns: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = returns - (rf.reindex(returns.index) if isinstance(rf, pd.Series) else rf)
    return ex.mean() / ex.std() * np.sqrt(C.TRADING_DAYS)


def sharpe_difference_bootstrap(a: pd.Series, b: pd.Series, rf: pd.Series | float = 0.0, block: int = 21,
                                n_boot: int = 2000, seed: int = 0) -> dict:
    """Sharpe(a) - Sharpe(b) with a circular block bootstrap of the paired daily returns, which keeps
    the correlation between the two series and short-range dependence within blocks.
    p-value: two-sided, from the bootstrap distribution re-centred on zero."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    r = rf.reindex(df.index).values if isinstance(rf, pd.Series) else np.full(len(df), rf)
    X = df.values - r[:, None]
    T = len(X)
    sr = lambda Y: Y.mean(axis=0) / Y.std(axis=0, ddof=1) * np.sqrt(C.TRADING_DAYS)
    est = sr(X)
    diff = est[0] - est[1]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = sr(X[idx])
        diffs[k] = s[0] - s[1]
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(diff)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"Sharpe a": est[0], "Sharpe b": est[1], "difference": diff, "ci low": lo, "ci high": hi, "p": p,
            "days": T, "block": block, "replications": n_boot}


def probabilistic_sharpe(sr: float, T: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given an observed per-period Sharpe `sr` over T periods, allowing
    for non-normal returns (`kurt` is the ordinary kurtosis, 3 for a normal distribution)."""
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials per-period Sharpe ratios when all true Sharpes are zero and the
    estimates have variance var_sr (the false-strategy theorem)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, trial_sharpes: np.ndarray) -> dict:
    """Deflated Sharpe ratio of the selected strategy: the probabilistic Sharpe ratio against the
    Sharpe one would expect from the best of all the trials by luck alone.
    `trial_sharpes` are the per-period Sharpe ratios of every trial, including the selected one."""
    r = returns.dropna()
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(len(trial_sharpes), np.var(trial_sharpes, ddof=1))
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    return {"Sharpe (annual)": sr * np.sqrt(C.TRADING_DAYS), "trials": len(trial_sharpes),
            "expected max Sharpe by luck (annual)": sr0 * np.sqrt(C.TRADING_DAYS),
            "PSR vs 0": probabilistic_sharpe(sr, len(r), skew, kurt, 0.0),
            "deflated Sharpe (PSR vs expected max)": probabilistic_sharpe(sr, len(r), skew, kurt, sr0),
            "skewness": skew, "kurtosis": kurt, "days": len(r)}


def pbo_cscv(returns: pd.DataFrame, n_splits: int = 16) -> dict:
    """Probability of backtest overfitting (CSCV). The T x N matrix of trial returns is cut into
    n_splits blocks; for every way of choosing half the blocks as in-sample, the trial with the best
    in-sample Sharpe is found and its rank out of sample recorded. PBO is the share of splits in which
    that trial ranks below the out-of-sample median (logit < 0)."""
    X = returns.dropna().values
    T, N = X.shape
    edges = np.linspace(0, T, n_splits + 1).astype(int)
    sums = np.array([X[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    sqs = np.array([(X[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges)

    def block_sharpe(blocks):
        n = cnt[list(blocks)].sum()
        s, q = sums[list(blocks)].sum(axis=0), sqs[list(blocks)].sum(axis=0)
        mean = s / n
        return mean / np.sqrt(q / n - mean ** 2)

    logits, degradation = [], []
    all_blocks = set(range(n_splits))
    for ins in combinations(range(n_splits), n_splits // 2):
        oos = tuple(all_blocks - set(ins))
        sr_in, sr_out = block_sharpe(ins), block_sharpe(oos)
        best = int(np.argmax(sr_in))
        rank = stats.rankdata(sr_out)[best] / (N + 1)          # relative rank in (0, 1)
        logits.append(np.log(rank / (1 - rank)))
        degradation.append((sr_in[best], sr_out[best]))
    logits = np.array(logits)
    d = np.array(degradation) * np.sqrt(C.TRADING_DAYS)
    return {"PBO": float(np.mean(logits < 0)), "splits": len(logits), "trials": N,
            "median logit": float(np.median(logits)),
            "best in-sample Sharpe, in sample (mean)": float(d[:, 0].mean()),
            "best in-sample Sharpe, out of sample (mean)": float(d[:, 1].mean()),
            "logits": logits}
