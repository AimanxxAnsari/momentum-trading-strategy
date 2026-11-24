"""Benchmark-relative portfolio optimiser (Phase 6).

Chooses weights w around benchmark weights b to maximise, over one holding period h (in years),

    h * alpha'(w - b)  -  h * lambda * (w - b)' Sigma (w - b)  -  cost * sum|w - w_current|

subject to: fully invested, long-only, w <= max_weight, and optionally a tracking-error cap
sqrt((w - b)' Sigma (w - b)) <= te_max. The absolute value is handled with buy and sell variables
(w - w_current = buy - sell, both >= 0), so the problem is smooth and SLSQP can solve it.

This is the Grinold-Kahn active utility: alpha is rewarded, active risk and trading are charged,
so weak alphas produce small, cheap tilts away from the benchmark instead of concentrated bets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_active(alpha: pd.Series, Sigma: np.ndarray, w_bench: pd.Series, w_current: pd.Series | None = None,
                    risk_aversion: float = 0.0, te_max: float | None = None, cost: float = 0.0,
                    max_weight: float = 0.20, horizon: float = 0.25) -> pd.Series:
    """alpha (annual), Sigma (annual) and w_bench are aligned on the same tickers (the universe).
    `cost` is the cost per unit traded (e.g. 0.001 for 10 bps). Holdings outside the universe are
    assumed sold. Returns weights on the universe, summing to 1."""
    tickers = list(alpha.index)
    n = len(tickers)
    a = alpha.values
    b = w_bench.reindex(tickers).fillna(0).values
    b = b / b.sum()
    w0 = np.zeros(n) if w_current is None else w_current.reindex(tickers).fillna(0).values
    ub = max(max_weight, 1.0 / n)

    def split(x):
        return x[:n], x[n:2 * n], x[2 * n:]

    def objective(x):
        w, buy, sell = split(x)
        act = w - b
        return -(horizon * a @ act - horizon * risk_aversion * act @ Sigma @ act - cost * (buy.sum() + sell.sum()))

    def gradient(x):
        w, _, _ = split(x)
        act = w - b
        g_w = -(horizon * a - 2 * horizon * risk_aversion * Sigma @ act)
        return np.concatenate([g_w, np.full(n, cost), np.full(n, cost)])

    I = np.eye(n)
    constraints = [
        {"type": "eq", "fun": lambda x: split(x)[0].sum() - 1,
         "jac": lambda x: np.concatenate([np.ones(n), np.zeros(2 * n)])},
        {"type": "eq", "fun": lambda x: split(x)[0] - w0 - split(x)[1] + split(x)[2],
         "jac": lambda x: np.hstack([I, -I, I])},
    ]
    if te_max is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda x: te_max ** 2 - (split(x)[0] - b) @ Sigma @ (split(x)[0] - b),
            "jac": lambda x: np.concatenate([-2 * Sigma @ (split(x)[0] - b), np.zeros(2 * n)])})

    w_start = np.clip(b, 0, ub)
    w_start /= w_start.sum()
    x0 = np.concatenate([w_start, np.clip(w_start - w0, 0, None), np.clip(w0 - w_start, 0, None)])
    bounds = [(0, ub)] * n + [(0, None)] * (2 * n)
    res = minimize(objective, x0, jac=gradient, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(split(res.x)[0], 0, None)
    w = pd.Series(w / w.sum(), index=tickers)
    w.attrs["success"], w.attrs["message"] = res.success, res.message
    return w


def max_sharpe_with_alpha(date, tickers, caps, alpha: pd.Series | None = None, max_weight: float = 0.20,
                          cov: str = "ledoit_wolf", tau: float = 0.05) -> pd.DataFrame:
    """Phase 5 allocation: equilibrium prior over all `tickers` plus alpha as the expected excess
    return (mu = pi + alpha), then the baseline's long-only max-Sharpe with a per-stock cap."""
    from . import baseline as BL
    from . import data
    rets = data.daily_returns(date, tickers)
    tickers = list(rets.columns)
    Sigma = BL.covariance(rets, cov)
    w_mkt = (caps[tickers] / caps[tickers].sum()).values
    pi = BL.implied_returns(Sigma, w_mkt)
    a = np.zeros(len(tickers)) if alpha is None else alpha.reindex(tickers).fillna(0).values
    w = BL.max_sharpe(pi + a, (1 + tau) * Sigma, max_weight)
    return pd.DataFrame({"weight": w, "w_mkt": w_mkt, "prior_pi": pi, "alpha": a, "mu": pi + a,
                         "vol": np.sqrt(np.diag(Sigma))}, index=tickers)
