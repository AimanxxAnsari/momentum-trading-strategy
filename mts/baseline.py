"""The baseline strategy of main.ipynb: momentum selection, ARIMA views, Black-Litterman, max-Sharpe.

The Black-Litterman maths is in `bl_posterior` (a pure function, unit-tested); `black_litterman`
wraps it with the data. Results are identical to main.ipynb (checked in the notebooks)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from . import config as C
from . import data


# -- Momentum selection --------------------------------------------------------

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative strength index with Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(series: pd.Series):
    line = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    return line, line.ewm(span=9, adjust=False).mean()


def momentum_signals(date, tickers) -> pd.DataFrame:
    window = data.prices().loc[:date, tickers].iloc[-(C.MOM_LOOKBACK + 300):]
    rows = {}
    for t in tickers:
        s = window[t].dropna()
        if len(s) <= C.MOM_LOOKBACK:
            continue
        line, signal = macd(s)
        rows[t] = {"momentum_12_1": s.iloc[-1 - C.MOM_SKIP] / s.iloc[-1 - C.MOM_LOOKBACK] - 1,
                   "rsi": rsi(s).iloc[-1], "macd_hist": (line - signal).iloc[-1]}
    return pd.DataFrame(rows).T


def select_stocks(signals: pd.DataFrame, n_select: int, rsi_max: float) -> pd.DataFrame:
    eligible = signals[signals["rsi"] < rsi_max]
    return eligible.sort_values("momentum_12_1", ascending=False).head(n_select)


# -- Black-Litterman -----------------------------------------------------------

def covariance(rets: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """Annualised covariance of daily returns: Ledoit-Wolf shrinkage or the sample estimate."""
    if method == "ledoit_wolf":
        return LedoitWolf().fit(rets.dropna().values).covariance_ * C.TRADING_DAYS
    return rets.cov().values * C.TRADING_DAYS


def implied_returns(Sigma: np.ndarray, w_mkt: np.ndarray, mrp: float = C.MARKET_RISK_PREMIUM) -> np.ndarray:
    """Equilibrium excess returns pi = delta * Sigma * w_mkt, with delta = MRP / market variance."""
    delta = mrp / (w_mkt @ Sigma @ w_mkt)
    return delta * Sigma @ w_mkt


def bl_posterior(Sigma: np.ndarray, pi: np.ndarray, Q: np.ndarray, tau: float = 0.05):
    """Black-Litterman posterior for one absolute view per asset (P = I) with the He-Litterman
    choice Omega = diag(tau * Sigma). Returns (posterior mean, posterior covariance)."""
    n = len(pi)
    P = np.eye(n)
    tS = tau * Sigma
    Omega = np.diag(np.diag(P @ tS @ P.T))
    mu = pi + tS @ P.T @ np.linalg.solve(P @ tS @ P.T + Omega, Q - P @ pi)
    Sigma_bl = Sigma + np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(Omega) @ P)
    return mu, Sigma_bl


def max_sharpe(mu: np.ndarray, Sigma: np.ndarray, max_weight: float) -> np.ndarray:
    """Long-only, fully invested max-Sharpe weights with a per-asset cap (min-variance if no mu > 0)."""
    n = len(mu)
    if mu.max() > 0:
        objective = lambda w: -(w @ mu) / np.sqrt(w @ Sigma @ w)
    else:
        objective = lambda w: w @ Sigma @ w
    result = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                      bounds=[(0, max(max_weight, 1 / n))] * n,
                      constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def black_litterman(date, tickers, caps, views, tau=0.05, max_weight=0.20, cov="ledoit_wolf",
                    use_views=True, risk_free=None) -> pd.DataFrame:
    """main.ipynb's allocation: market-cap prior over `tickers`, ARIMA views (annual) minus the
    risk-free rate, then max-Sharpe. Tickers with under 80% of the covariance window are dropped.
    `risk_free`: annual rate subtracted from the views; None keeps main.ipynb's constant 3%."""
    rets = data.daily_returns(date, tickers)
    tickers = list(rets.columns)
    Sigma = covariance(rets, cov)
    w_mkt = (caps[tickers] / caps[tickers].sum()).values
    pi = implied_returns(Sigma, w_mkt)
    Q = np.array([views[t] for t in tickers]) - (C.RISK_FREE if risk_free is None else risk_free)
    if use_views:
        mu, Sigma_bl = bl_posterior(Sigma, pi, Q, tau)
    else:
        mu, Sigma_bl = pi, (1 + tau) * Sigma
    w = max_sharpe(mu, Sigma_bl, max_weight)
    return pd.DataFrame({"weight": w, "w_mkt": w_mkt, "prior_pi": pi, "view_Q": Q, "posterior_mu": mu,
                         "volatility": np.sqrt(np.diag(Sigma_bl))}, index=tickers)


def arima_cache() -> dict:
    """ARIMA forecasts computed by main.ipynb: {(date, ticker): {view_annual, fc_quarter, ci_low, ci_high, order}}."""
    return pickle.load(open(C.CACHE_DIR / "arima_views.pkl", "rb"))


def arima_view_mean(fc_quarter: float, ci_low: float, ci_high: float) -> float:
    """Expected (not median) annual return from a quarterly log-return forecast:
    exp(4 m + 2 s^2) - 1, with s the forecast standard error from its 95% interval."""
    s = (ci_high - ci_low) / (2 * 1.959964)
    return np.exp(4 * fc_quarter + 2 * s ** 2) - 1
