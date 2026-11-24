"""Shared code for the signal research notebook (Phase 5).

Data loading, cleaning, the point-in-time universe, the Black-Litterman allocation and the
backtest engine are copied from main.ipynb without changes, so that the baseline strategy can
be reproduced exactly. New here: monthly signals for the full S&P 500, forward returns,
information coefficients, quintile portfolios, Fama-MacBeth regressions, and Grinold alphas.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

DATA_PATH = "tickers/spx_companies_historical_data.csv"
SNAPSHOT_DIR = "tickers"
CACHE_DIR = "cache"

# Settings from main.ipynb (unchanged)
UNIVERSE_SIZE = 50
MOM_LOOKBACK = 252
MOM_SKIP = 21
RISK_FREE = 0.03
MARKET_RISK_PREMIUM = 0.05
COV_LOOKBACK = 756
COST_BPS = 10
BASE = dict(n_select=25, rsi_max=70, max_weight=0.20, tau=0.05, cov="ledoit_wolf", use_views=True)

# New in Phase 5
BOOK_LAG_DAYS = 63   # book value is stamped at fiscal period end, before it is published: lag ~3 months


# ============================================================================
# Data (as main.ipynb sections 1-3)
# ============================================================================

def parse_snapshot_date(filename):
    match = re.search(r"as of (\w{3} \d{2} \d{4})", filename)
    if not match:
        raise ValueError(f"Unable to parse date from filename: {filename}")
    return pd.Timestamp(datetime.strptime(match.group(1), "%b %d %Y"))


def load_snapshots(folder=SNAPSHOT_DIR):
    import openpyxl
    snapshots = {}
    for path in glob.glob(os.path.join(folder, "*.xlsx")):
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(sheet.iter_rows(values_only=True))
        col = rows[0].index("Ticker")
        snapshots[parse_snapshot_date(os.path.basename(path))] = {r[col] for r in rows[1:] if r[col]}
    return dict(sorted(snapshots.items()))


def load_data():
    """Cleaned wide tables (date x ticker): prices, market caps, book value per share, volume,
    plus the constituent snapshots. Cleaning is identical to main.ipynb; the result is cached."""
    path = os.path.join(CACHE_DIR, "clean_data.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    snapshot_listings = load_snapshots()
    snapshots = {d: {l.split(" ")[0] for l in s} for d, s in snapshot_listings.items()}
    snapshot_dates = np.array(list(snapshots), dtype="datetime64[ns]")

    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = raw.dropna(subset=["PX_LAST"]).copy()
    df["listing"] = df["ticker"]
    df["ticker"] = df["listing"].str.split(" ").str[0]
    pos = np.searchsorted(snapshot_dates, df["date"].values, side="right") - 1
    df["snapshot"] = snapshot_dates[np.clip(pos, 0, None)]
    named = pd.DataFrame([(d, l) for d, ls in snapshot_listings.items() for l in ls], columns=["snapshot", "listing"])
    named["in_snapshot"] = True
    df = df.merge(named, on=["snapshot", "listing"], how="left")
    df["in_snapshot"] = df["in_snapshot"].fillna(False).astype(bool)
    df["history"] = df.groupby("listing")["date"].transform("size")
    df = (df.sort_values(["ticker", "date", "in_snapshot", "history"], ascending=[True, True, False, False])
            .drop_duplicates(["ticker", "date"]))
    data = {
        "prices": df.pivot(index="date", columns="ticker", values="PX_LAST"),
        "mcap": df.pivot(index="date", columns="ticker", values="CUR_MKT_CAP"),
        "book": df.pivot(index="date", columns="ticker", values="BOOK_VAL_PER_SH"),
        "volume": df.pivot(index="date", columns="ticker", values="PX_VOLUME"),
        "snapshots": snapshots,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(data, open(path, "wb"))
    return data


D = load_data()
prices, mcap, book, snapshots = D["prices"], D["mcap"], D["book"], D["snapshots"]


def constituents(date):
    latest = max(d for d in snapshots if d <= date)
    return snapshots[latest]


def universe(date, size=UNIVERSE_SIZE):
    members = [t for t in constituents(date) if t in prices.columns]
    caps = mcap.loc[:date, members].iloc[-5:].ffill().iloc[-1]
    caps = caps[prices.loc[date, members].notna()].dropna()
    return caps.sort_values(ascending=False).head(size)


def period_ends(start, end, freq="Q"):
    """Last trading day of each period (Q or M), from the period before `start`, excluding the last day."""
    idx = prices.index
    ends = pd.Series(idx, index=idx).groupby(idx.to_period(freq)).max()
    last_day = prices.loc[:end].index[-1]
    first = pd.Timestamp(start).to_period(freq) - 1
    return [d for q, d in ends.items() if q >= first and d < last_day]


# ============================================================================
# Momentum selection, Black-Litterman, backtest (as main.ipynb sections 4-7)
# ============================================================================

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(series):
    line = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    return line, line.ewm(span=9, adjust=False).mean()


def momentum_signals(date, tickers):
    window = prices.loc[:date, tickers].iloc[-(MOM_LOOKBACK + 300):]
    rows = {}
    for t in tickers:
        s = window[t].dropna()
        if len(s) <= MOM_LOOKBACK:
            continue
        line, signal = macd(s)
        rows[t] = {"momentum_12_1": s.iloc[-1 - MOM_SKIP] / s.iloc[-1 - MOM_LOOKBACK] - 1,
                   "rsi": rsi(s).iloc[-1], "macd_hist": (line - signal).iloc[-1]}
    return pd.DataFrame(rows).T


def select_stocks(signals, n_select, rsi_max):
    eligible = signals[signals["rsi"] < rsi_max]
    return eligible.sort_values("momentum_12_1", ascending=False).head(n_select)


def covariance(rets, method):
    if method == "ledoit_wolf":
        return LedoitWolf().fit(rets.dropna().values).covariance_ * 252
    return rets.cov().values * 252


def _max_sharpe(mu, Sigma_bl, max_weight):
    n = len(mu)
    if mu.max() > 0:
        objective = lambda w: -(w @ mu) / np.sqrt(w @ Sigma_bl @ w)
    else:
        objective = lambda w: w @ Sigma_bl @ w
    result = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                      bounds=[(0, max(max_weight, 1 / n))] * n,
                      constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def black_litterman(date, tickers, caps, views, tau=0.05, max_weight=0.20, cov="ledoit_wolf", use_views=True):
    """main.ipynb's allocation, unchanged: ARIMA views (annual) minus the risk-free rate, He-Litterman Omega."""
    rets = prices.loc[:date, tickers].iloc[-COV_LOOKBACK:].pct_change(fill_method=None).iloc[1:]
    tickers = list(rets.columns[rets.notna().mean() > 0.8])
    rets = rets[tickers]
    Sigma = covariance(rets, cov)
    w_mkt = (caps[tickers] / caps[tickers].sum()).values
    delta = MARKET_RISK_PREMIUM / (w_mkt @ Sigma @ w_mkt)
    pi = delta * Sigma @ w_mkt
    Q = np.array([views[t] for t in tickers]) - RISK_FREE
    if use_views:
        P = np.eye(len(tickers))
        tS = tau * Sigma
        Omega = np.diag(np.diag(P @ tS @ P.T))
        mu = pi + tS @ P.T @ np.linalg.solve(P @ tS @ P.T + Omega, Q - P @ pi)
        Sigma_bl = Sigma + np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(Omega) @ P)
    else:
        mu, Sigma_bl = pi, (1 + tau) * Sigma
    w = _max_sharpe(mu, Sigma_bl, max_weight)
    return pd.DataFrame({"weight": w, "w_mkt": w_mkt, "prior_pi": pi, "view_Q": Q, "posterior_mu": mu,
                         "volatility": np.sqrt(np.diag(Sigma_bl))}, index=tickers)


def simulate(weight_fn, dates, end, cost_bps=COST_BPS, band=0.0, adjust=1.0):
    ends = dates[1:] + [prices.loc[:end].index[-1]]
    level, held, pieces, traded, log = 1.0, None, [], [], {}
    for f, f_next in zip(dates, ends):
        target = weight_fn(f)
        drifted = held / held.sum() if held is not None else pd.Series(dtype=float)
        if held is not None and (band > 0 or adjust < 1):
            names = target.index.union(drifted.index)
            t, d = target.reindex(names, fill_value=0), drifted.reindex(names, fill_value=0)
            new = d + adjust * (t - d)
            if band > 0:
                new = new.where((t - d).abs() > band, d)
            new = new.where(prices.loc[f, names].notna(), 0).clip(lower=0)
            target = new[new > 1e-9] / new[new > 1e-9].sum()
        names = target.index.union(drifted.index)
        turnover = (target.reindex(names, fill_value=0) - drifted.reindex(names, fill_value=0)).abs().sum()
        level *= 1 - cost_bps / 1e4 * turnover
        if held is not None:
            traded.append(turnover)
        p = prices.loc[f:f_next, target.index].ffill()
        values = (p / p.iloc[0]).mul(target * level, axis=1)
        curve = values.sum(axis=1)
        pieces.append(curve if not pieces else curve.iloc[1:])
        level, held = curve.iloc[-1], values.iloc[-1]
        log[f] = target
    curve = pd.concat(pieces)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    curve.attrs["turnover"] = sum(traded) / years
    return curve, pd.DataFrame(log).fillna(0)


def performance(curves, benchmark="Cap-weight S&P 500"):
    bench = curves[benchmark].pct_change().dropna()
    out = {}
    for k, v in curves.items():
        r = v.pct_change().dropna()
        years = (v.index[-1] - v.index[0]).days / 365.25
        vol = r.std() * np.sqrt(252)
        active = r - bench.reindex(r.index)
        te = active.std() * np.sqrt(252)
        ir = active.mean() * 252 / te if te > 0 else np.nan
        out[k] = {"CAGR": (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1, "Volatility": vol,
                  "Sharpe": (r.mean() * 252 - RISK_FREE) / vol, "Max drawdown": (v / v.cummax() - 1).min(),
                  "Turnover / yr": v.attrs.get("turnover", np.nan), "Excess vs S&P": active.mean() * 252,
                  "Tracking error": te, "IR": ir, "t-stat": ir * np.sqrt(years) if te > 0 else np.nan}
    return pd.DataFrame(out).T


def arima_cache():
    """ARIMA forecasts computed by main.ipynb: {(date, ticker): {view_annual, fc_quarter, ci_low, ci_high, order}}."""
    return pickle.load(open(os.path.join(CACHE_DIR, "arima_views.pkl"), "rb"))


# ============================================================================
# Phase 5: signals, forward returns, IC, quintiles, Fama-MacBeth
# ============================================================================

SIGNALS = {
    "mom_12_1": "12-1 month momentum",
    "mom_6_1": "6-1 month momentum",
    "high_52w": "Price / 52-week high",
    "rev_1m": "1-month return (reversal)",
    "vol_1y": "1-year volatility",
    "size": "Log market cap",
    "book_to_price": "Book-to-price (book lagged 3 months)",
    "hist_mean": "Historical mean monthly log return",
}


def build_panel(dates, universe_size=10_000):
    """Long panel (date, ticker) of signals and forward returns for point-in-time S&P 500 members.

    Signals use only prices up to each date. Forward returns run from the formation date to the
    month-end 1, 3, 6 and 12 months later; a stock that stops trading keeps its last price, as in
    the backtest engine."""
    P = prices
    i_of = {d: i for i, d in enumerate(P.index)}
    daily_ret = P.pct_change(fill_method=None)
    vol = daily_ret.rolling(252, min_periods=200).std() * np.sqrt(252)
    hi = P.rolling(252, min_periods=200).max()
    book_lag = book.shift(BOOK_LAG_DAYS)

    month_ends = pd.Series(P.index, index=P.index).groupby(P.index.to_period("M")).max().values
    Pm = P.loc[month_ends].ffill()                      # month-end prices; delisted stocks keep last price
    log_m = np.log(Pm).diff()
    hist_mean = log_m.expanding(min_periods=36).mean()
    hist_mean = hist_mean.where(log_m.notna().cumsum() >= 36)
    pos_m = {d: i for i, d in enumerate(Pm.index)}

    rows = []
    for d in dates:
        members = universe(d, universe_size).index
        i, im = i_of[d], pos_m[d]
        if i < 252:
            continue
        p0 = P.iloc[i][members]
        f = pd.DataFrame(index=members)
        f["mom_12_1"] = P.iloc[i - MOM_SKIP][members] / P.iloc[i - MOM_LOOKBACK][members] - 1
        f["mom_6_1"] = P.iloc[i - MOM_SKIP][members] / P.iloc[i - 126][members] - 1
        f["high_52w"] = p0 / hi.iloc[i][members]
        f["rev_1m"] = p0 / P.iloc[i - 21][members] - 1
        f["vol_1y"] = vol.iloc[i][members]
        f["size"] = np.log(mcap.loc[d, members])
        f["book_to_price"] = book_lag.iloc[i][members] / p0
        f["hist_mean"] = hist_mean.iloc[im][members]
        f["mcap"] = mcap.loc[d, members]
        f["rank_cap"] = np.arange(1, len(members) + 1)
        for h in (1, 3, 6, 12):
            if im + h < len(Pm):
                f[f"fwd_{h}m"] = Pm.iloc[im + h][members] / Pm.iloc[im][members] - 1
            else:
                f[f"fwd_{h}m"] = np.nan
        f["date"] = d
        rows.append(f)
    panel = pd.concat(rows).rename_axis("ticker").reset_index().set_index(["date", "ticker"])
    return panel


def nw_mean(x: pd.Series, lags: int):
    """Mean of a time series with a Newey-West t-statistic."""
    x = x.dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.tvalues[0], len(x)


def rank_ic(panel, signal, horizon="fwd_1m", min_n=20):
    """Spearman rank correlation of the signal with forward returns, one value per date."""
    def ic(g):
        g = g[[signal, horizon]].dropna()
        return g[signal].rank().corr(g[horizon].rank()) if len(g) >= min_n else np.nan
    return panel.groupby(level="date").apply(ic)


def winsor_z(s, p=0.01):
    s = s.clip(s.quantile(p), s.quantile(1 - p))
    return (s - s.mean()) / s.std()


def quintile_returns(panel, signal, horizon="fwd_1m", n=5):
    """Equal-weighted forward return of each signal quintile (1 = lowest), one row per date."""
    def q(g):
        g = g[[signal, horizon]].dropna()
        if len(g) < 5 * n:
            return pd.Series(np.nan, index=range(1, n + 1))
        b = pd.qcut(g[signal].rank(method="first"), n, labels=range(1, n + 1))
        return g.groupby(b, observed=True)[horizon].mean()
    return panel.groupby(level="date").apply(q).unstack() if False else \
        pd.DataFrame({d: q(g) for d, g in panel.groupby(level="date")}).T


def fama_macbeth(panel, signals, horizon="fwd_1m", min_n=30):
    """Cross-sectional OLS each date of the (winsorised) forward return on winsorised, standardised
    signals. Coefficients are the return per one standard deviation of the signal."""
    coefs = {}
    for d, g in panel.groupby(level="date"):
        g = g[signals + [horizon]].dropna()
        if len(g) < min_n:
            continue
        X = sm.add_constant(g[signals].apply(winsor_z))
        y = g[horizon].clip(g[horizon].quantile(0.01), g[horizon].quantile(0.99))
        coefs[d] = sm.OLS(y.values, X.values).fit().params
    return pd.DataFrame(coefs, index=["const"] + signals).T


# ============================================================================
# Phase 5: Grinold alphas
# ============================================================================

def grinold_alpha(scores: pd.Series, vol: pd.Series, ic: float, horizon_q: float = 1.0):
    """alpha = IC x volatility x z-score (Grinold & Kahn), annualised.

    The IC is for quarterly returns, so with quarterly volatility vol/2 the quarterly alpha is
    IC * (vol/2) * z, and the annual alpha is 4x that = 2 * IC * vol * z."""
    z = winsor_z(scores.dropna())
    return (2 * ic * vol.reindex(z.index) * z).dropna()


def mean_variance_bl(date, tickers, caps, alpha=None, max_weight=0.20, cov="ledoit_wolf", tau=0.05):
    """Equilibrium prior over the whole universe, plus Grinold alpha as the expected excess return
    (mu = pi + alpha); same max-Sharpe optimiser and caps as the baseline."""
    rets = prices.loc[:date, tickers].iloc[-COV_LOOKBACK:].pct_change(fill_method=None).iloc[1:]
    tickers = list(rets.columns[rets.notna().mean() > 0.8])
    Sigma = covariance(rets[tickers], cov)
    w_mkt = (caps[tickers] / caps[tickers].sum()).values
    delta = MARKET_RISK_PREMIUM / (w_mkt @ Sigma @ w_mkt)
    pi = delta * Sigma @ w_mkt
    a = np.zeros(len(tickers)) if alpha is None else alpha.reindex(tickers).fillna(0).values
    mu = pi + a
    w = _max_sharpe(mu, (1 + tau) * Sigma, max_weight)
    return pd.DataFrame({"weight": w, "w_mkt": w_mkt, "prior_pi": pi, "alpha": a, "mu": mu,
                         "vol": np.sqrt(np.diag(Sigma))}, index=tickers)


def transfer_coefficient(weights, w_bench, alpha, vol):
    """Clarke, de Silva & Thorley (2002): TC = corr(active weight x vol, alpha / vol)."""
    idx = alpha.index
    act = weights.reindex(idx).fillna(0) - w_bench.reindex(idx).fillna(0)
    return np.corrcoef(act * vol.reindex(idx), alpha / vol.reindex(idx))[0, 1]
