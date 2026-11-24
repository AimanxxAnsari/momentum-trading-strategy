"""Signal research (Phase 5): the monthly signal panel, information coefficients, quintile portfolios,
Fama-MacBeth regressions, and turning scores into Grinold alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from . import data

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


def build_panel(dates, universe_size: int = 10_000) -> pd.DataFrame:
    """Long panel (date, ticker) of signals and forward returns for point-in-time index members.

    Signals use only prices up to each date. Forward returns run from the formation date to the
    month-end 1, 3, 6 and 12 months later; a stock that stops trading keeps its last price, as in
    the backtest engine."""
    P, M = data.prices(), data.mcap()
    i_of = {d: i for i, d in enumerate(P.index)}
    vol = P.pct_change(fill_method=None).rolling(252, min_periods=200).std() * np.sqrt(C.TRADING_DAYS)
    hi = P.rolling(252, min_periods=200).max()
    book_lag = data.book().shift(C.BOOK_LAG_DAYS)

    month_ends = pd.Series(P.index, index=P.index).groupby(P.index.to_period("M")).max().values
    Pm = P.loc[month_ends].ffill()
    log_m = np.log(Pm).diff()
    hist_mean = log_m.expanding(min_periods=36).mean().where(log_m.notna().cumsum() >= 36)
    pos_m = {d: i for i, d in enumerate(Pm.index)}

    rows = []
    for d in dates:
        members = data.universe(d, universe_size).index
        i, im = i_of[d], pos_m[d]
        if i < 252:
            continue
        p0 = P.iloc[i][members]
        f = pd.DataFrame(index=members)
        f["mom_12_1"] = P.iloc[i - C.MOM_SKIP][members] / P.iloc[i - C.MOM_LOOKBACK][members] - 1
        f["mom_6_1"] = P.iloc[i - C.MOM_SKIP][members] / P.iloc[i - 126][members] - 1
        f["high_52w"] = p0 / hi.iloc[i][members]
        f["rev_1m"] = p0 / P.iloc[i - 21][members] - 1
        f["vol_1y"] = vol.iloc[i][members]
        f["size"] = np.log(M.loc[d, members])
        f["book_to_price"] = book_lag.iloc[i][members] / p0
        f["hist_mean"] = hist_mean.iloc[im][members]
        f["mcap"] = M.loc[d, members]
        f["rank_cap"] = np.arange(1, len(members) + 1)
        for h in (1, 3, 6, 12):
            f[f"fwd_{h}m"] = Pm.iloc[im + h][members] / Pm.iloc[im][members] - 1 if im + h < len(Pm) else np.nan
        f["date"] = d
        rows.append(f)
    return pd.concat(rows).rename_axis("ticker").reset_index().set_index(["date", "ticker"])


def load_panel(residual_momentum: bool = False) -> pd.DataFrame:
    """The monthly panel for 2003-2023, built once and cached in cache/. With residual_momentum=True
    it also has `res_mom` (Phase 7), cached separately."""
    path = C.CACHE_DIR / ("panel_monthly_p7.pkl" if residual_momentum else "panel_monthly.pkl")
    if path.exists():
        return pd.read_pickle(path)
    if residual_momentum:
        panel = add_residual_momentum(load_panel())
    else:
        panel = build_panel(data.period_ends("2003-01-01", "2023-12-31", "M"))
    panel.to_pickle(path)
    return panel


# -- Statistics -------------------------------------------------------------------

def nw_mean(x: pd.Series, lags: int):
    """Mean of a time series with a Newey-West t-statistic: (mean, t, n)."""
    x = x.dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.tvalues[0], len(x)


def rank_ic(panel: pd.DataFrame, signal: str, horizon: str = "fwd_1m", min_n: int = 20) -> pd.Series:
    """Spearman rank correlation of the signal with forward returns, one value per date."""
    def ic(g):
        g = g[[signal, horizon]].dropna()
        return g[signal].rank().corr(g[horizon].rank()) if len(g) >= min_n else np.nan
    return panel.groupby(level="date").apply(ic)


def winsor_z(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Clip at the p and 1-p quantiles, then standardise to mean 0, standard deviation 1."""
    s = s.clip(s.quantile(p), s.quantile(1 - p))
    return (s - s.mean()) / s.std()


def quintile_returns(panel: pd.DataFrame, signal: str, horizon: str = "fwd_1m", n: int = 5) -> pd.DataFrame:
    """Equal-weighted forward return of each signal quantile (1 = lowest), one row per date."""
    def q(g):
        g = g[[signal, horizon]].dropna()
        if len(g) < 5 * n:
            return pd.Series(np.nan, index=range(1, n + 1))
        b = pd.qcut(g[signal].rank(method="first"), n, labels=range(1, n + 1))
        return g.groupby(b, observed=True)[horizon].mean()
    return pd.DataFrame({d: q(g) for d, g in panel.groupby(level="date")}).T


def fama_macbeth(panel: pd.DataFrame, signals: list, horizon: str = "fwd_1m", min_n: int = 30) -> pd.DataFrame:
    """Cross-sectional OLS each date of the (winsorised) forward return on winsorised, standardised
    signals. Coefficients are the return per one standard deviation of each signal."""
    coefs = {}
    for d, g in panel.groupby(level="date"):
        g = g[signals + [horizon]].dropna()
        if len(g) < min_n:
            continue
        X = sm.add_constant(g[signals].apply(winsor_z))
        y = g[horizon].clip(g[horizon].quantile(0.01), g[horizon].quantile(0.99))
        coefs[d] = sm.OLS(y.values, X.values).fit().params
    return pd.DataFrame(coefs, index=["const"] + signals).T


# -- Alphas -----------------------------------------------------------------------

def grinold_alpha(scores: pd.Series, vol: pd.Series, ic: float) -> pd.Series:
    """alpha = IC x volatility x z-score (Grinold & Kahn), annualised.

    The IC is for quarterly returns, so with quarterly volatility vol/2 the quarterly alpha is
    IC * (vol/2) * z, and the annual alpha is 4x that = 2 * IC * vol * z."""
    z = winsor_z(scores.dropna())
    return (2 * ic * vol.reindex(z.index) * z).dropna()


class PointInTimeIC:
    """Expanding mean of realised 3-month ICs, using only formations whose returns are known.

    On date f, a 3-month IC formed at month s is known once s + 3 months <= f."""

    def __init__(self, ic_series: pd.Series, horizon_months: int = 3, min_obs: int = 24):
        self.ic = ic_series.sort_index()
        self.months = list(self.ic.index)
        self.h, self.min_obs = horizon_months, min_obs

    def __call__(self, f) -> float:
        k = self.months.index(f)
        past = self.ic.iloc[:max(k - self.h + 1, 0)].dropna()
        return past.mean() if len(past) >= self.min_obs else np.nan


def transfer_coefficient(weights: pd.Series, w_bench: pd.Series, alpha: pd.Series, vol: pd.Series) -> float:
    """Clarke, de Silva & Thorley (2002): TC = corr(active weight x vol, alpha / vol)."""
    idx = alpha.index
    act = weights.reindex(idx).fillna(0) - w_bench.reindex(idx).fillna(0)
    return np.corrcoef(act * vol.reindex(idx), alpha / vol.reindex(idx))[0, 1]


# -- Phase 7: residual momentum and volatility-managed momentum ---------------------

def monthly_prices() -> pd.DataFrame:
    """Month-end prices; a stock that stops trading keeps its last price."""
    P = data.prices()
    month_ends = pd.Series(P.index, index=P.index).groupby(P.index.to_period("M")).max().values
    return P.loc[month_ends].ffill()


def market_monthly() -> pd.Series:
    """Cap-weighted monthly return of the point-in-time S&P 500, indexed by the month-end it ends on.
    Weights are market caps at the start of the month; members are those in the latest snapshot."""
    Pm, M = monthly_prices(), data.mcap()
    R = Pm.pct_change(fill_method=None)
    out = {}
    for prev, cur in zip(Pm.index[:-1], Pm.index[1:]):
        caps = data.universe(prev, 10_000)
        r = R.loc[cur, caps.index]
        ok = r.notna()
        out[cur] = (caps[ok] * r[ok]).sum() / caps[ok].sum()
    return pd.Series(out)


def residual_momentum(R: pd.DataFrame, m: pd.Series, skip: int = 1, formation: int = 11,
                      min_obs: int = 24) -> pd.Series:
    """Blitz, Huij & Martens (2011), with the market as the only factor.

    R: monthly stock returns over the estimation window (e.g. 36 months, oldest first); m: market
    returns on the same months. Each stock's returns are regressed on the market; residual momentum is
    the sum of its residuals over the `formation` months before the last `skip` months, divided by
    their standard deviation (so it measures how consistently the stock beat its beta, not by how much)."""
    m = m.reindex(R.index)
    mask = R.notna().values & m.notna().values[:, None]     # months with both a stock and a market return
    r = np.where(mask, R.values, 0.0)
    mv = np.nan_to_num(m.values)[:, None]
    n = mask.sum(axis=0)
    mm = (mv * mask).sum(axis=0) / np.maximum(n, 1)
    rm = r.sum(axis=0) / np.maximum(n, 1)
    dm, dr = (mv - mm) * mask, (r - rm) * mask
    beta = (dm * dr).sum(axis=0) / np.where((dm ** 2).sum(axis=0) > 0, (dm ** 2).sum(axis=0), np.nan)
    alpha = rm - beta * mm
    resid = np.where(mask, r - alpha - beta * mv, np.nan)
    window = resid[-(formation + skip):len(resid) - skip]
    s = np.nanstd(window, axis=0, ddof=1)
    score = np.nansum(window, axis=0) / np.where(s > 0, s, np.nan)
    score[(n < min_obs) | (np.isfinite(window).sum(axis=0) < formation - 2)] = np.nan
    return pd.Series(score, index=R.columns)


def add_residual_momentum(panel: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Adds `res_mom` (residual 12-1 momentum, 36-month market-model regression) to the panel."""
    Pm = monthly_prices()
    R = Pm.pct_change(fill_method=None)
    mkt = market_monthly()
    pos = {d: i for i, d in enumerate(Pm.index)}
    parts = []
    for d, g in panel.groupby(level="date"):
        i = pos[d]
        tick = g.index.get_level_values("ticker")
        Rw = R.iloc[max(i - window + 1, 0):i + 1][tick]
        s = residual_momentum(Rw, mkt.reindex(Rw.index))
        parts.append(pd.Series(s.values, index=g.index))
    out = panel.copy()
    out["res_mom"] = pd.concat(parts)
    return out


def long_short_daily(panel: pd.DataFrame, signal: str, n: int = 5) -> pd.Series:
    """Daily return of the top-minus-bottom quantile portfolio (equal-weighted, formed at each
    month-end in `panel`, held to the next). A stock that stops trading earns 0 from then on."""
    P = data.prices()
    daily = P.pct_change(fill_method=None)
    dates = sorted(panel.index.get_level_values("date").unique())
    ends = dates[1:] + [P.index[-1]]
    pieces = []
    for d, e in zip(dates, ends):
        g = panel.loc[d, signal].dropna()
        if len(g) < 5 * n:
            continue                                  # signal not available yet
        q = pd.qcut(g.rank(method="first"), n, labels=False)
        top, bottom = g.index[q == n - 1], g.index[q == 0]
        r = daily.loc[d:e].iloc[1:]
        pieces.append(r[top].fillna(0).mean(axis=1) - r[bottom].fillna(0).mean(axis=1))
    return pd.concat(pieces)


def vol_scale(daily: pd.Series, at, target: float = 0.12, window: int = 126, cap: float = 2.0) -> float:
    """Barroso & Santa-Clara (2015): exposure = target volatility / realised volatility of the
    strategy over the last `window` days before `at`, capped at `cap`."""
    past = daily.loc[:at].iloc[-window:]
    realised = np.sqrt((past ** 2).mean() * C.TRADING_DAYS)
    return float(min(target / realised, cap)) if realised > 0 else np.nan
