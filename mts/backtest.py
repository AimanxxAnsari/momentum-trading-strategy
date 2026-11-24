"""Walk-forward backtest engine and performance statistics (main.ipynb section 7, unchanged logic).

`simulate` takes a function `formation date -> target weights` and returns the value curve and the
weights log. Holdings drift with prices between rebalances; every trade pays `cost_bps` per unit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data


def simulate(weight_fn, dates, end, cost_bps: float = C.COST_BPS, band: float = 0.0, adjust: float = 1.0):
    """band: skip trades smaller than this weight change. adjust: move only this fraction toward the target.
    The weight function may read `simulate.current` (the drifted weights before trading)."""
    P = data.prices()
    ends = dates[1:] + [P.loc[:end].index[-1]]
    level, held, pieces, traded, log = 1.0, None, [], [], {}
    for f, f_next in zip(dates, ends):
        drifted = held / held.sum() if held is not None else pd.Series(dtype=float)
        simulate.current = drifted
        target = weight_fn(f)
        if held is not None and (band > 0 or adjust < 1):
            names = target.index.union(drifted.index)
            t, d = target.reindex(names, fill_value=0), drifted.reindex(names, fill_value=0)
            new = d + adjust * (t - d)
            if band > 0:
                new = new.where((t - d).abs() > band, d)
            new = new.where(P.loc[f, names].notna(), 0).clip(lower=0)
            target = new[new > 1e-9] / new[new > 1e-9].sum()
        names = target.index.union(drifted.index)
        turnover = (target.reindex(names, fill_value=0) - drifted.reindex(names, fill_value=0)).abs().sum()
        level *= 1 - cost_bps / 1e4 * turnover
        if held is not None:
            traded.append(turnover)
        p = P.loc[f:f_next, target.index].ffill()
        values = (p / p.iloc[0]).mul(target * level, axis=1)
        curve = values.sum(axis=1)
        pieces.append(curve if not pieces else curve.iloc[1:])
        level, held = curve.iloc[-1], values.iloc[-1]
        log[f] = target
    curve = pd.concat(pieces)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    curve.attrs["turnover"] = sum(traded) / years
    return curve, pd.DataFrame(log).fillna(0)


simulate.current = pd.Series(dtype=float)


def performance(curves: dict, benchmark: str = "Cap-weight S&P 500") -> pd.DataFrame:
    """CAGR, volatility, Sharpe, drawdown, turnover, and active statistics against `benchmark`:
    excess return, tracking error, information ratio and its t-statistic (IR x sqrt(years))."""
    bench = curves[benchmark].pct_change().dropna()
    out = {}
    for k, v in curves.items():
        r = v.pct_change().dropna()
        years = (v.index[-1] - v.index[0]).days / 365.25
        vol = r.std() * np.sqrt(C.TRADING_DAYS)
        active = r - bench.reindex(r.index)
        te = active.std() * np.sqrt(C.TRADING_DAYS)
        ir = active.mean() * C.TRADING_DAYS / te if te > 0 else np.nan
        out[k] = {"CAGR": (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1, "Volatility": vol,
                  "Sharpe": (r.mean() * C.TRADING_DAYS - C.RISK_FREE) / vol,
                  "Max drawdown": (v / v.cummax() - 1).min(),
                  "Turnover / yr": v.attrs.get("turnover", np.nan), "Excess vs S&P": active.mean() * C.TRADING_DAYS,
                  "Tracking error": te, "IR": ir, "t-stat": ir * np.sqrt(years) if te > 0 else np.nan}
    return pd.DataFrame(out).T


def active_stats(curves: dict, benchmark: str) -> pd.DataFrame:
    """Active return, tracking error, IR and t-statistic of every curve against one benchmark curve."""
    p = performance(curves, benchmark)
    return p[["Excess vs S&P", "Tracking error", "IR", "t-stat"]].rename(columns={"Excess vs S&P": "Active return"})
