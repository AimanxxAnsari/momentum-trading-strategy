"""Every strategy variant tested in Phases 1-7, rebuilt from one place (Phase 8).

Phase 8 needs all the variants on the same footing, to correct for how many were tried
(deflated Sharpe ratio, probability of backtest overfitting). Each `Trial` holds the weight
function and the backtest settings it was run with. Everything is over 2010-2023, quarterly
rebalancing, 10 bps costs, the period where every phase overlaps.

    reg = Registry()
    for t in reg.trials():               # Trial(name, phase, weights, sim_kwargs)
        curve, log = bt.simulate(t.weights, reg.dates, reg.end, **t.sim_kwargs)

Cost-only variants (the same portfolio charged a different cost) and benchmarks are not trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as bt
from . import baseline as BL
from . import config as C
from . import data
from . import optimize as opt
from . import risk
from . import signals as S


@dataclass
class Trial:
    name: str
    phase: str
    weights: callable
    sim_kwargs: dict = field(default_factory=dict)


class Registry:
    def __init__(self, start: str = "2010-01-01", end: str = "2023-12-31"):
        self.dates = data.period_ends(start, end)
        self.end = end
        self._ctx = self._ctx_full = None
        self._panel = None
        self._models = {}

    # -- shared inputs, built on first use ------------------------------------------------
    @property
    def ctx(self) -> dict:
        """Top-50 universe, S&P 500 caps, momentum signals and ARIMA views on each date (main.ipynb)."""
        if self._ctx is None:
            cache = BL.arima_cache()
            self._ctx = {}
            for f in self.dates:
                u = data.universe(f)
                sig = BL.momentum_signals(f, list(u.index))
                views = {t: cache[(f, t)] for t in sig.index}
                self._ctx[f] = {"universe": u, "sp500": data.universe(f, 10_000), "signals": sig,
                                "views": pd.Series({t: v["view_annual"] for t, v in views.items()}),
                                "views_mean": pd.Series({t: BL.arima_view_mean(v["fc_quarter"], v["ci_low"], v["ci_high"])
                                                         for t, v in views.items()})}
        return self._ctx

    @property
    def ctx_full(self) -> dict:
        """Momentum signals for every S&P 500 member and ARIMA views for the picks (main.ipynb section 12)."""
        if self._ctx_full is None:
            cache = BL.arima_cache()
            self._ctx_full = {}
            for f in self.dates:
                sp = self.ctx[f]["sp500"]
                sig = BL.momentum_signals(f, list(sp.index))
                picks = BL.select_stocks(sig, C.BASE["n_select"], C.BASE["rsi_max"]).index
                self._ctx_full[f] = {"universe": sp, "signals": sig,
                                     "views": pd.Series({t: cache[(f, t)]["view_annual"] for t in picks})}
        return self._ctx_full

    @property
    def panel(self) -> pd.DataFrame:
        """Top-50 monthly panel with the composite signal, point-in-time IC estimates, momentum factor."""
        if self._panel is None:
            panel = S.load_panel(residual_momentum=True)
            top = panel[panel.rank_cap <= C.UNIVERSE_SIZE].copy()
            z = lambda s: top.groupby(level="date")[s].transform(S.winsor_z)
            top["composite"] = (z("mom_12_1") + z("hist_mean")) / 2
            self._panel = top
            self.ic = {s: S.PointInTimeIC(S.rank_ic(top, s, "fwd_3m"), 3, 24)
                       for s in ["mom_12_1", "hist_mean", "composite", "res_mom"]}
            self.mom_factor = S.long_short_daily(panel, "mom_12_1")
        return self._panel

    def model(self, f, half_life=None):
        key = (f, half_life)
        if key not in self._models:
            u = data.universe(f)
            rets = data.daily_returns(f, u.index)
            self._models[key] = (u, rets, risk.fit_pca(rets, C.N_FACTORS, half_life=half_life))
        return self._models[key]

    # -- strategy constructors -----------------------------------------------------------------
    def baseline(self, n_select=25, rsi_max=70, max_weight=0.20, tau=0.05, cov="ledoit_wolf", use_views=True,
                 weighting="black_litterman", full=False, view_key="views"):
        def weights(f):
            c = (self.ctx_full if full else self.ctx)[f]
            sel = BL.select_stocks(c["signals"], n_select, rsi_max)
            if weighting == "equal":
                return pd.Series(1 / len(sel), index=sel.index)
            return BL.black_litterman(f, list(sel.index), c["universe"], c[view_key], tau=tau,
                                      max_weight=max_weight, cov=cov, use_views=use_views)["weight"]
        return weights

    def grinold_max_sharpe(self, sig):
        def weights(f):
            top = self.panel
            u = data.universe(f)
            ic = self.ic[sig](f)
            a = None
            if np.isfinite(ic):
                a = S.grinold_alpha(top.loc[f, sig].reindex(u.index), top.loc[f, "vol_1y"].reindex(u.index), ic)
            return opt.max_sharpe_with_alpha(f, list(u.index), u, a)["weight"]
        return weights

    def active(self, sig, lam=10, kappa=1, half_life=None, aligned=False, scale=False):
        def weights(f):
            top = self.panel
            u, rets, rm = self.model(f, half_life)
            t = rm.tickers
            b = u[t] / u[t].sum()
            ic = self.ic[sig](f)
            z = S.winsor_z(top.loc[f, sig].reindex(t).dropna())
            a = (2 * ic * rm.specific_vol * z).reindex(t).fillna(0) if np.isfinite(ic) else pd.Series(0.0, index=t)
            if scale:
                a = a * S.vol_scale(self.mom_factor, f)
            m = risk.add_alpha_factor(rm, rets, z, half_life) if aligned else rm
            cur = bt.simulate.current
            return opt.optimize_active(a, m.cov, b, cur if len(cur) else None, risk_aversion=lam, te_max=0.04,
                                       cost=kappa * C.COST_BPS / 1e4)
        return weights

    # -- the list of trials ------------------------------------------------------------------------
    def trials(self) -> list:
        T = []
        add = lambda name, phase, fn, **kw: T.append(Trial(name, phase, fn, kw))
        # Phases 1-4 (main.ipynb): baseline, sensitivity, layers, turnover controls, full universe
        add("Baseline", "1-4", self.baseline())
        for v in (0.10, 0.30, 1.00):
            add(f"Baseline, max weight {v:.0%}", "1-4", self.baseline(max_weight=v))
        for n in (15, 35):
            add(f"Baseline, {n} stocks", "1-4", self.baseline(n_select=n))
        add("Baseline, no RSI filter", "1-4", self.baseline(rsi_max=101))
        for tau in (0.025, 0.10):
            add(f"Baseline, tau {tau}", "1-4", self.baseline(tau=tau))
        add("Baseline, sample covariance", "1-4", self.baseline(cov="sample"))
        add("Momentum top 25, equal weight", "1-4", self.baseline(weighting="equal"))
        add("Momentum top 25, Black-Litterman without views", "1-4", self.baseline(use_views=False))
        add("Baseline, no-trade band 2%", "1-4", self.baseline(), band=0.02)
        add("Baseline, no-trade band 5%", "1-4", self.baseline(), band=0.05)
        add("Baseline, partial rebalance 50%", "1-4", self.baseline(), adjust=0.5)
        add("Baseline, band 2% + partial 50%", "1-4", self.baseline(), band=0.02, adjust=0.5)
        add("Strategy on full S&P 500", "1-4", self.baseline(full=True))
        add("Momentum top 50 of S&P 500, equal weight", "1-4",
            self.baseline(n_select=50, rsi_max=101, weighting="equal", full=True))
        # Phase 5
        add("Baseline, ARIMA views corrected to mean", "5", self.baseline(view_key="views_mean"))
        for s, label in [("mom_12_1", "momentum"), ("hist_mean", "historical mean"), ("composite", "composite")]:
            add(f"Max-Sharpe Grinold alpha: {label}", "5", self.grinold_max_sharpe(s))
        # Phase 6
        for s, label in [("hist_mean", "historical mean"), ("mom_12_1", "momentum"), ("composite", "composite")]:
            add(f"Active: {label}", "6", self.active(s))
        for s, label in [("hist_mean", "historical mean"), ("mom_12_1", "momentum")]:
            for lam in (2.5, 5, 20, 40):
                add(f"Active: {label}, lambda {lam:g}", "6", self.active(s, lam=lam))
            for kappa in (0, 3):
                add(f"Active: {label}, kappa {kappa}", "6", self.active(s, kappa=kappa))
        # Phase 7
        add("Active: residual momentum", "7", self.active("res_mom"))
        add("Active: momentum, volatility-managed", "7", self.active("mom_12_1", scale=True))
        for s, label in [("hist_mean", "historical mean"), ("mom_12_1", "momentum")]:
            add(f"Active: {label}, alpha-aligned", "7", self.active(s, aligned=True))
            for hl in (126, 63, 42):
                add(f"Active: {label}, half-life {hl}", "7", self.active(s, half_life=hl))
        return T

    def benchmarks(self) -> dict:
        return {"Cap-weight S&P 500": lambda f: self.ctx[f]["sp500"] / self.ctx[f]["sp500"].sum(),
                "Cap-weight top 50": lambda f: self.ctx[f]["universe"] / self.ctx[f]["universe"].sum(),
                "Equal-weight top 50": lambda f: pd.Series(1 / len(self.ctx[f]["universe"]), index=self.ctx[f]["universe"].index)}

    def run_all(self, cache_path=None) -> tuple[dict, pd.DataFrame]:
        """Value curves of every trial and benchmark (cached), and a table of the trials."""
        import pickle
        if cache_path is not None and cache_path.exists():
            return pickle.load(open(cache_path, "rb"))
        curves, rows = {}, []
        for t in self.trials():
            curves[t.name] = bt.simulate(t.weights, self.dates, self.end, **t.sim_kwargs)[0]
            rows.append({"trial": t.name, "phase": t.phase, **{k: v for k, v in t.sim_kwargs.items()}})
        for name, fn in self.benchmarks().items():
            curves[name] = bt.simulate(fn, self.dates, self.end)[0]
        out = (curves, pd.DataFrame(rows).set_index("trial"))
        if cache_path is not None:
            pickle.dump(out, open(cache_path, "wb"))
        return out
