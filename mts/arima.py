"""ARIMA return forecasts used as Black-Litterman views (main.ipynb section 5).

For each stock, `auto_arima` picks an ARIMA order for its quarterly log returns up to the formation
date (no seasonal term); that order is refitted with an intercept forced on, so a (0,0,0) model
forecasts the stock's mean return rather than 0. Forecasts are fitted in parallel and cached in
cache/arima_views.pkl, keyed by (date, ticker).
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data

CACHE_FILE = C.CACHE_DIR / "arima_views.pkl"


def quarterly_log_returns(date, ticker) -> pd.Series:
    q = data.prices().loc[:date, ticker].dropna().resample(pd.offsets.QuarterEnd()).last()
    return np.log(q).diff().dropna()


def fit_arima(q: pd.Series) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    from pmdarima import auto_arima
    from pmdarima.arima import ARIMA
    order = auto_arima(q, seasonal=False, max_p=3, max_q=3, suppress_warnings=True, error_action="ignore").order
    model = ARIMA(order=order, with_intercept=True, suppress_warnings=True).fit(q)
    fc, ci = model.predict(1, return_conf_int=True)
    fc = float(np.asarray(fc)[0])
    return {"view_annual": np.exp(4 * fc) - 1, "fc_quarter": fc,
            "ci_low": float(ci[0][0]), "ci_high": float(ci[0][1]), "order": order}


def arima_views(jobs, n_jobs: int = -1) -> dict:
    """jobs: (date, ticker) pairs. Returns {(date, ticker): forecast}, fitting only what is not cached."""
    from joblib import Parallel, delayed
    C.CACHE_DIR.mkdir(exist_ok=True)
    cache = pickle.load(open(CACHE_FILE, "rb")) if CACHE_FILE.exists() else {}
    todo = [j for j in dict.fromkeys(jobs) if j not in cache]
    if todo:
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs)(delayed(fit_arima)(quarterly_log_returns(d, t)) for d, t in todo)
        cache.update(zip(todo, results))
        pickle.dump(cache, open(CACHE_FILE, "wb"))
        print(f"Fitted {len(todo):,} ARIMA models in {time.time() - t0:.0f}s ({len(cache):,} cached)")
    return {j: cache[j] for j in jobs}
