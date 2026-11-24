"""Loading, cleaning and the point-in-time universe (main.ipynb sections 1-3, unchanged logic).

The cleaned tables are loaded once, lazily, and cached in cache/clean_data.pkl:

    from mts import data
    data.prices()            # date x ticker closing prices (split-adjusted, price returns only)
    data.universe(date)      # 50 largest index members on `date`, market caps, largest first
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


def parse_snapshot_date(filename: str) -> pd.Timestamp:
    """'SPX as of Sep 30 20201.xlsx' -> 2020-09-30 (some names carry a trailing copy suffix)."""
    match = re.search(r"as of (\w{3} \d{2} \d{4})", filename)
    if not match:
        raise ValueError(f"Unable to parse date from filename: {filename}")
    return pd.Timestamp(datetime.strptime(match.group(1), "%b %d %Y"))


def load_snapshots(folder=C.SNAPSHOT_DIR) -> dict:
    """{date: {'AAPL UW Equity', ...}} from the quarterly S&P 500 constituent files."""
    import openpyxl
    snapshots = {}
    for path in glob.glob(os.path.join(folder, "*.xlsx")):
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(sheet.iter_rows(values_only=True))
        col = rows[0].index("Ticker")
        snapshots[parse_snapshot_date(os.path.basename(path))] = {r[col] for r in rows[1:] if r[col]}
    return dict(sorted(snapshots.items()))


def _clean() -> dict:
    snapshot_listings = load_snapshots()
    snapshots = {d: {l.split(" ")[0] for l in s} for d, s in snapshot_listings.items()}
    snapshot_dates = np.array(list(snapshots), dtype="datetime64[ns]")

    df = pd.read_csv(C.DATA_PATH, parse_dates=["date"]).dropna(subset=["PX_LAST"])
    df["listing"] = df["ticker"]
    df["ticker"] = df["listing"].str.split(" ").str[0]
    # Keep the listing named in the latest snapshot when a company has two listings on a date
    pos = np.searchsorted(snapshot_dates, df["date"].values, side="right") - 1
    df["snapshot"] = snapshot_dates[np.clip(pos, 0, None)]
    named = pd.DataFrame([(d, l) for d, ls in snapshot_listings.items() for l in ls], columns=["snapshot", "listing"])
    named["in_snapshot"] = True
    df = df.merge(named, on=["snapshot", "listing"], how="left")
    df["in_snapshot"] = df["in_snapshot"].fillna(False).astype(bool)
    df["history"] = df.groupby("listing")["date"].transform("size")
    df = (df.sort_values(["ticker", "date", "in_snapshot", "history"], ascending=[True, True, False, False])
            .drop_duplicates(["ticker", "date"]))
    wide = lambda col: df.pivot(index="date", columns="ticker", values=col)
    return {"prices": wide("PX_LAST"), "mcap": wide("CUR_MKT_CAP"), "book": wide("BOOK_VAL_PER_SH"),
            "volume": wide("PX_VOLUME"), "snapshots": snapshots}


@lru_cache(maxsize=1)
def load() -> dict:
    path = C.CACHE_DIR / "clean_data.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    d = _clean()
    C.CACHE_DIR.mkdir(exist_ok=True)
    pickle.dump(d, open(path, "wb"))
    return d


def prices() -> pd.DataFrame:
    return load()["prices"]


def mcap() -> pd.DataFrame:
    return load()["mcap"]


def book() -> pd.DataFrame:
    return load()["book"]


def snapshots() -> dict:
    return load()["snapshots"]


def constituents(date) -> set:
    """Index members in the most recent snapshot on or before `date`."""
    s = snapshots()
    return s[max(d for d in s if d <= date)]


@lru_cache(maxsize=4096)
def universe(date, size: int = C.UNIVERSE_SIZE) -> pd.Series:
    """Largest index members with a price on `date`: market caps, largest first."""
    P, M = prices(), mcap()
    members = [t for t in constituents(date) if t in P.columns]
    caps = M.loc[:date, members].iloc[-5:].ffill().iloc[-1]
    caps = caps[P.loc[date, members].notna()].dropna()
    return caps.sort_values(ascending=False).head(size)


def period_ends(start, end, freq: str = "Q") -> list:
    """Last trading day of each period (Q or M), from the period before `start`, before the last day."""
    idx = prices().index
    ends = pd.Series(idx, index=idx).groupby(idx.to_period(freq)).max()
    last_day = prices().loc[:end].index[-1]
    first = pd.Timestamp(start).to_period(freq) - 1
    return [d for q, d in ends.items() if q >= first and d < last_day]


def daily_returns(date, tickers, lookback: int = C.COV_LOOKBACK, min_coverage: float = 0.8) -> pd.DataFrame:
    """Daily returns over the `lookback` days to `date`, for tickers with at least `min_coverage` of them."""
    rets = prices().loc[:date, list(tickers)].iloc[-lookback:].pct_change(fill_method=None).iloc[1:]
    return rets[rets.columns[rets.notna().mean() > min_coverage]]
