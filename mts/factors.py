"""Fama-French factors and the risk-free rate (Phase 8), from Kenneth French's data library.

    python -m mts.factors          # download (or check) the files; writes factors/manifest.json
    from mts import factors
    factors.load()                 # daily Mkt-RF, SMB, HML, RMW, CMA, Mom, RF as decimals

The zip files are not committed (see .gitignore); factors/manifest.json records each file's source
URL, size, SHA-256, download time and the CRSP vintage stated in the file, so the exact version used
can be checked or fetched again.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

from . import config as C

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "five_factors": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
    "five_factors_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",   # monthly RF: finer precision
}
FACTOR_DIR = C.ROOT / "factors"
MANIFEST = FACTOR_DIR / "manifest.json"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vintage(text: str) -> str:
    first = text.splitlines()[0]
    return first.split("using the ")[-1].split(" CRSP")[0] if "CRSP" in first else ""


def download(force: bool = False) -> dict:
    """Download any missing file (or all, with force) and rewrite the manifest."""
    FACTOR_DIR.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for key, name in FILES.items():
        path = FACTOR_DIR / name
        if force or not path.exists():
            req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                path.write_bytes(r.read())
            manifest[key] = {"downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with zipfile.ZipFile(path) as z:
            inner = z.namelist()[0]
            text = z.read(inner).decode("latin-1")
        manifest[key] = {**manifest.get(key, {}), "url": BASE_URL + name, "file": name, "csv_in_zip": inner,
                         "size_bytes": path.stat().st_size, "sha256": _sha256(path), "crsp_vintage": _vintage(text)}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _parse(path) -> pd.DataFrame:
    """The daily table inside a Ken French zip: header row starts with ',', ends at the first blank line."""
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    fmt = "%Y%m" if len(str(df.index[0]).strip()) == 6 else "%Y%m%d"
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format=fmt)
    df.columns = [c.strip() for c in df.columns]
    return df / 100.0                                  # the files are in percent


def daily_rf(index: pd.DatetimeIndex, monthly_rf: pd.Series) -> pd.Series:
    """Spread each month's T-bill return evenly over that month's trading days in `index`
    (compounding to the monthly rate)."""
    month = index.to_period("M")
    days = pd.Series(1, index=index).groupby(month).transform("size")
    m = monthly_rf.copy()
    m.index = m.index.to_period("M")
    rate = m.reindex(month).values
    return pd.Series((1 + rate) ** (1 / days.values) - 1, index=index)


def tbill_annual(date) -> float:
    """The annualised 1-month T-bill rate known on `date`: the return of the bill bought at the end of
    `date`'s month and held through the next month, compounded to a year. Used as the risk-free
    rate in Black-Litterman views, so it must not use returns after `date`."""
    monthly = _parse(FACTOR_DIR / FILES["five_factors_monthly"])["RF"]
    monthly.index = monthly.index.to_period("M")
    nxt = pd.Timestamp(date).to_period("M") + 1
    return float((1 + monthly.loc[nxt]) ** 12 - 1)


def load() -> pd.DataFrame:
    """Daily factor returns and the risk-free rate, as decimals: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF.

    RF is the monthly 1-month T-bill return spread over each month's trading days: the daily file
    rounds it to 0.01% a day, too coarse when the rate is a few thousandths of a percent a day."""
    if not all((FACTOR_DIR / n).exists() for n in FILES.values()):
        download()
    ff5 = _parse(FACTOR_DIR / FILES["five_factors"])
    mom = _parse(FACTOR_DIR / FILES["momentum"])
    mom.columns = ["Mom"]
    out = ff5.join(mom, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
    out["RF"] = daily_rf(out.index, _parse(FACTOR_DIR / FILES["five_factors_monthly"])["RF"])
    return out


if __name__ == "__main__":
    print(json.dumps(download(), indent=2))
