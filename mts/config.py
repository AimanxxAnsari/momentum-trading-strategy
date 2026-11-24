"""Settings shared by every notebook. Values are those fixed in main.ipynb unless marked otherwise."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "tickers" / "spx_companies_historical_data.csv"
SNAPSHOT_DIR = ROOT / "tickers"
CACHE_DIR = ROOT / "cache"
IMAGE_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"

# Universe and momentum signal (main.ipynb)
UNIVERSE_SIZE = 50            # largest S&P 500 members by market cap on each rebalance date
MOM_LOOKBACK = 252            # 12-month momentum window (trading days)
MOM_SKIP = 21                 # skip the most recent month

# Market assumptions, annual (main.ipynb)
RISK_FREE = 0.03
MARKET_RISK_PREMIUM = 0.05
COV_LOOKBACK = 756            # 3 years of daily returns
TRADING_DAYS = 252

# Baseline strategy (main.ipynb)
BASE = dict(n_select=25, rsi_max=70, max_weight=0.20, tau=0.05, cov="ledoit_wolf", use_views=True)
COST_BPS = 10                 # cost per unit traded, charged to every portfolio

# Phase 5
BOOK_LAG_DAYS = 63            # book value is stamped at fiscal period end, before publication

# Phase 6
N_FACTORS = 5                 # statistical risk model: principal components of daily returns
