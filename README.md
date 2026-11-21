# Momentum Trading Strategy

A research project that selects S&P 500 stocks and builds a portfolio from them. It uses time-series forecasting (ARIMA), technical momentum indicators and Black-Litterman portfolio optimisation, with daily data from 2000 to 2023.

> **Status:** Phase 1 (the baseline) is uploaded. Phase 2 (correctness fixes) is in progress. The Phase 1 results contain known bugs, listed below, and should not be read as real performance.

---

## How the pipeline works

All the main work is in [`main.ipynb`](main.ipynb). It runs in five stages:

```
Raw S&P 500 data ─▶ 1. Clean ─▶ 2. Split ─▶ 3. ARIMA screen ─▶ 4. Momentum indicators ─▶ 5. Black-Litterman + quarterly rebalancing
```

1. **Data cleaning.** Removes exchange suffixes from tickers (`AAPL UW Equity` → `AAPL`), so one company's listings merge into a single series, then removes duplicate dates for each ticker.
2. **Train/test split.** Training data is 2000–2022 (about 5.66M rows). Test data is 2023 (about 263k rows).
3. **ARIMA screen.** Takes the 50 largest stocks by market cap and resamples their prices to quarterly. For each one, `auto_arima` picks a (seasonal) ARIMA model, which is fitted and used to forecast 4 quarters ahead. Simple and exponential moving averages are plotted alongside. Stocks are ranked by how closely the forecast % change matched the actual 2023 % change, and the 25 closest are kept.
4. **Momentum indicators.** Computes RSI (14-day), MACD (12/26/9) and ROC (12-day) for the 25 selected stocks.
5. **Portfolio allocation.**
   - **Black-Litterman:** blends market-cap-implied returns with investor views, then finds the weights that maximise the Sharpe ratio (SLSQP, long-only, fully invested).
   - **Quarterly rebalancing:** re-runs the optimisation each quarter of 2023, starting from a $1M portfolio.

### Files

| File | Purpose |
|------|---------|
| `main.ipynb` | The full pipeline (stages 1–5) |
| `main-copy-pdf.ipynb` | An earlier version of the pipeline, which also contains the step that turns the Excel snapshots into the CSV |
| `data-cleaning.ipynb` | The cleaning step on its own |
| `draft-assets.ipynb` | Drafts: cross-sectional momentum strategies (v1–v3) and early ARIMA plots |

### Data (not included)

The data is not in this repository: it is too large for GitHub and may not be redistributable. The notebooks expect:

- `tickers/SPX as of <date>.xlsx`: quarterly snapshots of S&P 500 constituents, 2000–2023
- `tickers/spx_companies_historical_data.csv`: daily data for each ticker, with these columns:
  `ticker, date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, DVD_YLD_IND, EPS, BOOK_VAL_PER_SH, CUR_MKT_CAP`

### Requirements

Python 3.9+, `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`, `pmdarima`, `openpyxl`.

---

## Project phases

### Phase 1: Baseline ✅ Complete (this upload)

The first working version of the full pipeline.

- [x] Extracted the S&P 500 constituent snapshots from Excel into a single dataset
- [x] Cleaned the data: merged ticker suffixes and removed duplicate dates
- [x] Split into train (2000–2022) and test (2023)
- [x] Ran ARIMA forecasts with SMA/EMA overlays for the top 50 stocks by market cap
- [x] Computed RSI, MACD and ROC indicators
- [x] Built a Black-Litterman portfolio allocation
- [x] Built quarterly rebalancing over 2023

**Phase 1 output:** 25 selected stocks, allocation weights, and a Sharpe ratio of 12.91. A review found that these numbers come from the bugs below, so they are not valid results.

### Phase 2: Correctness fixes 🚧 In progress

Status key: ⬜ To do · 🔄 In progress · ✅ Done

#### Critical: these change the results

| # | Task | Problem | Status |
|---|------|---------|--------|
| 2.1 | Fix the Black-Litterman views | `P @ tau * cov_matrix @ P.T` runs `P @ tau` first, which raises an error. The code catches it and falls back to the implied returns, so the investor views are never applied. Should be `P @ (tau * cov_matrix) @ P.T`. | ⬜ To do |
| 2.2 | Fix the weight labels | The covariance matrix and returns are in alphabetical order, but the weights are labelled in ARIMA-rank order, so each printed weight belongs to a different stock. | ⬜ To do |
| 2.3 | Fix the return and Sharpe calculations | A 3% annual risk-free rate is added to daily returns, the Sharpe ratio never subtracts the risk-free rate, and the market risk premium is used where the formula needs a risk-aversion coefficient. Together these produce the impossible Sharpe of 12.91. | ⬜ To do |
| 2.4 | Remove look-ahead from stock selection | Stocks are chosen by comparing forecasts with the actual 2023 returns, and the strategy is then tested on 2023. Selection must use only data from before the test period. | ⬜ To do |
| 2.5 | Make the backtest walk forward | Fit weights on data up to quarter *t* and score them on quarter *t+1* using compounded returns, not the mean daily return. Score Q1 as well, and report or plot the portfolio value. | ⬜ To do |
| 2.6 | Add a benchmark | Compare the strategy against an equal-weight portfolio and the S&P 500, reporting return, volatility, Sharpe and maximum drawdown. | ⬜ To do |

#### Method improvements

| # | Task | Problem | Status |
|---|------|---------|--------|
| 2.7 | Use a real momentum signal | The ARIMA-accuracy screen favours flat, defensive stocks: about 20 of the 48 fitted models forecast no change at all. Rank stocks with a momentum signal (for example, 12-month minus 1-month return) and use RSI/MACD in the decision. | ⬜ To do |
| 2.8 | Improve the ARIMA setup | Model log returns instead of raw prices, and drop the seasonal `m=4` term, since quarterly prices have no real seasonality. | ⬜ To do |
| 2.9 | Remove survivorship bias | Build the stock universe for each date from the constituent snapshots, instead of using each stock's latest market cap. Handle delisted tickers (such as UN and CMCSK) properly. | ⬜ To do |

#### Housekeeping

| # | Task | Problem | Status |
|---|------|---------|--------|
| 2.10 | Make the notebook run from a fresh kernel | Restore the data path (`sp500-dataset.csv` → `tickers/spx_companies_historical_data.csv`) and the missing `import re`. | ⬜ To do |
| 2.11 | Speed up the data cleaning | Replace the loops that grow a DataFrame with `pd.concat` with a single `drop_duplicates(['ticker', 'date'])`, and remove helper functions that don't do anything. | ⬜ To do |
| 2.12 | Fix wrong labels | Plots and printouts say "2023–2024" and "Latest price (2024)", but the test data ends on 2023-12-31. | ⬜ To do |
| 2.13 | Consolidate the notebooks | Merge or remove the duplicate code in `main-copy-pdf.ipynb`, `data-cleaning.ipynb` and `draft-assets.ipynb`. | ⬜ To do |
| 2.14 | Pin the dependencies | Add a `requirements.txt`. | ⬜ To do |

**Progress: 0 / 14 done**
