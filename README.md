# Momentum Trading Strategy

A research project that selects S&P 500 stocks with a momentum signal, forms return views with ARIMA, and allocates the portfolio with Black-Litterman. The strategy is tested in a quarterly walk-forward backtest over 2023 and over 2010–2023.

> **Status:** Phase 1 (baseline) and Phase 2 (correctness fixes) are complete. Over 2010–2023 the strategy returned 12.6% a year against 11.1% for an S&P 500 proxy, with a Sharpe ratio of 0.54 against 0.52. That is no real risk-adjusted edge. See [Results](#results).

---

## How the pipeline works

All the main work is in [`main.ipynb`](main.ipynb). On each rebalance date (the last trading day of each quarter), it runs:

```
Clean data ─▶ Point-in-time universe ─▶ Momentum selection ─▶ ARIMA views ─▶ Black-Litterman ─▶ Hold one quarter
```

1. **Data cleaning.** Removes exchange suffixes from tickers (`AAPL UW Equity` → `AAPL`). Drops blank rows (28% of the raw file) *before* de-duplicating, so a blank row can never replace a real price.
2. **Point-in-time universe.** The 50 largest stocks that were **in the S&P 500 on that date**, taken from the quarterly constituent snapshots. This avoids survivorship bias.
3. **Momentum selection.** Ranks the universe by 12-1 month momentum (12-month return, skipping the latest month), excludes overbought stocks (RSI > 70), and keeps the top 25. MACD is reported to confirm the trend.
4. **ARIMA views.** `auto_arima` on each stock's quarterly **log returns**, with no seasonal term. The next-quarter forecast becomes that stock's Black-Litterman view.
5. **Black-Litterman.** A market-cap-implied prior blended with the ARIMA views, all in annual excess returns. The optimiser maximises Sharpe, long-only, with at most 20% in any one stock.
6. **Backtest.** The weights are held buy-and-hold for one quarter, then the whole process is re-run using only data up to the next rebalance date.

### Results

| | CAGR 2010–2023 | Volatility | Sharpe | Max drawdown | 2023 return |
|---|---|---|---|---|---|
| **Strategy** | **12.6%** | 20.5% | **0.54** | −33.0% | 12.7% |
| Equal-weight top 50 | 10.2% | 16.8% | 0.48 | −31.1% | 23.1% |
| Cap-weight top 50 | 11.6% | 17.9% | 0.54 | −30.0% | 37.0% |
| Cap-weight S&P 500 (index proxy) | 11.1% | 17.5% | 0.52 | −33.6% | 25.6% |

These are price returns (no dividends) before transaction costs, with quarterly rebalancing. The S&P 500 proxy tracks the real index closely: 25.6% vs 24.2% in 2023, and 11.1% vs about 10.9% a year over 2010–2023.

The strategy earns more than the benchmarks but takes more risk to do it. Its Sharpe ratio is no better than holding the 50 largest stocks, and it badly lagged in 2023. The notebook's findings section covers the caveats.

### Files

| File | Purpose |
|------|---------|
| `main.ipynb` | The full pipeline, backtests and findings (runs in about 5 minutes) |
| `requirements.txt` | Package versions the notebook was verified with |
| `archive/` | Phase 1 notebooks, kept for reference |

### Data (not included)

The data is not in this repository: it is too large for GitHub and may not be redistributable. The notebook expects:

- `tickers/SPX as of <date>.xlsx`: quarterly snapshots of S&P 500 constituents, 2000–2023 (columns include `Ticker`)
- `tickers/spx_companies_historical_data.csv`: daily data for each ticker, with these columns:
  `ticker, date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, DVD_YLD_IND, EPS, BOOK_VAL_PER_SH, CUR_MKT_CAP`

### Setup

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

---

## Project phases

### Phase 1: Baseline ✅ Complete

The first working version of the pipeline (now in `archive/`).

- [x] Extracted the S&P 500 constituent snapshots from Excel
- [x] Cleaned the data: merged ticker suffixes and removed duplicate dates
- [x] Split into train (2000–2022) and test (2023)
- [x] Ran ARIMA forecasts with SMA/EMA overlays for the top 50 stocks by market cap
- [x] Computed RSI, MACD and ROC indicators
- [x] Built a Black-Litterman portfolio allocation
- [x] Built quarterly rebalancing over 2023

**Phase 1 output:** a Sharpe ratio of 12.91. A review found this came from the bugs fixed in Phase 2, so it has no meaning.

### Phase 2: Correctness fixes ✅ Complete

Status key: ⬜ To do · 🔄 In progress · ✅ Done

#### Critical: these changed the results

| # | Task | Problem | Fix | Status |
|---|------|---------|-----|--------|
| 2.1 | Fix the Black-Litterman views | `P @ tau * cov_matrix @ P.T` ran `P @ tau` first, which raised an error. The code caught it and silently fell back to the implied returns, so the views were never applied. | Correct formula: `P @ (tau * Sigma) @ P.T`. Also fixed the posterior covariance, which used `tau * inv(Sigma)` instead of `inv(tau * Sigma)`. | ✅ Done |
| 2.2 | Fix the weight labels | The covariance matrix was in alphabetical order, but the weights were labelled in ranking order, so each printed weight belonged to a different stock. | Every vector is built from one ordered ticker list and stays labelled by ticker. | ✅ Done |
| 2.3 | Fix the return and Sharpe calculations | A 3% annual risk-free rate was added to *daily* returns, the Sharpe ratio never subtracted the risk-free rate, and the market risk premium was used as the risk-aversion coefficient. | Everything is annualised and in excess returns, with δ = MRP / market variance. Sharpe = (return − risk-free) / volatility. | ✅ Done |
| 2.4 | Remove look-ahead from stock selection | Stocks were chosen by comparing forecasts with the actual 2023 returns, then tested on 2023. | Selection uses only data up to each rebalance date. | ✅ Done |
| 2.5 | Make the backtest walk forward | Weights were fitted and scored on the same quarter, the quarterly return was the mean daily return, Q1 was never scored, and the portfolio value was never reported. | Form on quarter-end *t*, hold buy-and-hold to *t+1*, and chain the values. All four 2023 quarters are scored and plotted. | ✅ Done |
| 2.6 | Add a benchmark | There was nothing to compare against. | Equal-weight top 50, cap-weight top 50 and a cap-weight S&P 500 proxy, reporting CAGR, volatility, Sharpe and maximum drawdown. Also added a 2010–2023 backtest (56 rebalances). | ✅ Done |

#### Method improvements

| # | Task | Problem | Fix | Status |
|---|------|---------|-----|--------|
| 2.7 | Use a real momentum signal | The ARIMA-accuracy screen favoured flat, defensive stocks, and RSI/MACD were not used in any decision. | Stocks are selected by 12-1 momentum with an RSI < 70 filter. ARIMA now supplies the Black-Litterman views. | ✅ Done |
| 2.8 | Improve the ARIMA setup | Fitted to raw prices with a fake quarterly seasonality. TSLA's forecast fell to $4.25. | Fitted to quarterly log returns with no seasonal term. The chosen order is refitted with an intercept so a (0,0,0) model forecasts the stock's mean return, not 0. | ✅ Done |
| 2.9 | Remove survivorship bias | The universe was each stock's latest market cap over the whole period, and delisted tickers crashed the loop. | Point-in-time universe from the constituent snapshots. Delisted stocks hold their last price. | ✅ Done |

#### Housekeeping

| # | Task | Problem | Fix | Status |
|---|------|---------|-----|--------|
| 2.10 | Make the notebook run from a fresh kernel | Wrong data path and a missing `import re`. | Fixed. Verified with a full top-to-bottom run. | ✅ Done |
| 2.11 | Speed up the data cleaning | Loops that grew a DataFrame with `pd.concat` on 6.8M rows, and helper functions that did nothing. | One `dropna` + `drop_duplicates`. Loading and cleaning now take about 10 seconds. | ✅ Done |
| 2.12 | Fix wrong labels | Printouts said "2023–2024" when the data ends on 2023-12-29. | All labels match the data. | ✅ Done |
| 2.13 | Consolidate the notebooks | Three overlapping notebooks. | Old notebooks moved to `archive/`. `main.ipynb` is the single source. | ✅ Done |
| 2.14 | Pin the dependencies | No `requirements.txt`. | Added. | ✅ Done |

**Progress: 14 / 14 done.** The notebook also went from 18MB to under 1MB: it now has 5 summary charts instead of 81 per-stock plots.

### Phase 3: Robustness (proposed)

| # | Task | Why | Status |
|---|------|-----|--------|
| 3.1 | Include dividends (total return) | Price returns understate every portfolio and favour low-yield stocks. | ⬜ To do |
| 3.2 | Model transaction costs | The strategy turns over far more than the benchmarks. | ⬜ To do |
| 3.3 | Test parameter sensitivity | Results moved from 19% to 13% in 2023 after one ARIMA change, and the 20% weight cap was chosen after seeing results. Test a grid of settings and report the spread. | ⬜ To do |
| 3.4 | Compare against simpler alternatives | Check whether ARIMA and Black-Litterman add anything over equal-weighting the top-momentum stocks. | ⬜ To do |
| 3.5 | Shrink the covariance estimate (Ledoit-Wolf) | The sample covariance of 25 stocks is noisy, and max-Sharpe amplifies the noise. | ⬜ To do |
| 3.6 | Fix ticker reuse | Merging listings by base ticker can join two different companies that used the same symbol at different times. | ⬜ To do |
