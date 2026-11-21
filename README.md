# Momentum Trading Strategy

A research project that selects S&P 500 stocks with a momentum signal, forms return views with ARIMA, and allocates the portfolio with Black-Litterman. The strategy is tested in a quarterly walk-forward backtest over 2023 and over 2010–2023, after transaction costs.

> **Status:** Phases 1–3 are complete (except dividends, which the data does not support). **Conclusion:** over 2010–2023 the strategy returned 12.2% a year against 11.1% for an S&P 500 proxy, but its Sharpe ratio is the same (0.52 vs 0.52) and the outperformance is not statistically significant (t-statistic 0.58). See [Results](#results).

---

## How the pipeline works

All the main work is in [`main.ipynb`](main.ipynb). On each rebalance date (the last trading day of each quarter), it runs:

```
Clean data ─▶ Point-in-time universe ─▶ Momentum selection ─▶ ARIMA views ─▶ Black-Litterman ─▶ Hold one quarter (pay costs)
```

1. **Data cleaning.** Removes exchange suffixes from tickers (`AAPL UW Equity` → `AAPL`). Drops blank rows (28% of the raw file) *before* de-duplicating. When a company has two listings with different prices, keeps the one named in that date's S&P 500 snapshot.
2. **Point-in-time universe.** The 50 largest stocks that were **in the S&P 500 on that date**, taken from the quarterly constituent snapshots. This avoids survivorship bias.
3. **Momentum selection.** Ranks the universe by 12-1 month momentum (12-month return, skipping the latest month), excludes overbought stocks (RSI > 70), and keeps the top 25.
4. **ARIMA views.** `auto_arima` on each stock's quarterly **log returns**, with no seasonal term and the intercept forced on. The next-quarter forecast becomes that stock's Black-Litterman view. Forecasts are fitted in parallel and cached.
5. **Black-Litterman.** A market-cap-implied prior blended with the ARIMA views, all in annual excess returns, using a Ledoit-Wolf shrunk covariance. The optimiser maximises Sharpe, long-only, with at most 20% in any one stock.
6. **Backtest.** Holdings drift with prices for one quarter, then the process re-runs using only data up to the next rebalance date. Every portfolio, benchmarks included, pays 10 bps per unit traded.

### Results

2010–2023, after 10 bps costs, price returns:

| | CAGR | Volatility | Sharpe | Max drawdown | Turnover / yr | t-stat vs S&P | 2023 |
|---|---|---|---|---|---|---|---|
| **Strategy** | **12.2%** | 20.4% | **0.52** | −33.0% | 453% | 0.58 | 11.5% |
| Equal-weight top 50 | 10.1% | 16.8% | 0.48 | −31.1% | 67% | −1.46 | 23.1% |
| Cap-weight top 50 | 11.5% | 17.9% | 0.53 | −30.0% | 22% | 0.51 | 37.2% |
| Cap-weight S&P 500 (index proxy) | 11.1% | 17.5% | 0.52 | −33.6% | 8% | — | 25.7% |

The S&P 500 proxy tracks the real index closely: 25.6% vs 24.2% before costs in 2023, and 11.1% vs about 10.9% a year over 2010–2023.

**What the analysis shows**

- **No significant edge.** The strategy earns about 1.6% a year more than the S&P proxy, but with more volatility and the same Sharpe ratio. A t-statistic of 0.58 is well short of the ~2 needed to rule out luck.
- **No layer clearly adds value.** Adding momentum selection, then Black-Litterman, then ARIMA views moves the Sharpe ratio from 0.48 to 0.49, 0.49 and 0.52, while turnover rises from 67% to 453% a year.
- **The result is not fragile, just unremarkable.** Across 13 setting and cost variants the Sharpe ranges from 0.43 to 0.60, the best t-statistic is about 1.2, and the advantage disappears at costs of about 45 bps.
- **The two halves agree.** Sharpe 0.58 vs 0.53 for the S&P proxy in 2010–2016, and 0.50 vs 0.52 in 2017–2023.

### Files

| File | Purpose |
|------|---------|
| `main.ipynb` | The full pipeline, backtests, analysis and findings (about 5 minutes on the first run, 2 minutes once the ARIMA cache exists) |
| `requirements.txt` | Package versions the notebook was verified with |
| `archive/` | Phase 1 notebooks, kept for reference |
| `cache/` | ARIMA forecast cache, created by the notebook (not committed) |

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

### Phase 3: Robustness ✅ Complete (1 task blocked by data)

| # | Task | Outcome | Status |
|---|------|---------|--------|
| 3.1 | Include dividends (total return) | The dataset's dividend field (`DVD_YLD_IND`) is empty in every row, so total returns cannot be computed. Needs a total-return series (for example Bloomberg `TOT_RETURN_INDEX_GROSS_DVDS`) or dividend history. All results are labelled as price returns. | ⛔ Blocked (no data) |
| 3.2 | Model transaction costs | Holdings now drift between rebalances, turnover is measured against the drifted weights, and every portfolio pays 10 bps per unit traded. The strategy trades 453% a year against 8% for the S&P proxy; its advantage disappears at about 45 bps. | ✅ Done |
| 3.3 | Test parameter sensitivity | Varied the weight cap, number of stocks, RSI filter, τ, covariance method and costs one at a time, and split the period in half. Sharpe ranges from 0.43 to 0.60 (S&P: 0.52); none is significant. Also added information ratios and t-statistics against the S&P proxy. | ✅ Done |
| 3.4 | Compare against simpler alternatives | Layer-by-layer comparison: momentum adds +0.01 Sharpe, the optimiser +0.00, ARIMA views +0.03 (with about double the turnover). | ✅ Done |
| 3.5 | Shrink the covariance estimate (Ledoit-Wolf) | Now the default. It makes no measurable difference here, because the 20% weight cap already limits how far the optimiser can chase noise. Also found that τ has no effect on the posterior mean, since the He-Litterman Ω is proportional to τ. | ✅ Done |
| 3.6 | Fix ticker reuse | Investigated: Bloomberg already gives delisted companies unique codes, so no two companies share a symbol (the only name change in the snapshots is a rename, Boston Properties → BXP). The real issue was duplicate listings disagreeing on price (740 of 421k overlapping dates differ by more than 0.5%). Cleaning now keeps the listing named in each date's snapshot. | ✅ Done |

**Progress: 5 / 6 done, 1 blocked.**

### Possible next steps

- Get total-return data to complete 3.1.
- Test momentum on the full S&P 500 rather than the 50 largest stocks; momentum is usually stronger outside mega-caps.
- Lower turnover, for example by only trading when weights move by more than a threshold.
