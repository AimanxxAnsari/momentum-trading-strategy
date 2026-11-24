# Project phases

[← Back to the README](../README.md)

What each phase set out to do and what it found, task by task. Corrections are recorded where they were made.

### Phase 1: Baseline

The first working version of the pipeline (now in `archive/`).

- Extracted the S&P 500 constituent snapshots from Excel
- Cleaned the data: merged ticker suffixes and removed duplicate dates
- Split into train (2000–2022) and test (2023)
- Ran ARIMA forecasts with SMA/EMA overlays for the top 50 stocks by market cap
- Computed RSI, MACD and ROC indicators
- Built a Black-Litterman portfolio allocation
- Built quarterly rebalancing over 2023

**Phase 1 output:** a Sharpe ratio of 12.91. A review found this came from the bugs fixed in Phase 2, so it has no meaning.

### Phase 2: Correctness fixes

#### Critical: these changed the results

| # | Task | Problem | Fix |
|---|------|---------|-----|
| 2.1 | Fix the Black-Litterman views | `P @ tau * cov_matrix @ P.T` ran `P @ tau` first, which raised an error. The code caught it and silently fell back to the implied returns, so the views were never applied. | Correct formula: `P @ (tau * Sigma) @ P.T`. Also fixed the posterior covariance, which used `tau * inv(Sigma)` instead of `inv(tau * Sigma)`. |
| 2.2 | Fix the weight labels | The covariance matrix was in alphabetical order, but the weights were labelled in ranking order, so each printed weight belonged to a different stock. | Every vector is built from one ordered ticker list and stays labelled by ticker. |
| 2.3 | Fix the return and Sharpe calculations | A 3% annual risk-free rate was added to *daily* returns, the Sharpe ratio never subtracted the risk-free rate, and the market risk premium was used as the risk-aversion coefficient. | Everything is annualised and in excess returns, with δ = MRP / market variance. Sharpe = (return − risk-free) / volatility. |
| 2.4 | Remove look-ahead from stock selection | Stocks were chosen by comparing forecasts with the actual 2023 returns, then tested on 2023. | Selection uses only data up to each rebalance date. |
| 2.5 | Make the backtest walk forward | Weights were fitted and scored on the same quarter, the quarterly return was the mean daily return, Q1 was never scored, and the portfolio value was never reported. | Form on quarter-end *t*, hold buy-and-hold to *t+1*, and chain the values. All four 2023 quarters are scored and plotted. |
| 2.6 | Add a benchmark | There was nothing to compare against. | Equal-weight top 50, cap-weight top 50 and a cap-weight S&P 500 proxy, reporting CAGR, volatility, Sharpe and maximum drawdown. Also added a 2010–2023 backtest (56 rebalances). |

#### Method improvements

| # | Task | Problem | Fix |
|---|------|---------|-----|
| 2.7 | Use a real momentum signal | The ARIMA-accuracy screen favoured flat, defensive stocks, and RSI/MACD were not used in any decision. | Stocks are selected by 12-1 momentum with an RSI < 70 filter. ARIMA now supplies the Black-Litterman views. |
| 2.8 | Improve the ARIMA setup | Fitted to raw prices with a fake quarterly seasonality. TSLA's forecast fell to $4.25. | Fitted to quarterly log returns with no seasonal term. The chosen order is refitted with an intercept so a (0,0,0) model forecasts the stock's mean return, not 0. |
| 2.9 | Remove survivorship bias | The universe was each stock's latest market cap over the whole period, and delisted tickers crashed the loop. | Point-in-time universe from the constituent snapshots. Delisted stocks hold their last price. |

#### Housekeeping

| # | Task | Problem | Fix |
|---|------|---------|-----|
| 2.10 | Make the notebook run from a fresh kernel | Wrong data path and a missing `import re`. | Fixed. Verified with a full top-to-bottom run. |
| 2.11 | Speed up the data cleaning | Loops that grew a DataFrame with `pd.concat` on 6.8M rows, and helper functions that did nothing. | One `dropna` + `drop_duplicates`. Loading and cleaning now take about 10 seconds. |
| 2.12 | Fix wrong labels | Printouts said "2023–2024" when the data ends on 2023-12-29. | All labels match the data. |
| 2.13 | Consolidate the notebooks | Three overlapping notebooks. | Old notebooks moved to `archive/`. `main.ipynb` is the single source. |
| 2.14 | Pin the dependencies | No `requirements.txt`. | Added. |

The notebook also went from 18MB to under 1MB: it now has 5 summary charts instead of 81 per-stock plots.

### Phase 3: Robustness (1 task blocked by data)

| # | Task | Outcome |
|---|------|---------|
| 3.1 | Include dividends (total return) | The dataset's dividend field (`DVD_YLD_IND`) is empty in every row, so total returns cannot be computed. Needs a total-return series (for example Bloomberg `TOT_RETURN_INDEX_GROSS_DVDS`) or dividend history. All results are labelled as price returns. |
| 3.2 | Model transaction costs | Holdings now drift between rebalances, turnover is measured against the drifted weights, and every portfolio pays 10 bps per unit traded. The strategy trades 453% a year against 8% for the S&P proxy; its advantage disappears at about 45 bps. |
| 3.3 | Test parameter sensitivity | Varied the weight cap, number of stocks, RSI filter, τ, covariance method and costs one at a time, and split the period in half. Sharpe ranges from 0.43 to 0.60 (S&P: 0.52); none is significant. Also added information ratios and t-statistics against the S&P proxy. |
| 3.4 | Compare against simpler alternatives | Layer-by-layer comparison: momentum adds +0.01 Sharpe, the optimiser +0.00, ARIMA views +0.03 (with about double the turnover). |
| 3.5 | Shrink the covariance estimate (Ledoit-Wolf) | Now the default. It makes no measurable difference here, because the 20% weight cap already limits how far the optimiser can chase noise. Also found that τ has no effect on the posterior mean, since the He-Litterman Ω is proportional to τ. |
| 3.6 | Fix ticker reuse | Investigated: Bloomberg already gives delisted companies unique codes, so no two companies share a symbol (the only name change in the snapshots is a rename, Boston Properties → BXP). The real issue was duplicate listings disagreeing on price (740 of 421k overlapping dates differ by more than 0.5%). Cleaning now keeps the listing named in each date's snapshot. |

### Phase 4: Extensions

| # | Task | Outcome |
|---|------|---------|
| 4.1 | Run momentum on the full S&P 500 | All ~500 members, point-in-time. Results got worse: strategy Sharpe 0.29, classic top-decile momentum 0.33, against 0.47 for equal-weight S&P 500. Added an equal-weight S&P 500 benchmark. |
| 4.2 | Reduce turnover | Added no-trade bands and partial rebalancing to the backtest engine. Partial rebalancing (50%) halves turnover and lifts the Sharpe to 0.58 (t-statistic 1.15). Bands barely help, because trades are whole-position swaps. |
| 4.3 | Visual report | 12 charts, built on one colour-blind-checked palette with a fixed colour per portfolio, exported to `images/` and shown in the [results](results.md). |
| 3.1 | Dividends (carried over) | Still blocked. The free alternative (Yahoo Finance) has no data for most delisted companies, so it would cover the survivors only and reintroduce survivorship bias. Needs a point-in-time total-return source (CRSP, or Bloomberg `TOT_RETURN_INDEX_GROSS_DVDS`). |

### Phase 5: Signal research

| # | Task | Outcome |
|---|------|---------|
| 5.1 | Test each signal on its own | Eight signals, monthly 2003–2023, full S&P 500 and top 50: rank IC at 1/3/6/12 months with Newey-West t-statistics, three sub-periods, quintile portfolios, Fama-MacBeth regressions. No momentum premium; the historical-mean IC comes from 2017–2023 and is not significant after controls |
| 5.2 | Grinold alphas into Black-Litterman | alpha = IC × volatility × z-score with point-in-time IC estimates; prior over all 50 stocks; transfer coefficient measured (about 0.70). Backtested 2010–2023 and 2005–2023 |
| 5.3 | Check the data before using it | Prices are split-adjusted; `EPS` is empty; book value has about one quarter of look-ahead, so it is lagged |
| 5.4 | Correct the ARIMA views from median to expected returns | Found in review: e^(4m) − 1 understates the expected return by about σ²/2. Tested as a variant (Sharpe 0.52 → 0.60); `main.ipynb` itself is unchanged |

### Phase 6: Portfolio construction

| # | Task | Outcome |
|---|------|---------|
| 6.1 | Statistical risk model | 5 principal components of 3 years of daily returns plus specific risk, refitted each quarter; explains 63% of variance on average. Gives specific volatility for the alphas and splits active risk into factor and specific parts |
| 6.2 | Benchmark-relative optimiser | Alpha minus risk and trading penalties, tracking-error cap, long-only, 20% cap; starts from the drifted holdings so each trade's cost counts. Turnover 453% → 25–40%, tracking error 0.8–1.6% |
| 6.3 | Check the risk model's forecasts | Bias statistic 1.17–1.57: realised tracking error is higher than forecast |
| 6.4 | Package and tests | `research.py` replaced by the `mts/` package; 20 unit tests on synthetic data; every Phase 5 and 6 table exported to `results/`. The Phase 5 notebook gives the same numbers on the package |

### Phase 7: Momentum variants and risk forecasts

| # | Task | Outcome |
|---|------|---------|
| 7.1 | Residual momentum | 36-month market-model residuals, 12-1 window, scaled by residual volatility. Steadier than 12-1 momentum in the top 50, no premium in the S&P 500 |
| 7.2 | Volatility-managed momentum | Momentum factor: drawdown −71.8% → −29.5%, Sharpe −0.05 → 0.31 (t 1.50). No effect in the long-only portfolio |
| 7.3 | Exponentially weighted risk model | Half-lives of 126, 63 and 42 days make the forecasts slightly worse; it was the wrong diagnosis in Phase 6 |
| 7.4 | Diagnose the risk forecasts | Accurate for random portfolios (0.98) and specific risk (0.98); the alpha-aligned model fixes the historical-mean portfolio (1.17 → 1.10); momentum's error comes from two extreme quarters |
| 7.5 | Code | `risk.add_alpha_factor`, exponential weights, `signals.residual_momentum`, `signals.vol_scale`, `backtest.bias_statistic`; 29 unit tests in all |

### Phase 8: Attribution and robust inference

| # | Task | Outcome |
|---|------|---------|
| 8.1 | Fama-French data and the real risk-free rate | Daily five factors and momentum, monthly T-bill (CRSP vintage 202609), with a manifest of URLs and checksums. T-bill averaged 0.84% a year; the baseline's Sharpe ratio is 0.625 vs 0.640 for the S&P 500 proxy. With the T-bill also inside the views (a correction check, not a trial): 0.623 |
| 8.2 | Every variant in one place | `mts.strategies` rebuilds all 47 variants from Phases 1–7 and reproduces their reported results |
| 8.3 | Factor attribution | Relative to the S&P 500 proxy: momentum-and-growth tilts for the baseline, mega-cap tilts for the optimised portfolios; factor alpha t-statistics from −1.79 to +1.02 |
| 8.4 | Sharpe-ratio tests | Block bootstrap: no difference significant (lowest p-value 0.37) |
| 8.5 | Multiple-testing corrections | Deflated Sharpe ratio of the best variant 0.65–0.85; PBO 0.49 and 0.65 within each group of variants |
| 8.6 | Code | `mts.factors`, `mts.inference`, `mts.strategies`, `mts.arima`; `main.ipynb` now runs on the package with identical results; universe ordering made deterministic; 37 unit tests in all |
