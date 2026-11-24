# Next steps: ideas

[← Back to the README](../README.md)

Phase 5 covered the signal research in section 2 below (evaluating each signal, the historical-mean follow-up, extending back to 2005) and replaced Idzorek-style view confidence with Grinold alphas. Phase 6 covered section 4 (benchmark-relative optimisation with a tracking-error budget and a turnover penalty, with the transfer coefficient reported) and the package and unit tests in section 7. Phase 7 tested residual and volatility-managed momentum (section 2) and diagnosed the risk forecasts: the risk model is accurate, and aligning it with the alpha fixes the forecasts for the historical-mean strategy. Phase 8 covered the factor attribution and real T-bill rates (section 3) and the deflated Sharpe ratio, bootstrap tests and PBO (section 5). What remains is mainly **signals with real information in large caps**, which none tested so far have, and **out-of-sample confirmation** on data from 2024 onwards.

### What a re-check of the results found

These checks were run on the existing pipeline (2010–2023, top-50 universe, after 10 bps costs). They change where the project should go next.

| Check | Result | What it means |
|---|---|---|
| ARIMA orders chosen (4,049 fits) | 63% are (0,0,0) and another 12% are (0,0,1) | The "ARIMA views" are mostly each stock's historical average return |
| Beta and alpha vs the S&P 500 proxy (daily, Newey-West) | Beta 1.01, alpha +1.45% a year, t = 0.57 | The extra volatility is not market exposure; it is stock-specific risk |
| Concentration | Median 7 names held; 3 or more names at the 20% cap in 73% of rebalances | The portfolio is a few concentrated bets, so low t-statistics are almost built in |
| Rank IC: how well each signal ranks next-quarter returns (56 quarters) | ARIMA view +0.065 (t 2.3), plain historical mean +0.079 (t 2.5), 12-1 momentum +0.015 (t 0.5) | The ARIMA views are the only signal with measurable skill; momentum has almost none in this universe |
| One-day execution lag (decide on the close, trade the next day) | Sharpe 0.52 → 0.51 | Trading on the close the signal was observed at does not inflate the results |

**The main finding:** the forecasts rank stocks better than chance, but the portfolio construction does not turn that into returns. The fundamental law of active management (IR ≈ TC × IC × √breadth) gives a theoretical information ratio of about 0.9 for an IC of 0.065 across 50 stocks × 4 quarters a year. The actual IR is 0.15, so the transfer coefficient (TC, how much of the signal reaches the portfolio) is only about 0.16.

**Caveat:** 2010–2023 was a decade in which long-run winners such as mega-cap tech kept winning, so the historical-mean IC may simply reflect that. It needs testing on 2003–2009 and controlling for size and sector before it is trusted.

### 1. Reframe the research question
| Idea | Why |
|---|---|
| ~~Lead with "Does a forecast-driven Black-Litterman overlay add value to large-cap momentum?"~~ | Done: the title question of the research note |
| ~~Write a 3–5 page research note (PDF)~~ | Done: [`note/research-note.pdf`](../note/research-note.pdf), linked at the top of the README |

### 2. Signal research
| Idea | Why |
|---|---|
| Evaluate each signal on its own before building portfolios | Standard practice: IC, IC decay at 1, 3, 6 and 12 months, quintile long-short spreads, and how often the signal's ranking changes |
| Follow up the historical-mean IC | Control for size and sector, and test 2003–2009, to see whether it is real skill or just the mega-cap tech decade |
| ~~Other momentum definitions~~ | Done: 6-1 and 52-week high (Phase 5), residual momentum (Phase 7); none has a premium in the S&P 500 |
| ~~Volatility-scaled momentum~~ | Done in Phase 7: removes the factor's crashes, adds nothing to the long-only top-50 portfolio |
| Multi-factor model with book-to-price | Value and momentum tend to be negatively correlated, so combining them usually gives a smoother result than either alone. `EPS` is empty in the data, so value is book-to-price only (lagged 3 months; see Phase 5) |
| Market trend filter | Hold cash or the index when the S&P 500 is below its 200-day average, to avoid momentum's worst periods |
| Monthly rebalancing | Momentum signals decay quickly; quarterly may be too slow, though costs rise |
| Cross-sectional model instead of ARIMA | For example gradient boosting on momentum, value and volatility features to rank stocks, instead of 25 separate time-series models |

### 3. Performance attribution
| Idea | Why |
|---|---|
| ~~Factor regressions against Fama-French 5 factors plus momentum~~ | Done in Phase 8: the baseline's extra return is a momentum-and-growth tilt; no factor alpha is significant |
| ~~Use real T-bill rates for the risk-free rate~~ | Done in Phase 8, for every Sharpe ratio and, as a check, inside the baseline's views (Sharpe 0.625 → 0.623) |
| Sector classifications (GICS) | Enables sector attribution, sector-neutral momentum and sector caps; in 2023 the strategy was effectively a sector bet against tech |

### 4. Portfolio construction
| Idea | Why |
|---|---|
| ~~Mean-variance against the benchmark with a tracking-error target~~ | Done in Phase 6 |
| ~~Set view confidence from forecast accuracy (Idzorek)~~ | Superseded by Grinold alphas in Phase 5 |
| ~~Turnover penalty inside the optimiser~~ | Done in Phase 6 |
| ~~Exponentially weighted risk model~~ | Tested in Phase 7: it does not help; the cause was alpha–risk misalignment, fixed with an alpha factor |
| Compare other allocators | Minimum-variance, risk parity or hierarchical risk parity on the same picks |

### 5. Making the results trustworthy
| Idea | Why |
|---|---|
| Extend the backtest back to 2003 | The data starts in 2000; 2003 is the earliest date with full lookback windows. It adds the 2008 crisis and the 2009 rebound, when momentum famously crashed |
| ~~Deflated Sharpe ratio, block-bootstrap confidence intervals and the probability of backtest overfitting (PBO)~~ | Done in Phase 8 across all 47 variants |
| Out-of-sample confirmation | Fix the best settings (such as partial rebalancing) now and test them only on new data from 2024 onwards |
| Walk-forward parameter selection | Let the backtest choose its settings using only past data at each date, instead of fixing them up front |
| Random-portfolio benchmark | Compare against thousands of random portfolios of the same size to see where the strategy ranks by luck alone |

### 6. Better data and costs
| Idea | Why |
|---|---|
| Total-return data (CRSP or Bloomberg) | Unblocks 3.1 and removes special-dividend artifacts such as KDP's −82% "crash" in 2018 |
| Spread plus square-root market-impact cost model | The data already has `PX_VOLUME`, which is enough to model impact and estimate how much capital the strategy could run |

### 7. Engineering
| Idea | Why |
|---|---|
| ~~Move the code into a Python package with unit tests~~ | Done in Phases 6–8 (`mts/`, `tests/`); every notebook, `main.ipynb` included, runs on it. Still to do: test against the published He-Litterman example |
| Synthetic sample dataset | Lets anyone run the notebook end-to-end without Bloomberg data |
| Continuous integration (GitHub Actions) | Run the tests and a small backtest on every push |
| Config file instead of notebook constants | Run experiments from the command line and log each run's settings and results |
| Interactive dashboard (Streamlit or Plotly) | Let readers change settings and see the backtest update |
