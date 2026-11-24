# Project structure, data and setup

[← Back to the README](../README.md)

## Files

| File | Purpose |
|------|---------|
| `main.ipynb` | The baseline strategy: pipeline, backtests, analysis and findings, run on the `mts` package (about 4 minutes once the ARIMA cache exists) |
| `signal-research.ipynb` | Phase 5: signal tests (IC, quintiles, Fama-MacBeth) and Grinold-alpha portfolios (about 1 minute once `main.ipynb` has built the ARIMA cache) |
| `portfolio-construction.ipynb` | Phase 6: risk model, benchmark-relative optimiser, forecast vs realised risk, sensitivity (about 4 minutes) |
| `momentum-variants-and-risk.ipynb` | Phase 7: residual and volatility-managed momentum, risk-forecast diagnostics, alpha-aligned risk model (about 4 minutes) |
| `attribution-and-inference.ipynb` | Phase 8: all 47 variants rebuilt; T-bill Sharpe ratios, factor attribution, bootstrap tests, deflated Sharpe ratio, probability of backtest overfitting (about 8 minutes the first time, then cached) |
| `mts/` | Package used by every notebook: `data` (cleaning, point-in-time universe), `baseline` (the `main.ipynb` strategy), `arima` (the ARIMA views), `backtest`, `signals`, `risk`, `optimize`, `strategies` (every variant tested), `factors` (Fama-French download), `inference` (bootstrap, deflated Sharpe, PBO), `plotting`, `config` |
| `tests/` | Unit tests on synthetic data (Black-Litterman identities, Grinold alpha, point-in-time IC, risk model, optimiser constraints); no Bloomberg data needed |
| `results/` | Tables behind every Phase 5–8 number in the README, the docs and the research note, as CSV |
| `note/` | The research note (`research-note.pdf`), its LaTeX source, and `build_note.py`, which fills every number in it from `results/` and compiles it with `pdflatex` |
| `factors/` | Fama-French factor files (downloaded, not committed) and `manifest.json` with their URLs, checksums and CRSP vintage |
| `images/` | Charts exported by the notebooks and used in the README and docs |
| `requirements.txt` | Package versions the notebook was verified with |
| `archive/` | Phase 1 notebooks, kept for reference |
| `cache/` | ARIMA forecasts, cleaned data and the monthly signal panel, created by the notebooks (not committed) |

## Data (not included)

The data is not in this repository: it is too large for GitHub and may not be redistributable. The notebook expects:

- `tickers/SPX as of <date>.xlsx`: quarterly snapshots of S&P 500 constituents, 2000–2023 (columns include `Ticker`)
- `tickers/spx_companies_historical_data.csv`: daily data for each ticker, with these columns:
  `ticker, date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, DVD_YLD_IND, EPS, BOOK_VAL_PER_SH, CUR_MKT_CAP`

Prices are split-adjusted. `DVD_YLD_IND` and `EPS` are empty in every row. `BOOK_VAL_PER_SH` is stamped at the fiscal period end, before it is published, so Phase 5 lags it by about 3 months.

Phase 8 also uses the daily Fama-French five factors, the daily momentum factor and the monthly 1-month T-bill return from Kenneth French's data library, downloaded by `python -m mts.factors` (public; recorded in `factors/manifest.json`).

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/                       # unit tests, no data needed (about 2 seconds)
jupyter notebook main.ipynb                   # baseline; builds the ARIMA cache used by Phases 5 and 6
python -m mts.factors                         # Fama-French factors and T-bill rate for Phase 8
jupyter nbconvert --to notebook --execute --inplace signal-research.ipynb portfolio-construction.ipynb \
    momentum-variants-and-risk.ipynb attribution-and-inference.ipynb
python note/build_note.py                     # rebuild the research note from results/ (needs pdflatex)
```

`signal-research.ipynb` first checks that the `mts` package reproduces the baseline exactly (Sharpe 0.52,
453% turnover), and `attribution-and-inference.ipynb` checks results reported in Phases 1–6 before using them.
