# Momentum Trading Strategy

S&P 500 stock selection and portfolio allocation (2000–2023), in `main.ipynb`:

1. Clean and de-duplicate daily price data
2. Train/test split: 2000–2022 / 2023
3. ARIMA forecasts on quarterly prices for the top 50 stocks by market cap
4. Momentum indicators (RSI, MACD, ROC)
5. Black-Litterman portfolio allocation with quarterly rebalancing

Other files: `main-copy-pdf.ipynb` (earlier version, includes the Excel-extraction step), `draft-assets.ipynb` (drafts), `data-cleaning.ipynb`, and HTML exports of the notebooks.

## Data

The data is not included in this repo. The notebooks expect:

- `tickers/`: quarterly S&P 500 constituent snapshots (`SPX as of <date>.xlsx`) and `spx_companies_historical_data.csv` (columns: `ticker, date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, DVD_YLD_IND, EPS, BOOK_VAL_PER_SH, CUR_MKT_CAP`)

## Status

Work in progress. Known issues are being fixed.
