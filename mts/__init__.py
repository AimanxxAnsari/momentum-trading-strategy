"""Momentum trading strategy research package.

Modules:
    config    settings shared by every notebook
    data      loading, cleaning, point-in-time universe (lazy; cached in cache/)
    baseline  main.ipynb's strategy: momentum selection, ARIMA views, Black-Litterman, max-Sharpe
    backtest  walk-forward engine with drifting holdings and costs; performance statistics
    signals   Phase 5 signal research: panel, IC, quintiles, Fama-MacBeth, Grinold alphas
    risk      Phase 6 statistical factor risk model
    optimize  Phase 6 benchmark-relative optimiser with risk and turnover penalties
    plotting  shared chart style
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
