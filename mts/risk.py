"""Statistical factor risk model (Phase 6).

Daily returns over the last three years are decomposed into k principal components (the common
factors) and what is left over for each stock (specific risk):

    Sigma = B F B' + D

B holds the factor loadings (n x k), F the factor covariance (k x k, diagonal for principal
components) and D the specific variances. Everything is annualised. With no sector data in the
dataset, statistical factors stand in for a fundamental model's market, sector and style factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class RiskModel:
    tickers: list
    B: np.ndarray            # loadings, n x k
    F: np.ndarray            # factor covariance, k x k (annual)
    D: np.ndarray            # specific variances, n (annual)

    @property
    def cov(self) -> np.ndarray:
        return self.B @ self.F @ self.B.T + np.diag(self.D)

    @property
    def specific_vol(self) -> pd.Series:
        return pd.Series(np.sqrt(self.D), index=self.tickers)

    @property
    def total_vol(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov)), index=self.tickers)

    def align(self, w: pd.Series) -> np.ndarray:
        return w.reindex(self.tickers).fillna(0).values

    def tracking_error(self, active: pd.Series) -> float:
        a = self.align(active)
        return float(np.sqrt(a @ self.cov @ a))

    def decompose(self, active: pd.Series) -> dict:
        """Split active variance into its common-factor and specific parts."""
        a = self.align(active)
        x = self.B.T @ a
        factor, specific = float(x @ self.F @ x), float(a ** 2 @ self.D)
        return {"factor variance": factor, "specific variance": specific,
                "specific share": specific / (factor + specific) if factor + specific > 0 else np.nan}


def fit_pca(rets: pd.DataFrame, n_factors: int = C.N_FACTORS, min_specific_share: float = 0.05) -> RiskModel:
    """Fit the model to daily returns (dates x tickers). Missing returns are set to the stock's mean
    before decomposing. Specific variance is floored at `min_specific_share` of each stock's variance,
    so no stock looks riskless after the factors are removed."""
    X = rets - rets.mean()
    X = X.fillna(0.0).values
    S = X.T @ X / (len(X) - 1)
    eigval, eigvec = np.linalg.eigh(S)
    order = np.argsort(eigval)[::-1][:n_factors]
    lam, V = eigval[order], eigvec[:, order]
    common = V @ np.diag(lam) @ V.T
    total = np.diag(S)
    specific = np.maximum(total - np.diag(common), min_specific_share * total)
    return RiskModel(tickers=list(rets.columns), B=V, F=np.diag(lam) * C.TRADING_DAYS, D=specific * C.TRADING_DAYS)
