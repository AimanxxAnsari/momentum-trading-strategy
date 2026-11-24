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


def ewma_weights(n: int, half_life: float | None) -> np.ndarray:
    """Observation weights for n days, oldest first, summing to 1. A day `half_life` days older
    counts half as much; None gives equal weights."""
    if half_life is None:
        return np.full(n, 1.0 / n)
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    return w / w.sum()


def fit_pca(rets: pd.DataFrame, n_factors: int = C.N_FACTORS, min_specific_share: float = 0.05,
            half_life: float | None = None) -> RiskModel:
    """Fit the model to daily returns (dates x tickers). Missing returns are set to the stock's mean
    before decomposing. With `half_life` (days), the covariance is exponentially weighted so recent
    days count more. Specific variance is floored at `min_specific_share` of each stock's variance,
    so no stock looks riskless after the factors are removed."""
    w = ewma_weights(len(rets), half_life)
    mean = pd.Series(np.nansum(rets.values * w[:, None], axis=0) / (rets.notna().values * w[:, None]).sum(axis=0),
                     index=rets.columns)
    X = (rets - mean).fillna(0.0).values
    if half_life is None:
        S = X.T @ X / (len(X) - 1)
    else:
        S = (X * w[:, None]).T @ X / (1 - (w ** 2).sum())    # unbiased weighted covariance
    eigval, eigvec = np.linalg.eigh(S)
    order = np.argsort(eigval)[::-1][:n_factors]
    lam, V = eigval[order], eigvec[:, order]
    common = V @ np.diag(lam) @ V.T
    total = np.diag(S)
    specific = np.maximum(total - np.diag(common), min_specific_share * total)
    return RiskModel(tickers=list(rets.columns), B=V, F=np.diag(lam) * C.TRADING_DAYS, D=specific * C.TRADING_DAYS)


def add_alpha_factor(rm: RiskModel, rets: pd.DataFrame, exposures: pd.Series, half_life: float | None = None) -> RiskModel:
    """Align the risk model with the alpha (Lee & Stefek 2008; Ceria, Saxena & Stubbs 2012).

    The optimiser bets along the alpha signal; if the model has no factor in that direction it treats
    those bets as diversifiable specific risk, and optimised portfolios end up riskier than forecast.
    This adds the signal as one more factor: its direction is the part of the exposures not already
    spanned by the existing factors (normalised to unit length), and its variance is the variance of
    returns along that direction minus the specific variance the model already counts there."""
    x = exposures.reindex(rm.tickers).fillna(0).values
    x_perp = x - rm.B @ (rm.B.T @ x)
    norm = np.linalg.norm(x_perp)
    if norm == 0:
        return rm
    u = x_perp / norm
    w = ewma_weights(len(rets), half_life)
    R = rets[rm.tickers]
    R = (R - R.mean()).fillna(0.0).values
    f = R @ u
    total = (w * f ** 2).sum() / (1 - (w ** 2).sum()) * C.TRADING_DAYS
    extra = max(total - (u ** 2) @ rm.D, 0.0)
    return RiskModel(tickers=rm.tickers, B=np.column_stack([rm.B, u]),
                     F=np.diag(np.r_[np.diag(rm.F), extra]), D=rm.D)
