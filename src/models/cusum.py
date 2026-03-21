"""
CUSUM (Cumulative SUM) anomaly detector.

Fits per-sensor mean and standard deviation from normal training data,
then computes the one-sided CUSUM statistic for each sensor in the
test window. The maximum statistic across all sensors is the anomaly score.

This is the primary Earliness driver in the ensemble: CUSUM accumulates
evidence of a sustained upward drift even before individual readings
cross a hard threshold.
"""

import numpy as np
import pandas as pd


class CUSUMDetector:
    """
    One-sided CUSUM applied independently to each input sensor column.

    Parameters
    ----------
    k_factor : float
        Slack parameter as a multiple of sigma.  k = k_factor * sigma.
        Controls sensitivity to drift magnitude (default 0.5).
    h_factor : float
        Decision interval as a multiple of sigma.  h = h_factor * sigma.
        Lower values detect earlier but increase false alarms (default 4.0).
    """

    def __init__(self, k_factor: float = 0.5, h_factor: float = 4.0):
        self.k_factor  = k_factor
        self.h_factor  = h_factor
        self.mus_       : dict[str, float] = {}
        self.sigmas_    : dict[str, float] = {}
        self.k_vals_    : dict[str, float] = {}
        self.h_vals_    : dict[str, float] = {}
        self.feature_cols_: list[str]      = []

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame) -> "CUSUMDetector":
        """
        Estimate per-column (μ, σ) from normal training rows.

        Parameters
        ----------
        X_train : DataFrame of numeric sensor columns (normal operation only).
        """
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        self.feature_cols_ = numeric_cols

        for col in numeric_cols:
            vals = X_train[col].dropna().values
            if len(vals) < 5:
                self.mus_[col]    = 0.0
                self.sigmas_[col] = 1.0
            else:
                mu    = float(np.mean(vals))
                sigma = float(np.std(vals, ddof=1))
                if sigma < 1e-6:
                    sigma = 1e-6
                self.mus_[col]    = mu
                self.sigmas_[col] = sigma

            self.k_vals_[col] = self.k_factor * self.sigmas_[col]
            self.h_vals_[col] = self.h_factor * self.sigmas_[col]

        return self

    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the normalised CUSUM anomaly score for each timestep.

        Score = max_across_sensors( S(t) / h ) where S(t) is the CUSUM
        statistic normalised by the decision interval h.  A value ≥ 1.0
        means the CUSUM has crossed its decision boundary.

        Parameters
        ----------
        X : DataFrame with the same columns used during fit.

        Returns
        -------
        scores : np.ndarray of shape (len(X),)  — higher = more anomalous.
        """
        n = len(X)
        per_sensor = np.zeros((n, len(self.feature_cols_)))

        for j, col in enumerate(self.feature_cols_):
            if col not in X.columns:
                continue
            vals  = X[col].values.astype(float)
            mu    = self.mus_.get(col, 0.0)
            k     = self.k_vals_.get(col, 0.5)
            h     = self.h_vals_.get(col, 4.0)
            if h < 1e-9:
                continue

            S = 0.0
            for i in range(n):
                v = vals[i]
                if np.isnan(v):
                    per_sensor[i, j] = S / h
                    continue
                S = max(0.0, S + (v - mu) - k)
                per_sensor[i, j] = S / h   # normalised: ≥1 means boundary crossed

        # Ensemble: max normalised CUSUM across all sensors at each timestep
        return per_sensor.max(axis=1)

    # ------------------------------------------------------------------
    def score_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Like score() but returns a DataFrame with per-sensor scores and
        the ensemble max, for inspection.
        """
        n = len(X)
        data = {}

        for col in self.feature_cols_:
            if col not in X.columns:
                continue
            vals  = X[col].values.astype(float)
            mu    = self.mus_.get(col, 0.0)
            k     = self.k_vals_.get(col, 0.5)
            h     = self.h_vals_.get(col, 4.0)
            if h < 1e-9:
                continue

            S_arr = np.zeros(n)
            S = 0.0
            for i in range(n):
                v = vals[i]
                if np.isnan(v):
                    S_arr[i] = S / h
                    continue
                S = max(0.0, S + (v - mu) - k)
                S_arr[i] = S / h

            data[f"cusum_{col}"] = S_arr

        result = pd.DataFrame(data, index=X.index)
        if not result.empty:
            result["cusum_max"] = result.max(axis=1)
        else:
            result["cusum_max"] = 0.0
        return result
