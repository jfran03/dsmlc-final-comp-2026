"""
Isolation Forest anomaly detector wrapper.

Provides Reliability to the ensemble: IF handles operational transients
and multi-modal operating regimes well, reducing false alarms on
high-wind vs. low-wind state switches that can fool threshold-based methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class IFDetector:
    """
    Isolation Forest wrapper with RobustScaler preprocessing.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in training data (default 0.05).
    n_estimators : int
        Number of isolation trees (default 200).
    random_state : int
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

        self.scaler_      = RobustScaler()
        self.model_       : IsolationForest | None = None
        self.feature_cols_: list[str]              = []

    # ------------------------------------------------------------------
    def _select_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include=np.number)
        if self.feature_cols_:
            cols = [c for c in self.feature_cols_ if c in numeric.columns]
            return numeric[cols]
        return numeric

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame) -> "IFDetector":
        """
        Fit on normal training rows.

        Parameters
        ----------
        X_train : DataFrame — normal operation feature rows.
        """
        Xn = self._select_cols(X_train)
        Xn = Xn.loc[:, Xn.std() > 1e-6].dropna(axis=1, how="all")
        self.feature_cols_ = Xn.columns.tolist()

        Xn = Xn.fillna(Xn.mean())
        Xs = self.scaler_.fit_transform(Xn.values)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(Xs)
        return self

    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute per-timestep anomaly score.

        Returns the *negative* decision function (higher = more anomalous),
        so the sign convention matches CUSUM and AE scores.

        Parameters
        ----------
        X : DataFrame with the same feature columns as training.

        Returns
        -------
        scores : np.ndarray of shape (len(X),)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before score().")

        Xn = self._select_cols(X)
        Xn = Xn.fillna(Xn.mean() if not Xn.empty else 0)
        Xn = Xn.reindex(columns=self.feature_cols_, fill_value=0.0)
        Xs = self.scaler_.transform(Xn.values)

        # decision_function: positive = normal, negative = anomalous
        # negate so higher value = more anomalous
        return -self.model_.decision_function(Xs)

    # ------------------------------------------------------------------
    def feature_importance(self, X_ref: pd.DataFrame,
                           n_repeats: int = 5) -> pd.Series:
        """
        Permutation feature importance on a reference dataset.

        For each feature, shuffles it n_repeats times and measures the mean
        increase in anomaly score (higher = feature is more important for
        detecting anomalies).

        Returns
        -------
        pd.Series sorted descending, indexed by feature name.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before feature_importance().")
        Xn = self._select_cols(X_ref)
        Xn = Xn.fillna(Xn.mean() if not Xn.empty else 0)
        Xn = Xn.reindex(columns=self.feature_cols_, fill_value=0.0)
        Xs   = self.scaler_.transform(Xn.values)
        base = float(np.mean(-self.model_.decision_function(Xs)))
        rng  = np.random.default_rng(42)
        importances = {}
        for i, col in enumerate(self.feature_cols_):
            deltas = []
            for _ in range(n_repeats):
                Xp = Xs.copy()
                Xp[:, i] = rng.permutation(Xp[:, i])
                deltas.append(float(np.mean(-self.model_.decision_function(Xp))) - base)
            importances[col] = float(np.mean(deltas))
        return pd.Series(importances).sort_values(ascending=False)

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.feature_cols_)
