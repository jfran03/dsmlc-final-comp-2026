"""
Dense Autoencoder anomaly detector (sklearn-only, no deep learning framework needed).

Trains a bottleneck MLP on normal training data.  At inference time,
reconstruction error (MSE per timestep) is the anomaly score.

Architecture: input_dim -> 64 -> 32 -> 16 -> 32 -> 64 -> input_dim
Implemented as two MLPRegressors sharing the bottleneck representation,
or more simply as a single MLPRegressor that maps input -> input with
hidden layers acting as the encoder/decoder.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


class DenseAutoencoder:
    """
    Reconstruction-error autoencoder using sklearn's MLPRegressor.

    Parameters
    ----------
    hidden_layer_sizes : tuple
        Hidden layer sizes for the MLP (default bottleneck architecture).
    max_iter : int
        Maximum training iterations.
    random_state : int
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32, 16, 32, 64),
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter           = max_iter
        self.random_state       = random_state

        self.scaler_      = RobustScaler()
        self.model_       : MLPRegressor | None = None
        self.feature_cols_: list[str]           = []
        self.threshold_   : float               = 0.0   # mean + 3*std on training MSE

    # ------------------------------------------------------------------
    def _select_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only float-compatible columns used during fit."""
        numeric = X.select_dtypes(include=np.number)
        if self.feature_cols_:
            cols = [c for c in self.feature_cols_ if c in numeric.columns]
            return numeric[cols]
        return numeric

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame) -> "DenseAutoencoder":
        """
        Fit scaler and MLP on normal training rows.

        Parameters
        ----------
        X_train : DataFrame — normal operation rows (already feature-engineered).
        """
        Xn = self._select_cols(X_train)
        # drop columns with all-nan or near-zero variance
        Xn = Xn.loc[:, Xn.std() > 1e-6].dropna(axis=1, how="all")
        self.feature_cols_ = Xn.columns.tolist()

        # impute any remaining nans with column mean
        Xn = Xn.fillna(Xn.mean())

        Xs = self.scaler_.fit_transform(Xn.values)

        # Clip hidden layers to input dimensionality to avoid over-parameterisation
        # on small feature sets
        dim = Xs.shape[1]
        sizes = tuple(min(s, max(dim, 4)) for s in self.hidden_layer_sizes)

        self.model_ = MLPRegressor(
            hidden_layer_sizes=sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )
        self.model_.fit(Xs, Xs)   # reconstruction target = input

        # Calibrate anomaly threshold on training data
        train_mse = self._reconstruction_mse(Xs)
        self.threshold_ = float(train_mse.mean() + 3 * train_mse.std())
        return self

    # ------------------------------------------------------------------
    def _reconstruction_mse(self, Xs: np.ndarray) -> np.ndarray:
        """Per-row MSE between input and reconstruction."""
        recon = self.model_.predict(Xs)
        return np.mean((Xs - recon) ** 2, axis=1)

    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute per-timestep reconstruction MSE anomaly score.

        Parameters
        ----------
        X : DataFrame with the same feature columns as training.

        Returns
        -------
        mse : np.ndarray of shape (len(X),)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before score().")

        Xn = self._select_cols(X)
        Xn = Xn.fillna(Xn.mean() if not Xn.empty else 0)
        # Ensure column order matches training
        Xn = Xn.reindex(columns=self.feature_cols_, fill_value=0.0)
        Xs = self.scaler_.transform(Xn.values)
        return self._reconstruction_mse(Xs)

    # ------------------------------------------------------------------
    def per_feature_error(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Per-timestep, per-feature squared reconstruction error (in scaled space).

        Returns
        -------
        DataFrame of shape (n_timesteps, n_features) — columns match feature_cols_.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before per_feature_error().")
        Xn = self._select_cols(X)
        Xn = Xn.fillna(Xn.mean() if not Xn.empty else 0)
        Xn = Xn.reindex(columns=self.feature_cols_, fill_value=0.0)
        Xs    = self.scaler_.transform(Xn.values)
        recon = self.model_.predict(Xs)
        return pd.DataFrame((Xs - recon) ** 2, columns=self.feature_cols_)

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.feature_cols_)
