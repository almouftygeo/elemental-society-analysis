"""Anomaly detectors following the Strategy pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class AnomalyDetector(ABC):
    """Abstract interface for all anomaly detectors."""

    @abstractmethod
    def fit(self, X, y=None): ...

    @abstractmethod
    def decision_function(self, X): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def score_samples(self, X): ...


@dataclass
class AnomalyDetectorConfig:
    n_estimators: int = 100
    max_samples: int = 256
    contamination: float = 0.08
    random_state: int = 42
    method: str = "IsolationForest"
    n_neighbors: int = 20
    novelty: bool = True


class ExtendedIsolationForest(AnomalyDetector):
    """Isolation Forest detector with strict column enforcement."""

    def __init__(self, config: AnomalyDetectorConfig | None = None):
        self.config = config or AnomalyDetectorConfig()
        self.model: IsolationForest | None = None
        self.feature_names = None
        self._fitted = False

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=min(self.config.max_samples, len(X_array)),
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self.model.fit(X_array)
        self._fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        return self.model.decision_function(self._enforce_columns(X).values)

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(self._enforce_columns(X).values)

    def score_samples(self, X):
        self._check_fitted()
        return self.model.score_samples(self._enforce_columns(X).values)

    def get_anomalies(self, X, threshold_multiplier: float = 1.0):
        scores = self.decision_function(X)
        return np.where(scores < 0 * threshold_multiplier, -1, 1)

    def _enforce_columns(self, X) -> pd.DataFrame:
        if self.feature_names is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return X[self.feature_names].copy()
        X = np.asarray(X)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            )
        return pd.DataFrame(X, columns=self.feature_names)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


class LocalOutlierFactorDetector(AnomalyDetector):
    """LOF detector wrapper."""

    def __init__(self, config: AnomalyDetectorConfig | None = None):
        self.config = config or AnomalyDetectorConfig()
        self.model: LocalOutlierFactor | None = None
        self.feature_names = None
        self._fitted = False

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        self.model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination,
            novelty=self.config.novelty,
        )
        self.model.fit(X_array)
        self._fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        return self.model.decision_function(self._enforce_columns(X).values)

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(self._enforce_columns(X).values)

    def score_samples(self, X):
        self._check_fitted()
        return -self.model.decision_function(self._enforce_columns(X).values)

    def _enforce_columns(self, X) -> pd.DataFrame:
        if self.feature_names is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return X[self.feature_names].copy()
        return pd.DataFrame(X, columns=self.feature_names)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


def create_detector(config: AnomalyDetectorConfig) -> AnomalyDetector:
    """Factory for anomaly detectors."""
    if config.method == "IsolationForest":
        return ExtendedIsolationForest(config)
    if config.method == "LOF":
        return LocalOutlierFactorDetector(config)
    raise ValueError(f"Unknown detector method: {config.method}")
