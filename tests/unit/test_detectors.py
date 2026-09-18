"""Unit tests for anomaly detectors."""

import numpy as np
import pytest

from elemental_society_analysis.models.detectors import (
    AnomalyDetectorConfig,
    ExtendedIsolationForest,
    LocalOutlierFactorDetector,
    create_detector,
)


def test_isolation_forest_fit_predict(numeric_data):
    cfg = AnomalyDetectorConfig(contamination=0.1, random_state=42)
    detector = ExtendedIsolationForest(cfg)
    detector.fit(numeric_data)
    preds = detector.predict(numeric_data)
    assert set(np.unique(preds)).issubset({-1, 1})


def test_detector_raises_if_not_fitted(numeric_data):
    detector = ExtendedIsolationForest()
    with pytest.raises(RuntimeError):
        detector.predict(numeric_data)


def test_create_detector_factory_isolation_forest():
    cfg = AnomalyDetectorConfig(method="IsolationForest")
    d = create_detector(cfg)
    assert isinstance(d, ExtendedIsolationForest)


def test_create_detector_factory_lof():
    cfg = AnomalyDetectorConfig(method="LOF")
    d = create_detector(cfg)
    assert isinstance(d, LocalOutlierFactorDetector)


def test_create_detector_factory_unknown():
    cfg = AnomalyDetectorConfig(method="banana")
    with pytest.raises(ValueError):
        create_detector(cfg)
