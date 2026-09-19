"""End-to-end pipeline: load -> closure -> CLR -> LOF -> metrics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..geochem.loader import load_dataframe, detect_oxide_columns
from ..utils.config import load_config, Config


logger = logging.getLogger(__name__)


def _build_corrector(method: str):
    from ..geochem import correctors as corr

    name_map = {"CLR": "CLRCorrector", "ILR": "ILRCorrector", "ALR": "ALRCorrector"}
    cls = getattr(corr, name_map[method.upper()], None)
    if cls is None:
        raise ImportError(f"Corrector not found for {method}")
    return cls()


def _build_detector(method: str, n_neighbors: int, contamination: float):
    """Instantiate a detector from the detectors module by name.

    The detector classes expect an AnomalyDetectorConfig object, not raw kwargs.
    """
    from ..models import detectors as det

    # Try to import the config dataclass
    config_cls = getattr(det, "AnomalyDetectorConfig", None)
    if config_cls is None:
        # Fallback: try importing from models.config
        try:
            from ..models.config import AnomalyDetectorConfig as config_cls  # type: ignore
        except Exception as exc:
            raise ImportError(
                "AnomalyDetectorConfig not found in detectors.py or models.config"
            ) from exc

    method_l = method.lower()
    if method_l == "lof":
        cls = getattr(det, "LocalOutlierFactorDetector", None)
        if cls is None:
            raise ImportError("LocalOutlierFactorDetector not found")
        cfg = config_cls(n_neighbors=n_neighbors, contamination=contamination)
        return cls(cfg)

    if method_l == "isolationforest":
        cls = getattr(det, "ExtendedIsolationForest", None)
        if cls is None:
            raise ImportError("ExtendedIsolationForest not found")
        # Isolation Forest usually does not need n_neighbors; use contamination only
        try:
            cfg = config_cls(contamination=contamination)
        except TypeError:
            cfg = config_cls()
        return cls(cfg)

    raise ValueError(f"Unsupported detector: {method}")


def _prepare_core_matrix(
    df: pd.DataFrame,
    core_features: list[str],
    min_coverage: float = 0.80,
) -> tuple[pd.DataFrame, list[str]]:
    """Select high-coverage features, then drop remaining NaN rows."""
    missing = [c for c in core_features if c not in df.columns]
    if missing:
        raise ValueError(f"Core features missing in dataframe: {missing}")

    subset = df[core_features].copy()
    coverage = subset.notna().mean()

    kept = [c for c in core_features if coverage[c] >= min_coverage]
    dropped = [c for c in core_features if c not in kept]
    if dropped:
        logger.info(
            "Dropped low-coverage features (< %.0f%%): %s",
            min_coverage * 100,
            dropped,
        )

    subset = subset[kept]

    before = len(subset)
    subset = subset.dropna(how="any")
    logger.info(
        "Rows after dropping NaN in kept features: %d / %d",
        len(subset),
        before,
    )

    if subset.empty:
        raise ValueError(
            "No samples left after dropping NaN rows. "
            "Consider lowering min_coverage or imputing."
        )
    return subset, kept


def run_pipeline(config: Config | None = None) -> dict:
    config = config or load_config()

    df = load_dataframe(config.io)
    oxide_cols = detect_oxide_columns(df)
    if not oxide_cols:
        raise ValueError("No oxide columns detected.")

    # Closure check
    oxide_df = df[oxide_cols].dropna(how="any")
    if len(oxide_df) > 1:
        oxide_sums = oxide_df.sum(axis=1)
        cv = float(oxide_sums.std(ddof=1) / oxide_sums.mean())
    else:
        cv = float("nan")
    logger.info("Oxide CV=%.4f (closed=%s)", cv, cv < 0.05)

    # Core matrix
    core_matrix, kept_features = _prepare_core_matrix(
        df,
        config.io.core_features,
        min_coverage=config.io.min_coverage,
    )
    logger.info(
        "Core matrix shape=%s kept_features=%s",
        core_matrix.shape,
        kept_features,
    )

    # CLR on the core matrix
    corrector = _build_corrector(config.correction.method)
    transformed = corrector.apply(core_matrix)
    logger.info("Transformed shape=%s", transformed.shape)

    if transformed.isna().any().any():
        raise ValueError("Transformed matrix contains NaN — check zero replacement.")

    # Detector
    detector = _build_detector(
        config.anomaly.method,
        config.anomaly.n_neighbors,
        config.anomaly.contamination,
    )
    detector.fit(transformed)
    labels = np.asarray(detector.predict(transformed))
    scores = np.asarray(detector.decision_function(transformed))
    anomaly_count = int((labels == -1).sum())

    # Metrics
    out_dir = Path(config.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_samples_after_cleaning": int(transformed.shape[0]),
        "n_core_features": int(transformed.shape[1]),
        "kept_features": kept_features,
        "oxide_cv": float(cv),
        "is_closed": bool(cv < 0.05) if not np.isnan(cv) else False,
        "correction_method": config.correction.method,
        "anomaly_method": config.anomaly.method,
        "n_neighbors": int(config.anomaly.n_neighbors),
        "contamination": float(config.anomaly.contamination),
        "anomaly_count": anomaly_count,
        "anomaly_ratio": anomaly_count / max(len(transformed), 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info("Wrote reports/metrics.json")

    # Per-sample labels
    result_df = core_matrix.copy()
    result_df["anomaly_label"] = labels
    result_df["anomaly_score"] = scores
    result_df.to_csv(out_dir / "anomaly_labels.csv")
    logger.info("Wrote reports/anomaly_labels.csv")

    return {
        "metrics": metrics,
        "labels": labels,
        "scores": scores,
        "transformed": transformed,
        "result_df": result_df,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_pipeline()
    print(json.dumps(result["metrics"], indent=2))
