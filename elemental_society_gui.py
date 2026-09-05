# =====================================================================
# Elemental Society Analysis Pro v1.0.0 – Complete Integrated Application
# with Isolation Forest Anomaly Detection & Coordinated Workflow
# =====================================================================
import sys, os, re, warnings, traceback
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
from dataclasses import dataclass, field

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QComboBox, QSpinBox, QStackedWidget, QTabWidget, QHeaderView,
    QProgressBar, QFrame, QDoubleSpinBox, QCheckBox, QGroupBox, QLineEdit,
    QSplitter, QRadioButton, QButtonGroup, QAbstractItemView, QInputDialog,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QRectF, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QPen, QColor, QPainterPath, QPolygonF

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

from scipy import stats as scipystats
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

# =====================================================================
# PART 0: ANOMALY DETECTION INTERFACE (Strategy Pattern)
# =====================================================================
class AnomalyDetector(ABC):
    """Abstract interface for all anomaly detectors."""
    
    @abstractmethod
    def fit(self, X, y=None):
        pass
    
    @abstractmethod
    def decision_function(self, X):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def score_samples(self, X):
        pass


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detectors."""
    n_estimators: int = 100
    max_samples: int = 256
    contamination: float = 0.08
    random_state: int = 42
    extension_level: int = 2
    use_eif: bool = False
    method: str = "IsolationForest"  # or "LOF"
    n_neighbors: int = 20
    novelty: bool = True


class ExtendedIsolationForest(AnomalyDetector):
    """
    Extended Isolation Forest wrapper that implements AnomalyDetector interface.
    Can use either sklearn IsolationForest or true EIF if available.
    """
    
    def __init__(self, config: AnomalyDetectorConfig = None):
        self.config = config or AnomalyDetectorConfig()
        self.model = None
        self.feature_names = None
        self.threshold = None
        self._fitted = False
        
    def fit(self, X, y=None):
        """Fit the isolation forest model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create the base estimator
        if self.config.use_eif:
            try:
                # Try to import EIF if available
                from eif import IsolationForest as EIF
                self.model = EIF(
                    n_estimators=self.config.n_estimators,
                    max_samples=self.config.max_samples,
                    extension_level=self.config.extension_level,
                    random_state=self.config.random_state
                )
                self.model.fit(X_array)
                # EIF doesn't have a decision_function, we need to use predict
                # Store scores separately
                self._scores = -self.model.predict(X_array)  # negative scores for anomaly
                self.threshold = np.percentile(self._scores, 100 * self.config.contamination)
            except ImportError:
                warnings.warn("EIF not available, falling back to sklearn IsolationForest")
                self._use_sklearn_fallback(X_array)
        else:
            self._use_sklearn_fallback(X_array)
        
        self._fitted = True
        return self
    
    def _use_sklearn_fallback(self, X_array):
        """Use sklearn IsolationForest as fallback."""
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state
        )
        self.model.fit(X_array)
        self.threshold = None  # sklearn uses its own threshold
    
    def decision_function(self, X):
        """Get anomaly scores (lower = more anomalous)."""
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        
        if self.config.use_eif and self.model is not None:
            # EIF doesn't have decision_function, use our stored scores
            # For new data, we need to compute scores
            if hasattr(self.model, 'predict'):
                scores = -self.model.predict(X_checked.values)
                return scores
            else:
                raise RuntimeError("EIF model not properly initialized")
        else:
            return self.model.decision_function(X_checked.values)
    
    def predict(self, X):
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        
        if self.config.use_eif and self.model is not None:
            # EIF returns scores directly
            scores = -self.model.predict(X_checked.values)
            if self.threshold is not None:
                return np.where(scores < self.threshold, -1, 1)
            else:
                return np.where(scores < 0, -1, 1)
        else:
            return self.model.predict(X_checked.values)
    
    def score_samples(self, X):
        """Get anomaly scores (higher = more anomalous)."""
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        
        if self.config.use_eif and self.model is not None:
            return self.model.predict(X_checked.values)
        else:
            return self.model.score_samples(X_checked.values)
    
    def get_anomalies(self, X, threshold_multiplier: float = 1.0):
        """Get anomalies with adjustable threshold."""
        scores = self.decision_function(X)
        if self.threshold is not None:
            adjusted_threshold = self.threshold * threshold_multiplier
            return np.where(scores < adjusted_threshold, -1, 1)
        else:
            return self.predict(X)
    
    def _enforce_columns(self, X):
        """Ensure X has the correct columns in the right order."""
        if self.feature_names is None:
            raise RuntimeError("Model not fitted - no feature names stored")
        
        if isinstance(X, pd.DataFrame):
            # Check for missing columns
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            # Reorder columns
            return X[self.feature_names].copy()
        else:
            # Assume array-like, use stored feature count
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
            return X
    
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
    
    def save(self, path: str):
        """Save the model to disk."""
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'config': self.config,
            'fitted': self._fitted
        }, path)
    
    def load(self, path: str):
        """Load the model from disk."""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.threshold = data['threshold']
        self.config = data['config']
        self._fitted = data['fitted']
        return self


class LocalOutlierFactorDetector(AnomalyDetector):
    """Wrapper for sklearn's LocalOutlierFactor."""
    
    def __init__(self, config: AnomalyDetectorConfig = None):
        self.config = config or AnomalyDetectorConfig()
        self.model = None
        self.feature_names = None
        self._fitted = False
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination,
            novelty=self.config.novelty
        )
        self.model.fit(X_array)
        self._fitted = True
        return self
    
    def decision_function(self, X):
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        return self.model.decision_function(X_checked.values)
    
    def predict(self, X):
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        return self.model.predict(X_checked.values)
    
    def score_samples(self, X):
        self._check_fitted()
        X_checked = self._enforce_columns(X)
        return -self.model.decision_function(X_checked.values)
    
    def _enforce_columns(self, X):
        if self.feature_names is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return X[self.feature_names].copy()
        return X
    
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


def create_detector(config: AnomalyDetectorConfig) -> AnomalyDetector:
    """Factory function to create anomaly detectors."""
    if config.method == "IsolationForest":
        return ExtendedIsolationForest(config)
    elif config.method == "LOF":
        return LocalOutlierFactorDetector(config)
    else:
        raise ValueError(f"Unknown detector method: {config.method}")


# =====================================================================
# PART 1: ZERO REPLACEMENT STRATEGIES
# =====================================================================
class ZeroReplacementStrategy(ABC):
    @abstractmethod
    def replace(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    def name(self) -> str:
        return self.__class__.__name__


class ConstantReplacement(ZeroReplacementStrategy):
    def __init__(self, value: float = 1e-10):
        if value <= 0:
            raise ValueError("Replacement must be > 0 (log-ratios).")
        self.value = value

    def replace(self, data: pd.DataFrame) -> pd.DataFrame:
        n = int((data == 0).sum().sum())
        if n:
            warnings.warn(f"[zero policy] {n} zero cell(s) -> {self.value}")
        return data.mask(data == 0, self.value)

    def name(self) -> str:
        return f"ConstantReplacement({self.value:.1e})"


class HalfDetectionLimit(ZeroReplacementStrategy):
    def __init__(self, dl_map: Dict[str, float] = None):
        self.dl_map = dl_map or {}

    def replace(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        for col in out.columns:
            dl = self.dl_map.get(col.upper())
            if dl is None:
                continue
            n = int((out[col] == 0).sum())
            if n:
                warnings.warn(f"[zero policy] {col}: {n} zero cell(s) -> DL/2 = {dl/2}")
            out[col] = out[col].mask(out[col] == 0, dl / 2)
        return out

    def name(self) -> str:
        return "HalfDetectionLimit"


class MultiplicativeReplacement(ZeroReplacementStrategy):
    def __init__(self, factor: float = 0.65):
        self.factor = factor

    def replace(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in df.columns:
            min_pos = df[col][df[col] > 0].min()
            if not pd.isna(min_pos):
                df[col] = df[col].replace(0, min_pos * self.factor)
            else:
                df[col] = df[col].replace(0, 1e-10)
        return df

    def name(self) -> str:
        return f"MultiplicativeReplacement({self.factor:.2f})"


class ZeroReplacementDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zero Replacement Strategy")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select zero replacement strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Constant Replacement", "Half Detection Limit", "Multiplicative Factor"])
        self.strategy_combo.currentTextChanged.connect(self.toggle_param)
        layout.addWidget(self.strategy_combo)

        self.param_label = QLabel("Value:")
        self.param_spin = QDoubleSpinBox()
        self.param_spin.setRange(1e-12, 1e6)
        self.param_spin.setDecimals(10)
        self.param_spin.setValue(1e-10)
        layout.addWidget(self.param_label)
        layout.addWidget(self.param_spin)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self.toggle_param(self.strategy_combo.currentText())

    def toggle_param(self, text):
        if text == "Constant Replacement":
            self.param_label.setText("Constant value:")
            self.param_spin.setValue(1e-10)
            self.param_spin.setDecimals(10)
        elif text == "Half Detection Limit":
            self.param_label.setText("Fraction of DL:")
            self.param_spin.setValue(0.5)
            self.param_spin.setDecimals(3)
        else:
            self.param_label.setText("Multiplicative factor:")
            self.param_spin.setValue(0.65)
            self.param_spin.setDecimals(3)

    def get_strategy(self) -> ZeroReplacementStrategy:
        strategy = self.strategy_combo.currentText()
        value = self.param_spin.value()
        if strategy == "Constant Replacement":
            return ConstantReplacement(value=value)
        elif strategy == "Half Detection Limit":
            return HalfDetectionLimit(dl_map={})
        else:
            return MultiplicativeReplacement(factor=value)


# =====================================================================
# PART 2: CLOSURE CORRECTORS
# =====================================================================
def _helmert_submatrix(k):
    H = np.zeros((k - 1, k))
    for i in range(1, k):
        H[i-1, :i] = 1.0 / np.sqrt(i * (i + 1))
        H[i-1, i] = -i / np.sqrt(i * (i + 1))
    return H


def classify_relative_value(value: float, tolerance: float = 0.05) -> str:
    lower_limit = 1 - tolerance
    upper_limit = 1 + tolerance
    if value > upper_limit:
        return "enriched"
    if value < lower_limit:
        return "depleted"
    return "near_geometric_mean"


def format_enrichment_factor(value: float) -> str:
    if value >= 1:
        return f"+{value:.3f}x"
    else:
        return f"{value:.3f}x"


class ClosureCorrector(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    def name(self) -> str:
        return self.__class__.__name__


class CLRCorrector(ClosureCorrector):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = data.to_numpy(dtype=float)
        if (vals <= 0).any():
            raise ValueError("Non-positive values reached CLR. Zero policy must run BEFORE correction.")
        logv = np.log(vals)
        return pd.DataFrame(logv - logv.mean(axis=1, keepdims=True),
                           index=data.index, columns=data.columns)

    def apply_with_enrichment(self, data: pd.DataFrame, tolerance: float = 0.05) -> Dict:
        clr = self.apply(data)
        enrichment = np.exp(clr)
        
        result_df = enrichment.copy()
        for col in result_df.columns:
            class_col = f"interpretation_{col}"
            result_df[class_col] = result_df[col].apply(
                lambda x: classify_relative_value(x, tolerance=tolerance)
            )
            result_df[f"clr_{col}"] = clr[col]
            factor_col = f"factor_{col}"
            result_df[factor_col] = result_df[col].apply(format_enrichment_factor)
        
        sample_interpretations = []
        for idx in clr.index:
            sample = clr.loc[idx]
            enriched = sample[sample > 0].index.tolist()
            depleted = sample[sample < 0].index.tolist()
            classification_counts = {'enriched': 0, 'depleted': 0, 'near_geometric_mean': 0}
            for col in enrichment.columns:
                class_val = classify_relative_value(enrichment.loc[idx, col], tolerance=tolerance)
                classification_counts[class_val] += 1
            
            sample_interpretations.append({
                'sample': idx,
                'enriched_elements': enriched,
                'depleted_elements': depleted,
                'most_enriched': sample.idxmax() if not sample.empty else None,
                'most_depleted': sample.idxmin() if not sample.empty else None,
                'enrichment_factors': enrichment.loc[idx].to_dict(),
                'classification_summary': classification_counts
            })
        
        summary = {
            'clr_mean': clr.mean().to_dict(),
            'enrichment_mean': enrichment.mean().to_dict(),
            'most_enriched_overall': clr.mean().idxmax(),
            'most_depleted_overall': clr.mean().idxmin(),
            'range_clr': {col: (clr[col].min(), clr[col].max()) for col in clr.columns},
            'overall_classification_counts': {}
        }
        
        for col in enrichment.columns:
            counts = {'enriched': 0, 'depleted': 0, 'near_geometric_mean': 0}
            for idx in enrichment.index:
                class_val = classify_relative_value(enrichment.loc[idx, col], tolerance=tolerance)
                counts[class_val] += 1
            summary['overall_classification_counts'][col] = counts
        
        return {
            'clr': clr,
            'enrichment': enrichment,
            'result_with_interpretation': result_df,
            'sample_interpretations': sample_interpretations,
            'summary': summary
        }


class ALRCorrector(ClosureCorrector):
    def __init__(self, base: str = "SiO2"):
        self.base = base

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base not in data.columns:
            raise ValueError(f"ALR base {self.base!r} not in {list(data.columns)}")
        if (data[self.base] <= 0).any():
            raise ValueError(f"ALR base {self.base!r} has non-positive values")
        return np.log(data.div(data[self.base], axis=0)).drop(columns=[self.base])

    def name(self) -> str:
        return f"ALR(ref={self.base})"


class ILRCorrector(ClosureCorrector):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        closed = data.div(data.sum(axis=1), axis=0)
        clr = CLRCorrector().apply(closed)
        H = _helmert_submatrix(clr.shape[1])
        return pd.DataFrame(clr.to_numpy() @ H.T, index=clr.index,
                           columns=[f"ilr_{i+1}" for i in range(H.shape[0])])


# =====================================================================
# PART 3: CORRECTION SERVICE
# =====================================================================
class CorrectionService:
    REGISTRY = {'CLR': CLRCorrector, 'ALR': ALRCorrector, 'ILR': ILRCorrector}

    def __init__(self, zero_strategy: Optional[ZeroReplacementStrategy] = None):
        self._zero_strategy = zero_strategy
        self._cache: Dict[tuple, pd.DataFrame] = {}

    def set_zero_strategy(self, strategy):
        self._zero_strategy = strategy

    def prepare(self, dataset) -> pd.DataFrame:
        data = dataset.composition()
        if int((data < 0).sum().sum()):
            raise ValueError("Negative values present: log-ratios undefined.")
        
        n_zero = int((data == 0).sum().sum())
        if n_zero and self._zero_strategy is None:
            warnings.warn(f"{n_zero} zero cell(s), no strategy set -> 1e-10")
        
        if self._zero_strategy:
            return self._zero_strategy.replace(data)
        return data.replace(0, 1e-10)

    def correct(self, dataset, method: str = 'CLR', use_cache: bool = True, **kwargs):
        corrector = self._build(method, **kwargs)
        key = self._cache_key(dataset, method, kwargs)
        if use_cache and key in self._cache:
            return self._cache[key].copy()
        result = corrector.apply(self.prepare(dataset))
        if use_cache:
            self._cache[key] = result.copy()
        return result

    def _build(self, method, **kwargs):
        try:
            return self.REGISTRY[method](**kwargs)
        except KeyError:
            raise ValueError(f"Unknown correction {method!r}. Registry: {list(self.REGISTRY)}")

    def _cache_key(self, dataset, method, kwargs):
        fp = int(pd.util.hash_pandas_object(dataset.composition(), index=True).values.sum())
        strat = repr(vars(self._zero_strategy)) if self._zero_strategy else None
        return (method, tuple(sorted(kwargs.items())), strat, fp)


# =====================================================================
# PART 4: GEOCHEMICAL DATA ENCAPSULATION
# =====================================================================
class GeochemicalDataset:
    def __init__(self, data: pd.DataFrame, oxide_columns: List[str]):
        self._validate(data, oxide_columns)
        self._raw = data.copy()
        self._oxide_columns = list(oxide_columns)

    @staticmethod
    def _validate(data, oxide_columns):
        missing = set(oxide_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing oxide columns: {missing}")
        dupes = [c for c in set(oxide_columns) if list(oxide_columns).count(c) > 1]
        if dupes:
            raise ValueError(f"Duplicate oxide columns: {dupes}")
        non_num = [c for c in oxide_columns if not pd.api.types.is_numeric_dtype(data[c])]
        if non_num:
            raise ValueError(f"Non-numeric oxide columns: {non_num}")

    def composition(self):
        return self._raw[self._oxide_columns].copy()

    @property
    def raw(self):
        return self._raw.copy()

    @property
    def oxide_columns(self):
        return list(self._oxide_columns)

    @property
    def closure_cv(self):
        s = self._raw[self._oxide_columns].sum(axis=1)
        return s.std() / s.mean() if s.mean() > 0 else 0.0

    def is_closed(self, threshold: float = 0.10):
        return self.closure_cv < threshold


# =====================================================================
# PART 5: GEOCHEM ANALYZER (Facade Pattern)
# =====================================================================
class GeochemAnalyzer:
    """Facade that unifies preprocessing, transformation, and detection."""
    
    def __init__(self, correction_service: CorrectionService = None,
                 transform_method: str = 'CLR',
                 detector: AnomalyDetector = None):
        self.correction_service = correction_service or CorrectionService()
        self.transform_method = transform_method
        self.detector = detector
        self.dataset = None
        self._is_fitted = False
    
    def set_data(self, data: pd.DataFrame, oxide_columns: List[str]):
        """Set the dataset for analysis."""
        self.dataset = GeochemicalDataset(data, oxide_columns)
        return self
    
    def fit_detector(self, data: pd.DataFrame = None, oxide_columns: List[str] = None):
        """Fit the anomaly detector on the data."""
        if data is not None and oxide_columns is not None:
            self.set_data(data, oxide_columns)
        
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_data() first.")
        
        # Prepare data (zero replacement)
        prepared = self.correction_service.prepare(self.dataset)
        
        # Transform data (CLR/ALR/ILR)
        transformed = self.correction_service.correct(self.dataset, self.transform_method)
        
        if self.detector is None:
            raise ValueError("No detector set. Set a detector first.")
        
        self.detector.fit(transformed)
        self._is_fitted = True
        return self
    
    def analyze(self, data: pd.DataFrame = None, oxide_columns: List[str] = None):
        """Run full analysis: preprocess -> transform -> detect."""
        if data is not None and oxide_columns is not None:
            self.set_data(data, oxide_columns)
        
        if self.dataset is None:
            raise ValueError("No dataset set. Call set_data() first.")
        
        if not self._is_fitted:
            self.fit_detector()
        
        # Prepare and transform
        prepared = self.correction_service.prepare(self.dataset)
        transformed = self.correction_service.correct(self.dataset, self.transform_method)
        
        # Detect anomalies
        scores = self.detector.decision_function(transformed)
        predictions = self.detector.predict(transformed)
        
        return {
            'scores': scores,
            'labels': predictions,
            'features': transformed,
            'prepared_data': prepared,
            'raw_data': self.dataset.composition()
        }
    
    def get_anomaly_report(self, data: pd.DataFrame = None, threshold_multiplier: float = 1.0):
        """Get anomalies with adjustable threshold."""
        if self.dataset is None:
            raise ValueError("No dataset set.")
        
        if not self._is_fitted:
            raise ValueError("Detector not fitted. Call fit_detector() first.")
        
        prepared = self.correction_service.prepare(self.dataset)
        transformed = self.correction_service.correct(self.dataset, self.transform_method)
        
        if hasattr(self.detector, 'get_anomalies'):
            labels = self.detector.get_anomalies(transformed, threshold_multiplier)
        else:
            labels = self.detector.predict(transformed)
        
        scores = self.detector.decision_function(transformed)
        
        return {
            'labels': labels,
            'scores': scores,
            'transformed': transformed,
            'anomaly_count': np.sum(labels == -1),
            'normal_count': np.sum(labels == 1)
        }


# =====================================================================
# PART 6: TRAINING COORDINATOR (Orchestration Pattern)
# =====================================================================
class TrainingCoordinator:
    """Orchestrates the training pipeline with experiment tracking."""
    
    def __init__(self, analyzer: GeochemAnalyzer = None, experiment_tracker=None):
        self.analyzer = analyzer or GeochemAnalyzer()
        self.tracker = experiment_tracker
        self.metrics_history = []
    
    def run_training(self, data: pd.DataFrame, oxide_columns: List[str],
                     test_size: float = 0.3, config: Dict = None):
        """Run the complete training pipeline."""
        self.analyzer.set_data(data, oxide_columns)
        
        # Split data (can be spatial, here we do random split)
        X = data[oxide_columns].copy()
        
        # We need labels for evaluation - use the detector's scores as pseudo-labels
        # For now, we just fit and evaluate on test set
        
        # Fit detector on all data (unsupervised)
        self.analyzer.fit_detector()
        
        # Get results
        result = self.analyzer.analyze()
        
        # Compute metrics based on anomaly detection
        n_anomalies = np.sum(result['labels'] == -1)
        n_normals = np.sum(result['labels'] == 1)
        anomaly_ratio = n_anomalies / (n_anomalies + n_normals) if (n_anomalies + n_normals) > 0 else 0
        
        metrics = {
            'n_samples': len(result['labels']),
            'n_anomalies': n_anomalies,
            'n_normals': n_normals,
            'anomaly_ratio': anomaly_ratio,
            'mean_score': np.mean(result['scores']),
            'min_score': np.min(result['scores']),
            'max_score': np.max(result['scores']),
            'std_score': np.std(result['scores'])
        }
        
        self.metrics_history.append(metrics)
        
        if self.tracker:
            self._log_experiment(config, metrics, self.analyzer.detector)
        
        return metrics, result
    
    def _log_experiment(self, config, metrics, detector):
        """Log experiment to tracker."""
        if self.tracker is None:
            return
        
        # This would integrate with MLflow
        pass
    
    def get_summary(self) -> Dict:
        """Get summary of all training runs."""
        if not self.metrics_history:
            return {'message': 'No training runs performed'}
        
        return {
            'runs': len(self.metrics_history),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else None
        }


# =====================================================================
# PART 7: ORCHESTRATION TRACER (Observer Pattern)
# =====================================================================
class OrchestrationTracer:
    """Tracer for observability - emits events at key steps."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events = []
    
    def emit(self, event_type: str, component: str, details: str, **kwargs):
        """Emit an event."""
        timestamp = datetime.now().isoformat()
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'component': component,
            'details': details,
            **kwargs
        }
        self.events.append(event)
        if self.verbose:
            print(f"[{component}] {event_type}: {details}")
        return event
    
    def get_events(self, event_type: str = None) -> List[Dict]:
        """Get all events, optionally filtered by type."""
        if event_type is None:
            return self.events
        return [e for e in self.events if e['type'] == event_type]
    
    def clear(self):
        """Clear all events."""
        self.events = []


# =====================================================================
# PART 8: REUSABLE PLOTTING FUNCTIONS
# =====================================================================
def plot_pca_biplot(ax, data: pd.DataFrame, n_components: int = 2,
                     show_loadings: bool = True, max_loading_labels: int = 15,
                     title: str = "PCA Biplot"):
    clean = data.dropna()
    if clean.shape[0] < 3 or clean.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough data for PCA", ha='center', va='center',
                transform=ax.transAxes)
        return None, None

    scaled = StandardScaler().fit_transform(clean)
    pca = SklearnPCA(n_components=min(n_components, scaled.shape[1], scaled.shape[0]))
    scores = pca.fit_transform(scaled)

    ax.scatter(scores[:, 0], scores[:, 1], c=scores[:, 0], cmap='viridis',
               alpha=0.75, edgecolors='white', s=70)

    if show_loadings:
        loadings = pca.components_.T
        scale = np.max(np.abs(scores)) * 0.8 if scores.size else 1.0
        for i, col in enumerate(clean.columns[:max_loading_labels]):
            ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                     head_width=scale * 0.04, head_length=scale * 0.04,
                     fc='red', ec='red', alpha=0.8, linewidth=1.3)
            ax.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
                    col, fontsize=8, color='darkred', weight='bold')

    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.3)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)" if len(ev) > 1 else "PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return pca, scores


def plot_explained_variance(ax, pca, n_components: int = None):
    if pca is None:
        ax.text(0.5, 0.5, "No PCA data available", ha='center', va='center',
                transform=ax.transAxes)
        return

    ev = pca.explained_variance_ratio_
    n = n_components if n_components else len(ev)
    bars = ax.bar(range(1, n + 1), ev[:n] * 100, color='#4F46E5', alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Explained Variance Ratio')
    ax.set_xticks(range(1, n + 1))
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, ev[:n] * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    return ev


def plot_correlation_heatmap(ax, data: pd.DataFrame, annotate_threshold: int = 10,
                             title: str = "Correlation Matrix"):
    corr = data.corr()
    n = len(corr.columns)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    if n <= annotate_threshold:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=7)
    ax.set_title(title)
    return corr, im


def plot_clustering_scatter(ax, X_pca, labels, n_clusters: int, method: str = "Clustering",
                            colors=None, markers=None):
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x']

    unique_labels = np.unique(labels)

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        if lab == -1:
            color, marker, lbl = 'gray', 'x', f'Noise ({np.sum(mask)})'
            size = 50
        else:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            lbl = f'Cluster {lab} ({np.sum(mask)})'
            size = 80
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], marker=marker,
                   alpha=0.7, edgecolors='white', s=size, label=lbl)

    ax.set_title(f"{method} Clustering Results ({n_clusters} clusters)")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_cluster_sizes(ax, labels, colors=None):
    unique_labels = np.unique(labels)
    cluster_names = []
    cluster_sizes = []

    for lab in unique_labels:
        if lab == -1:
            continue
        cluster_names.append(f'Cluster {lab}')
        cluster_sizes.append(np.sum(labels == lab))

    if not cluster_sizes:
        ax.text(0.5, 0.5, "No clusters found", ha='center', va='center',
                transform=ax.transAxes)
        return

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_sizes)))

    bars = ax.bar(cluster_names, cluster_sizes, color=colors[:len(cluster_sizes)], alpha=0.7)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Sample Count')
    ax.set_title('Cluster Sizes')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, cluster_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=8)
    return cluster_sizes


def plot_anomaly_scatter(ax, data, labels, scores, title="Anomaly Detection Results"):
    """Plot anomaly detection results with normal vs anomaly points."""
    # Use PCA for visualization
    pca = SklearnPCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(data))
    
    # Normal points (label = 1)
    normal_mask = labels == 1
    anomaly_mask = labels == -1
    
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
               c='blue', marker='o', alpha=0.6, s=50, label=f'Normal ({np.sum(normal_mask)})')
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
               c='red', marker='x', alpha=0.8, s=100, label=f'Anomaly ({np.sum(anomaly_mask)})')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_anomaly_scores(ax, scores, threshold=None, title="Anomaly Scores Distribution"):
    """Plot histogram of anomaly scores."""
    ax.hist(scores, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')
        ax.legend()
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def pca_summary_text(pca, columns, top_n: int = 5) -> str:
    if pca is None:
        return "PCA could not be computed."
    lines = [f"Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%"]
    if len(pca.explained_variance_ratio_) > 1:
        lines[0] += f", PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
        lines.append(f"Cumulative: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
    loadings = pca.components_.T
    for pc_idx in range(min(2, loadings.shape[1])):
        top_idx = np.argsort(np.abs(loadings[:, pc_idx]))[-top_n:][::-1]
        top = [f"{columns[i]} ({loadings[i, pc_idx]:.2f})" for i in top_idx]
        lines.append(f"PC{pc_idx+1} top contributors: {', '.join(top)}")
    return "\n".join(lines)


def get_strong_correlations(corr, threshold: float = 0.7) -> list:
    strong_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold:
                strong_pairs.append({
                    'pair': f"{corr.columns[i]}-{corr.columns[j]}",
                    'correlation': corr.iloc[i, j]
                })
    return strong_pairs


# =====================================================================
# PART 9: DATA CLEANING & STATISTICS ENGINES
# =====================================================================
def normalize_column_name(c):
    if c is None:
        return "Unnamed"
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    return "Unnamed" if c == "" or c.lower() in ["nan", "none", "null"] else c


def clean_dataframe_advanced(df, options):
    if df is None or df.empty:
        return df, "No data to clean"
    original_shape = df.shape
    report_lines = []
    df.columns = [str(c).strip().replace('\n', ' ').replace('\r', ' ') for c in df.columns]
    df.columns = [re.sub(r'\s+', ' ', c) for c in df.columns]
    report_lines.append(f"✓ Normalized {len(df.columns)} column names")
    if options.get('remove_spaces', True):
        for col in df.columns:
            try:
                df[col] = df[col].astype(str).str.strip()
                numeric_test = pd.to_numeric(df[col], errors='coerce')
                if numeric_test.notna().sum() / max(len(numeric_test), 1) > 0.9:
                    df[col] = numeric_test
            except:
                pass
        report_lines.append("✓ Removed leading/trailing spaces")
    if options.get('remove_special', True):
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype(str).str.replace(r'[^\w\s\.\-\+]', '', regex=True)
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            except:
                pass
        report_lines.append("✓ Removed special characters")
    if options.get('standardize_delim', True):
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].dropna().astype(str).head(20)
                    european_pattern = sample.str.match(r'^-?\d{1,3}(\.\d{3})*,\d+$')
                    if european_pattern.any():
                        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                        df[col] = df[col].str.replace(',', '.', regex=False)
                    else:
                        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        report_lines.append("✓ Standardized numeric delimiters")
    before = len(df)
    df = df.dropna(how='all')
    if before - len(df) > 0:
        report_lines.append(f"✓ Removed {before - len(df)} completely empty rows")
    if options.get('drop_null_rows', False):
        before = len(df)
        df = df.dropna()
        report_lines.append(f"✓ Removed {before - len(df)} rows with null values")
    if options.get('drop_null_cols', False):
        thresh = len(df) * 0.5
        cols_to_drop = df.columns[df.isnull().sum() > thresh]
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            report_lines.append(f"✓ Removed {len(cols_to_drop)} columns with >50% nulls")
    if options.get('fill_numeric', False):
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        report_lines.append("✓ Filled numeric nulls with median values")
    if options.get('fill_text', False):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
        report_lines.append("✓ Filled text nulls with 'Unknown'")
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        report_lines.append(f"✓ Removed {before - len(df)} duplicate rows")
    converted = []
    for col in df.columns:
        try:
            if not pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].astype(str).str.strip()
                series = series.str.replace(r'[^0-9\.\-eE\+\s]', '', regex=True)
                series = pd.to_numeric(series, errors='coerce')
                if series.notna().sum() / max(len(series), 1) > 0.8:
                    df[col] = series
                    converted.append(col)
        except:
            pass
    if converted:
        report_lines.append(f"✓ Converted {len(converted)} columns to numeric")
    final_shape = df.shape
    report_lines.insert(0, f"✅ Cleaning complete: {original_shape} → {final_shape}")
    return df, "\n".join(report_lines)


class EnhancedStatisticsEngine:
    @staticmethod
    def compute(df):
        if df is None or df.empty:
            return pd.DataFrame()
        nd = df.select_dtypes(include=[np.number])
        if nd.empty:
            return pd.DataFrame()
        rows = []
        for col in nd.columns:
            x = nd[col].dropna()
            if len(x) < 3:
                continue
            rows.append({
                "Variable": col,
                "N": len(x),
                "Mean": round(x.mean(), 4),
                "Median": round(x.median(), 4),
                "Std": round(x.std(), 4),
                "Min": round(x.min(), 4),
                "Max": round(x.max(), 4),
                "Skewness": round(float(scipystats.skew(x)), 4),
                "Kurtosis": round(float(scipystats.kurtosis(x)), 4)
            })
        return pd.DataFrame(rows)


class EnhancedGeochemistryEngine:
    @staticmethod
    def calculate_alteration_indices(df, indices):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        al2o3_col = find_col("Al2O3")
        cao_col = find_col("CaO")
        na2o_col = find_col("Na2O")
        k2o_col = find_col("K2O")
        for idx in indices:
            try:
                if idx == "CIA" and all(c is not None for c in [al2o3_col, cao_col, na2o_col, k2o_col]):
                    al2o3 = pd.to_numeric(df[al2o3_col], errors='coerce')
                    cao = pd.to_numeric(df[cao_col], errors='coerce')
                    na2o = pd.to_numeric(df[na2o_col], errors='coerce')
                    k2o = pd.to_numeric(df[k2o_col], errors='coerce')
                    al_m = al2o3 / 101.961
                    ca_m = cao / 56.077
                    na_m = na2o / 61.979
                    k_m = k2o / 94.196
                    ca_star = np.minimum(ca_m, na_m)
                    results["CIA"] = round(float((al_m / (al_m + ca_star + na_m + k_m + 1e-9)).mean() * 100), 2)
                elif idx == "CIA":
                    results["CIA"] = "Missing: Al2O3, CaO, Na2O, K2O"
            except Exception as e:
                results[f"{idx}_Error"] = str(e)
        return results

    @staticmethod
    def calculate_element_ratios(df, elements):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        std_ratios = [("K2O/Na2O", "K2O", "Na2O"), ("Rb/Sr", "Rb", "Sr"), ("Nb/Y", "Nb", "Y"),
                     ("Cu/Zn", "Cu", "Zn"), ("Au/Ag", "Au", "Ag"), ("Sr/Y", "Sr", "Y")]
        for name, num, den in std_ratios:
            try:
                nc = find_col(num)
                dc = find_col(den)
                if nc and dc:
                    r = pd.to_numeric(df[nc], errors='coerce') / (pd.to_numeric(df[dc], errors='coerce') + 1e-9)
                    results[name] = round(float(r.mean()), 3)
            except:
                pass
        return results

    @staticmethod
    def calculate_grade_statistics(df, elements, threshold=95):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        for el in elements:
            col = find_col(el)
            if col:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 0:
                    thresh = data.quantile(threshold / 100)
                    anomalies = data[data > thresh]
                    results[el] = {
                        "N": len(data),
                        "Mean": round(data.mean(), 2),
                        "Median": round(data.median(), 2),
                        "Std": round(data.std(), 2),
                        "Min": round(data.min(), 2),
                        "Max": round(data.max(), 2),
                        f"P{threshold}": round(thresh, 2),
                        "Anomalies": len(anomalies),
                        "Anomaly%": round(len(anomalies) / len(data) * 100, 1)
                    }
        return results


# =====================================================================
# PART 10: UI COMPONENTS & APPLICATION
# =====================================================================
APPNAME = "Elemental Society Analysis"
APPVERSION = " v1.0.0"
DEVELOPER = "Mohamed Almoufty Abdoalmoutleb Abaker"
EMAIL = "almoufty121@gmail.com"
GITHUB_URL = "https://github.com/almouftygeo/elemental-society-analysis"

class Palette:
    PRIMARY = "#4F46E5"
    PRIMARYDARK = "#4338CA"
    PRIMARYLIGHT = "#EEF2FF"
    SUCCESS = "#059669"
    WARNING = "#D97706"
    DANGER = "#DC2626"
    INFO = "#0284C7"
    SIDEBARBG = "#111827"
    SIDEBARBGHOVER = "#1F2937"
    SIDEBARTEXT = "#9CA3AF"
    SIDEBARTEXTACTIVE = "#FFFFFF"
    BG = "#F3F4F6"
    SURFACE = "#FFFFFF"
    BORDER = "#E5E7EB"
    TEXTPRIMARY = "#111827"
    TEXTSECONDARY = "#6B7280"
    TEXTMUTED = "#9CA3AF"


# =====================================================================
# UI ICONS
# =====================================================================
class Icons:
    cache = {}
    @classmethod
    def get(cls, name, color=Palette.TEXTPRIMARY, size=20):
        key = (name, color, size)
        if key in cls.cache:
            return cls.cache[key]
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(max(1.2, size / 11.0))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        m = size * 0.15
        s = size - 2 * m
        draw = getattr(cls, f"draw{name}", lambda *a: None)
        draw(p, m, s, size, color)
        p.end()
        icon = QIcon(pix)
        cls.cache[key] = icon
        return icon

    @staticmethod
    def drawhome(p, m, s, sz, c):
        path = QPainterPath()
        path.moveTo(m, m + s * 0.45)
        path.lineTo(sz / 2, m)
        path.lineTo(m + s, m + s * 0.45)
        p.drawPath(path)
        p.drawRect(QRectF(m + s * 0.18, m + s * 0.45, s * 0.64, s * 0.55))

    @staticmethod
    def drawupload(p, m, s, sz, c):
        cx = sz / 2
        p.drawLine(QPointF(cx, m), QPointF(cx, m + s * 0.6))
        path = QPainterPath()
        path.moveTo(QPointF(cx - s * 0.25, m + s * 0.22))
        path.lineTo(QPointF(cx, m))
        path.lineTo(QPointF(cx + s * 0.25, m + s * 0.22))
        p.drawPath(path)
        p.drawLine(QPointF(m, m + s * 0.85), QPointF(m + s, m + s * 0.85))

    @staticmethod
    def drawrefresh(p, m, s, sz, c):
        rect = QRectF(m, m, s, s)
        p.drawArc(rect, 30 * 16, 300 * 16)
        p.drawLine(QPointF(m + s, m + s * 0.1), QPointF(m + s * 0.75, m))
        p.drawLine(QPointF(m + s, m + s * 0.1), QPointF(m + s * 0.85, m + s * 0.35))

    @staticmethod
    def drawchart(p, m, s, sz, c):
        base = m + s
        p.drawRect(QRectF(m, base - s * 0.35, s * 0.2, s * 0.35))
        p.drawRect(QRectF(m + s * 0.4, base - s * 0.65, s * 0.2, s * 0.65))
        p.drawRect(QRectF(m + s * 0.8 - s * 0.2, base - s * 0.95, s * 0.2, s * 0.95))

    @staticmethod
    def drawdownload(p, m, s, sz, c):
        cx = sz / 2
        p.drawLine(QPointF(cx, m), QPointF(cx, m + s * 0.6))
        path = QPainterPath()
        path.moveTo(QPointF(cx - s * 0.25, m + s * 0.38))
        path.lineTo(QPointF(cx, m + s * 0.6))
        path.lineTo(QPointF(cx + s * 0.25, m + s * 0.38))
        p.drawPath(path)
        p.drawLine(QPointF(m, m + s * 0.85), QPointF(m + s, m + s * 0.85))

    @staticmethod
    def drawbroom(p, m, s, sz, c):
        p.drawLine(QPointF(m + s * 0.2, m), QPointF(m + s * 0.75, m + s * 0.65))
        tri = QPolygonF([QPointF(m + s * 0.6, m + s * 0.55), QPointF(m + s, m + s * 0.6), QPointF(m + s * 0.7, m + s)])
        p.drawPolygon(tri)

    @staticmethod
    def drawlist(p, m, s, sz, c):
        for i in range(3):
            y = m + s * 0.15 + i * 0.35
            p.drawEllipse(QPointF(m + s * 0.06, y), 1.2, 1.2)
            p.drawLine(QPointF(m + s * 0.22, y), QPointF(m + s, y))

    @staticmethod
    def drawtrend(p, m, s, sz, c):
        path = QPainterPath()
        path.moveTo(m, m + s * 0.8)
        path.lineTo(m + s * 0.3, m + s * 0.4)
        path.lineTo(m + s * 0.55, m + s * 0.6)
        path.lineTo(m + s, m)
        p.drawPath(path)

    @staticmethod
    def drawtrash(p, m, s, sz, c):
        p.drawLine(QPointF(m, m + s * 0.2), QPointF(m + s, m + s * 0.2))
        p.drawRect(QRectF(m + s * 0.15, m + s * 0.2, s * 0.7, s * 0.75))
        p.drawLine(QPointF(m + s * 0.35, m + s * 0.2), QPointF(m + s * 0.4, m))
        p.drawLine(QPointF(m + s * 0.65, m + s * 0.2), QPointF(m + s * 0.6, m))

    @staticmethod
    def drawhelp(p, m, s, sz, c):
        p.drawEllipse(QRectF(m, m, s, s))
        p.drawText(QRectF(m, m, s, s), Qt.AlignCenter, "?")

    @staticmethod
    def drawdocument(p, m, s, sz, c):
        p.drawRect(QRectF(m, m, s * 0.75, s))
        for i in range(3):
            y = m + s * 0.3 + i * 0.22
            p.drawLine(QPointF(m + s * 0.15, y), QPointF(m + s * 0.6, y))

    @staticmethod
    def drawcorrection(p, m, s, sz, c):
        p.drawArc(QRectF(m, m, s, s), 30 * 16, 300 * 16)
        p.drawLine(QPointF(m + s * 0.8, m + s * 0.1), QPointF(m + s * 0.6, m))
        p.drawLine(QPointF(m + s * 0.8, m + s * 0.1), QPointF(m + s * 0.9, m + s * 0.3))

    @staticmethod
    def drawanomaly(p, m, s, sz, c):
        # Draw anomaly detection icon (magnifying glass with exclamation)
        p.drawEllipse(QRectF(m + s * 0.1, m, s * 0.5, s * 0.5))
        p.drawLine(QPointF(m + s * 0.55, m + s * 0.5), QPointF(m + s * 0.85, m + s * 0.8))
        p.drawText(QRectF(m + s * 0.3, m + s * 0.15, s * 0.1, s * 0.1), Qt.AlignCenter, "!")


def styledbutton(text, iconname=None, kind="default", tooltip=None):
    btn = QPushButton(text)
    if iconname:
        col = "#FFFFFF" if kind in ("primary", "success", "danger", "warning", "info") else Palette.TEXTSECONDARY
        btn.setIcon(Icons.get(iconname, color=col, size=16))
        btn.setIconSize(QSize(16, 16))
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setProperty("kind", kind)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFont(QFont("Segoe UI", 10))
    return btn


# =====================================================================
# FULL APPLICATION - Complete with all tabs including Anomaly Detection
# =====================================================================
class ElementalSocietyAnalysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.raw_df = None
        self.df = None
        self.filename = None
        self.index_column = None
        self.active_columns = []
        self.display_rows = 100
        self.display_cols = 20
        self.show_all = False
        
        # Anomaly detection components
        self.analyzer = None
        self.detector = None
        self.anomaly_results = None
        self.detector_config = AnomalyDetectorConfig()
        
        # Existing components
        self.coordinator = None
        self.workflow_report = None
        self.zero_strategy = None

        self.last_plot_config = ["Histogram", "PDF KDE"]
        self.last_geochem_data = {"alterationindices": {}, "elementratios": {}, "gradestats": {}}
        self.last_cluster_data = None
        self.svm_params = None
        self.last_svm_data = None
        self.last_analysis_report = ""

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"{APPNAME}{APPVERSION} - Anomaly Detection")
        self.setGeometry(50, 50, 1600, 950)
        self.setStyleSheet(self.get_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.create_home_page()
        self.create_data_page()
        self.create_dist_page()
        self.create_analysis_page()
        self.create_geochem_page()
        self.create_clustering_page()
        self.create_svm_page()
        self.create_anomaly_page()
        self.create_workflow_page()
        self.create_log_page()

        self.sidebar = self.build_sidebar()
        main_layout.addWidget(self.sidebar)

        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = self.build_toolbar()
        r_layout.addWidget(self.toolbar)
        self.status_panel = self.build_status_panel()
        r_layout.addWidget(self.status_panel)
        r_layout.addWidget(self.stack)
        main_layout.addWidget(right)

        self.stack.setCurrentIndex(0)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'log_text'):
            self.log_text.append(f"[{ts}] {msg}")

    def set_status(self, msg, error=False):
        if hasattr(self, 'status_lbl'):
            self.status_lbl.setText(msg)
            self.status_lbl.setStyleSheet(f"color:{Palette.DANGER if error else Palette.PRIMARY};font-weight:bold;")

    # =====================================================================
    # UI BUILDING METHODS
    # =====================================================================
    def build_sidebar(self):
        w = QWidget()
        w.setFixedWidth(230)
        w.setObjectName("Sidebar")
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)

        brand = QLabel("Elemental\nSociety")
        brand.setObjectName("BrandLabel")
        brand.setAlignment(Qt.AlignCenter)
        l.addWidget(brand)

        self.nav = QListWidget()
        self.nav.addItems(["Home", "Dataset", "Distributions", "Analysis",
                          "Geochemistry", "Clustering", "SVM", "Anomaly Detection",
                          "Workflow", "Logs"])
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        l.addWidget(self.nav)

        l.addStretch()
        help_btn = QPushButton("  Help / About")
        help_btn.setIcon(Icons.get("help", color=Palette.SIDEBARTEXT, size=18))
        help_btn.setFlat(True)
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setStyleSheet(f"color:{Palette.SIDEBARTEXT};padding:10px;border:none;")
        help_btn.clicked.connect(self.show_help)
        l.addWidget(help_btn)

        ver = QLabel(f"{APPVERSION}\n{DEVELOPER}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(f"color:{Palette.SIDEBARTEXT};padding:10px;font-size:10px;")
        l.addWidget(ver)
        return w

    def build_toolbar(self):
        w = QWidget()
        w.setObjectName("Toolbar")
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)

        self.btn_upload = styledbutton("Upload", "upload", "primary")
        self.btn_upload.clicked.connect(self.upload_file)
        self.btn_clean = styledbutton("Clean", "broom", "warning")
        self.btn_clean.clicked.connect(self.advanced_clean_data)
        self.btn_correction = styledbutton("Correction", "correction", "info", "Apply closure correction")
        self.btn_correction.clicked.connect(self.show_correction_dialog)
        self.btn_filter = styledbutton("Filter", "list", "default")
        self.btn_filter.clicked.connect(self.open_filter_dialog)
        self.btn_reset = styledbutton("Reset", "refresh", "default")
        self.btn_reset.clicked.connect(self.reset_data)

        l.addWidget(self.btn_upload)
        l.addWidget(self.btn_clean)
        l.addWidget(self.btn_correction)
        l.addWidget(self.btn_filter)
        l.addWidget(self.btn_reset)

        self.btn_global_refresh = styledbutton("Refresh All", "refresh", "info")
        self.btn_global_refresh.clicked.connect(self.global_refresh)
        l.addWidget(self.btn_global_refresh)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        l.addWidget(sep)

        self.btn_export_csv = styledbutton("Export Data", "download", "default")
        self.btn_export_csv.clicked.connect(self.export_data_csv)
        self.btn_export_fig = styledbutton("Export Fig", "trend", "success")
        self.btn_export_fig.clicked.connect(self.export_current_figure)
        self.btn_report = styledbutton("Report", "document", "info")
        self.btn_report.clicked.connect(self.generate_full_report)

        l.addWidget(self.btn_export_csv)
        l.addWidget(self.btn_export_fig)
        l.addWidget(self.btn_report)

        l.addStretch()
        return w

    def build_status_panel(self):
        w = QWidget()
        w.setObjectName("StatusPanel")
        l = QHBoxLayout(w)
        l.setContentsMargins(10, 5, 10, 5)
        self.status_lbl = QLabel("Ready - Upload a data file to begin")
        l.addWidget(self.status_lbl)
        l.addStretch()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        l.addWidget(self.progress)
        return w

    # =====================================================================
    # CREATE PAGE METHODS
    # =====================================================================
    def create_home_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        l.setAlignment(Qt.AlignCenter)

        card = QWidget()
        card.setStyleSheet("background:white;border-radius:12px;padding:40px;margin:20px;")
        cl = QVBoxLayout(card)

        title = QLabel(f"Welcome to {APPNAME}")
        title.setStyleSheet(f"font-size:32px;font-weight:bold;color:{Palette.PRIMARY};")
        title.setAlignment(Qt.AlignCenter)
        cl.addWidget(title)

        version_label = QLabel(f"Version {APPVERSION}")
        version_label.setStyleSheet(f"font-size:14px;color:{Palette.TEXTSECONDARY};")
        version_label.setAlignment(Qt.AlignCenter)
        cl.addWidget(version_label)

        sub = QLabel("Comprehensive Geochemical Data Analysis Platform with Isolation Forest Anomaly Detection")
        sub.setStyleSheet(f"font-size:16px;color:{Palette.TEXTSECONDARY};margin-bottom:15px;")
        sub.setAlignment(Qt.AlignCenter)
        cl.addWidget(sub)

        features = QLabel(
            "✨ Key Features:\n"
            "• Coordinated Workflow Management\n"
            "• Isolation Forest Anomaly Detection\n"
            "• Zero Replacement Strategies (Constant, Half-DL, Multiplicative)\n"
            "• CLR Enrichment Interpretation (exp(CLR))\n"
            "• CLR/ALR/ILR Closure Correction\n"
            "• PCA Analysis with Loading Vectors\n"
            "• K-Means & DBSCAN Clustering\n"
            "• SVM Classification\n"
            "• Geochemistry Indices & Element Ratios\n"
            "• Comprehensive HTML Reports"
        )
        features.setStyleSheet(f"font-size:13px;color:{Palette.TEXTPRIMARY};line-height:1.6;margin:10px;")
        cl.addWidget(features)

        dev = QLabel(f"Developed by: {DEVELOPER}")
        dev.setStyleSheet(f"font-size:12px;color:{Palette.TEXTMUTED};margin-top:10px;")
        dev.setAlignment(Qt.AlignCenter)
        cl.addWidget(dev)

        email_label = QLabel(f"📧 {EMAIL}")
        email_label.setStyleSheet(f"font-size:12px;color:{Palette.TEXTMUTED};")
        email_label.setAlignment(Qt.AlignCenter)
        cl.addWidget(email_label)

        github_label = QLabel(f'🔗 <a href="{GITHUB_URL}" style="color:#4F46E5;">GitHub Repository</a>')
        github_label.setStyleSheet(f"font-size:12px;color:{Palette.INFO};")
        github_label.setAlignment(Qt.AlignCenter)
        github_label.setOpenExternalLinks(True)
        github_label.setTextFormat(Qt.RichText)
        cl.addWidget(github_label)

        bl = QHBoxLayout()
        bl.setAlignment(Qt.AlignCenter)
        up = styledbutton("📂 Upload Data File", "upload", "primary")
        up.setMinimumHeight(50)
        up.setMinimumWidth(200)
        up.clicked.connect(self.upload_file)
        bl.addWidget(up)
        cl.addLayout(bl)

        tips = QLabel(
            "💡 Quick Start:\n"
            "1. Upload your CSV or Excel file\n"
            "2. Choose a zero replacement strategy if zeros exist\n"
            "3. Apply closure correction from the toolbar\n"
            "4. Go to Anomaly Detection to run Isolation Forest\n"
            "5. View CLR Enrichment Interpretation in Geochemistry tab"
        )
        tips.setStyleSheet(f"font-size:11px;color:{Palette.TEXTMUTED};margin-top:15px;padding:10px;background:{Palette.PRIMARYLIGHT};border-radius:6px;")
        tips.setAlignment(Qt.AlignCenter)
        cl.addWidget(tips)

        l.addWidget(card)
        self.stack.addWidget(p)

    def create_data_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        t1 = QHBoxLayout()
        t1.addWidget(QLabel("View:"))
        self.combo_data_view = QComboBox()
        self.combo_data_view.addItems(["Working Data", "Raw Data", "Analysis Data", "Corrected Data"])
        self.combo_data_view.currentTextChanged.connect(self.switch_data_view)
        t1.addWidget(self.combo_data_view)
        t1.addStretch()

        self.btn_refresh_view = styledbutton("Refresh", "refresh", "default")
        self.btn_refresh_view.clicked.connect(lambda: self.update_table())
        self.btn_display = styledbutton("Display", "chart", "default")
        self.btn_display.clicked.connect(self.control_display)
        self.btn_index = styledbutton("Index/Active", "list", "default")
        self.btn_index.clicked.connect(self.set_index_columns)
        self.btn_rename = styledbutton("Rename", "list", "default")
        self.btn_rename.clicked.connect(self.rename_columns_dialog)
        self.btn_units = styledbutton("Units", "refresh", "default")
        self.btn_units.clicked.connect(self.unify_units)
        self.btn_delete = styledbutton("Delete", "trash", "danger")
        self.btn_delete.clicked.connect(self.delete_data)

        for btn in [self.btn_refresh_view, self.btn_display, self.btn_index,
                   self.btn_rename, self.btn_units, self.btn_delete]:
            t1.addWidget(btn)
        l.addLayout(t1)

        t2 = QHBoxLayout()
        self.btn_element_ratio = styledbutton("Calculate Ratio", "flask", "info")
        self.btn_element_ratio.clicked.connect(self.calculate_element_ratio)
        t2.addWidget(self.btn_element_ratio)
        t2.addStretch()
        l.addLayout(t2)

        self.data_info = QLabel("No data loaded")
        l.addWidget(self.data_info)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        l.addWidget(self.table)
        self.stack.addWidget(p)

    def create_dist_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        c = QHBoxLayout()
        c.addWidget(QLabel("Variable:"))
        self.dist_var = QComboBox()
        c.addWidget(self.dist_var)
        self.btn_plot_config = styledbutton("Config", "chart", "primary")
        self.btn_plot_config.clicked.connect(self.show_plot_config_safe)
        c.addWidget(self.btn_plot_config)
        self.btn_update_dist = styledbutton("Update", "refresh", "success")
        self.btn_update_dist.clicked.connect(self.update_distribution)
        c.addWidget(self.btn_update_dist)
        self.btn_export_dist = styledbutton("Export", "download", "info")
        self.btn_export_dist.clicked.connect(self.export_distribution_plot)
        c.addWidget(self.btn_export_dist)
        c.addStretch()
        l.addLayout(c)

        self.plot_title_label = QLabel("Distribution Plot")
        self.plot_title_label.setStyleSheet(f"font-size:14px;font-weight:bold;color:{Palette.PRIMARY};padding:5px;")
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        l.addWidget(self.plot_title_label)

        self.dist_fig = plt.Figure(figsize=(10, 8))
        self.dist_canvas = FigureCanvas(self.dist_fig)
        l.addWidget(self.dist_canvas)
        self.stack.addWidget(p)

    def create_analysis_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        tabs = QTabWidget()

        # Statistics tab
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        sc = QHBoxLayout()
        self.btn_refresh_stats = styledbutton("Refresh", "refresh", "primary")
        self.btn_refresh_stats.clicked.connect(self.update_statistics)
        self.btn_export_stats = styledbutton("Export", "download", "default")
        self.btn_export_stats.clicked.connect(self.export_statistics)
        sc.addWidget(self.btn_refresh_stats)
        sc.addWidget(self.btn_export_stats)
        sc.addStretch()
        l1.addLayout(sc)
        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        l1.addWidget(self.stats_table)
        tabs.addTab(t1, "Statistics")

        # Plots tab
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        c2 = QHBoxLayout()
        c2.addWidget(QLabel("Type:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Correlation Matrix", "PCA Analysis", "Scatter Plot"])
        self.analysis_type.currentTextChanged.connect(self.on_analysis_type_changed)
        c2.addWidget(self.analysis_type)
        self.cb_show_pca = QCheckBox("Show PCA Vectors")
        self.cb_show_pca.setChecked(True)
        c2.addWidget(self.cb_show_pca)
        c2.addWidget(QLabel("Vars:"))
        self.analysis_vars = QListWidget()
        self.analysis_vars.setMaximumHeight(80)
        self.analysis_vars.setSelectionMode(QAbstractItemView.MultiSelection)
        c2.addWidget(self.analysis_vars)
        self.btn_update_analysis = styledbutton("Update", "refresh", "success")
        self.btn_update_analysis.clicked.connect(self.update_analysis)
        c2.addWidget(self.btn_update_analysis)
        self.btn_export_analysis = styledbutton("Export", "download", "info")
        self.btn_export_analysis.clicked.connect(self.export_analysis_plot)
        c2.addWidget(self.btn_export_analysis)
        c2.addStretch()
        l2.addLayout(c2)

        self.ana_fig = plt.Figure(figsize=(12, 8))
        self.ana_canvas = FigureCanvas(self.ana_fig)
        l2.addWidget(self.ana_canvas)

        self.ana_report = QTextEdit()
        self.ana_report.setReadOnly(True)
        self.ana_report.setMaximumHeight(150)
        l2.addWidget(self.ana_report)
        tabs.addTab(t2, "Plots")

        l.addWidget(tabs)
        self.stack.addWidget(p)

    def on_analysis_type_changed(self, text):
        self.cb_show_pca.setVisible(text == "PCA Analysis")

    def create_geochem_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        tabs = QTabWidget()

        # Alteration Indices
        t1 = QWidget()
        t1l = QVBoxLayout()
        h1 = QHBoxLayout()
        self.alteration_cbs = {}
        for idx in ["CIA", "CIW", "PIA", "ICV", "WIP"]:
            cb = QCheckBox(idx)
            cb.setChecked(idx == "CIA")
            self.alteration_cbs[idx] = cb
            h1.addWidget(cb)
        h1.addStretch()
        btn_alt = styledbutton("Run", "refresh", "primary")
        btn_alt.clicked.connect(self.calculate_alteration_only)
        h1.addWidget(btn_alt)
        t1l.addLayout(h1)
        self.alteration_view = QTextEdit()
        self.alteration_view.setReadOnly(True)
        t1l.addWidget(self.alteration_view)
        t1.setLayout(t1l)
        tabs.addTab(t1, "Alteration Indices")

        # Element Ratios
        t2 = QWidget()
        t2l = QVBoxLayout()
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Elements:"))
        self.ratio_selector = QListWidget()
        self.ratio_selector.setMaximumHeight(80)
        self.ratio_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        h2.addWidget(self.ratio_selector)
        btn_rat = styledbutton("Run", "refresh", "primary")
        btn_rat.clicked.connect(self.calculate_ratios_only)
        h2.addWidget(btn_rat)
        t2l.addLayout(h2)
        self.ratio_view = QTextEdit()
        self.ratio_view.setReadOnly(True)
        t2l.addWidget(self.ratio_view)
        t2.setLayout(t2l)
        tabs.addTab(t2, "Element Ratios")

        # Grade Statistics
        t3 = QWidget()
        t3l = QVBoxLayout()
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Threshold %:"))
        self.grade_threshold = QDoubleSpinBox()
        self.grade_threshold.setRange(50, 99)
        self.grade_threshold.setValue(95)
        h3.addWidget(self.grade_threshold)
        h3.addWidget(QLabel("Elements:"))
        self.grade_elements_list = QListWidget()
        self.grade_elements_list.setMaximumHeight(80)
        self.grade_elements_list.addItems(["Au", "Ag", "Cu", "Pb", "Zn", "Ni", "Co", "As"])
        self.grade_elements_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(5):
            self.grade_elements_list.item(i).setSelected(True)
        h3.addWidget(self.grade_elements_list)
        btn_grd = styledbutton("Run", "refresh", "primary")
        btn_grd.clicked.connect(self.calculate_grades_only)
        h3.addWidget(btn_grd)
        t3l.addLayout(h3)
        self.grade_view = QTextEdit()
        self.grade_view.setReadOnly(True)
        t3l.addWidget(self.grade_view)
        t3.setLayout(t3l)
        tabs.addTab(t3, "Grade Statistics")

        # Closure Correction Tab
        closure_tab = QWidget()
        cl = QVBoxLayout(closure_tab)
        cl.addWidget(QLabel("Closure Correction & CLR Enrichment Interpretation"))

        # Zero Strategy
        hz = QHBoxLayout()
        btn_zero = styledbutton("Choose Zero Strategy", "list", "info")
        btn_zero.clicked.connect(self.choose_zero_strategy)
        hz.addWidget(btn_zero)
        self.zero_strategy_label = QLabel("Strategy: None")
        hz.addWidget(self.zero_strategy_label)
        hz.addStretch()
        cl.addLayout(hz)

        # Correction Method
        hc = QHBoxLayout()
        hc.addWidget(QLabel("Method:"))
        self.corrector_selector = QComboBox()
        self.corrector_selector.addItems(["CLR", "ALR", "ILR"])
        hc.addWidget(self.corrector_selector)
        apply_btn = styledbutton("Apply Correction", "refresh", "primary")
        apply_btn.clicked.connect(lambda: self.apply_correction(
            self.corrector_selector.currentText()))
        hc.addWidget(apply_btn)
        cl.addLayout(hc)

        self.closure_status_label = QLabel("")
        cl.addWidget(self.closure_status_label)

        # CLR Interpretation Button
        h_int = QHBoxLayout()
        btn_interpret = styledbutton("Show CLR Enrichment Interpretation", "chart", "success")
        btn_interpret.clicked.connect(self.show_clr_interpretation)
        h_int.addWidget(btn_interpret)
        h_int.addStretch()
        cl.addLayout(h_int)

        self.cb_use_clr = QCheckBox("Auto-use CLR in Analysis/Clustering")
        self.cb_use_clr.toggled.connect(self.toggle_use_clr)
        cl.addWidget(self.cb_use_clr)

        closure_tab.setLayout(cl)
        tabs.addTab(closure_tab, "Closure")

        l.addWidget(tabs)
        exp = QHBoxLayout()
        btn_exp_all = styledbutton("Export All to Excel", "download", "success")
        btn_exp_all.clicked.connect(self.export_geochem_excel)
        exp.addWidget(btn_exp_all)
        exp.addStretch()
        l.addLayout(exp)
        self.stack.addWidget(p)

    def create_clustering_page(self):
        p = QWidget()
        l = QVBoxLayout(p)

        c = QHBoxLayout()
        c.addWidget(QLabel("Method:"))
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(["K-Means", "DBSCAN"])
        self.cluster_method.currentTextChanged.connect(self.toggle_cluster_params)
        c.addWidget(self.cluster_method)

        self.kmeans_group = QGroupBox("K-Means")
        kg = QHBoxLayout()
        kg.addWidget(QLabel("k:"))
        self.kmeans_k = QSpinBox()
        self.kmeans_k.setRange(2, 20)
        self.kmeans_k.setValue(3)
        kg.addWidget(self.kmeans_k)
        self.kmeans_group.setLayout(kg)
        c.addWidget(self.kmeans_group)

        self.dbscan_group = QGroupBox("DBSCAN")
        dg = QHBoxLayout()
        dg.addWidget(QLabel("eps:"))
        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setRange(0.01, 10)
        self.dbscan_eps.setValue(0.5)
        dg.addWidget(self.dbscan_eps)
        dg.addWidget(QLabel("MinPts:"))
        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setRange(2, 100)
        self.dbscan_min_samples.setValue(5)
        dg.addWidget(self.dbscan_min_samples)
        self.dbscan_group.setLayout(dg)
        self.dbscan_group.setVisible(False)
        c.addWidget(self.dbscan_group)
        c.addStretch()
        l.addLayout(c)

        vl = QHBoxLayout()
        vl.addWidget(QLabel("Variables:"))
        self.cluster_vars = QListWidget()
        self.cluster_vars.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cluster_vars.setMaximumHeight(80)
        vl.addWidget(self.cluster_vars)
        l.addLayout(vl)

        bl = QHBoxLayout()
        self.btn_run_cluster = styledbutton("Run Clustering", "refresh", "primary")
        self.btn_run_cluster.clicked.connect(self.run_clustering)
        self.btn_export_cluster_fig = styledbutton("Export Figure", "download", "success")
        self.btn_export_cluster_fig.clicked.connect(self.export_cluster_figure)
        self.btn_export_assign = styledbutton("Export Assignments", "download", "default")
        self.btn_export_assign.clicked.connect(self.export_cluster_assignments)
        bl.addWidget(self.btn_run_cluster)
        bl.addWidget(self.btn_export_cluster_fig)
        bl.addWidget(self.btn_export_assign)
        bl.addStretch()
        l.addLayout(bl)

        split = QSplitter(Qt.Horizontal)
        split.setSizes([600, 300])

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.cluster_fig = plt.Figure(figsize=(8, 6))
        self.cluster_canvas = FigureCanvas(self.cluster_fig)
        left_layout.addWidget(self.cluster_canvas)
        split.addWidget(left_widget)

        right_widget = QWidget()
        rl = QVBoxLayout(right_widget)
        self.cluster_stats = QTableWidget()
        self.cluster_stats.setAlternatingRowColors(True)
        rl.addWidget(QLabel("Cluster Statistics:"))
        rl.addWidget(self.cluster_stats)
        self.cluster_features_label = QLabel("🔍 Distinguishing features will appear here after running clustering")
        self.cluster_features_label.setWordWrap(True)
        self.cluster_features_label.setStyleSheet(f"padding:10px; background:white; border:1px solid {Palette.BORDER}; border-radius:4px;")
        rl.addWidget(QLabel("Cluster Characteristics:"))
        rl.addWidget(self.cluster_features_label)
        split.addWidget(right_widget)

        l.addWidget(split)
        self.stack.addWidget(p)

    def create_svm_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        c = QHBoxLayout()
        self.btn_svm_config = styledbutton("Configure", "list", "primary")
        self.btn_svm_config.clicked.connect(self.configure_svm)
        self.btn_run_svm = styledbutton("Run", "refresh", "success")
        self.btn_run_svm.clicked.connect(self.run_svm)
        self.btn_overfit = styledbutton("Overfitting", "trend", "warning")
        self.btn_overfit.clicked.connect(self.visualize_overfitting)
        self.btn_export_svm = styledbutton("Export Fig", "download", "info")
        self.btn_export_svm.clicked.connect(self.export_svm_figure)
        c.addWidget(self.btn_svm_config)
        c.addWidget(self.btn_run_svm)
        c.addWidget(self.btn_overfit)
        c.addWidget(self.btn_export_svm)
        c.addStretch()
        l.addLayout(c)

        split = QSplitter(Qt.Vertical)
        split.setSizes([400, 200])

        self.svm_fig = plt.Figure(figsize=(12, 5))
        self.svm_canvas = FigureCanvas(self.svm_fig)
        split.addWidget(self.svm_canvas)

        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.addWidget(QLabel("Results:"))
        self.svm_accuracy_table = QTableWidget()
        self.svm_accuracy_table.setMaximumHeight(80)
        rl.addWidget(self.svm_accuracy_table)
        self.svm_report = QTextEdit()
        self.svm_report.setReadOnly(True)
        rl.addWidget(self.svm_report)
        split.addWidget(rw)

        l.addWidget(split)
        self.stack.addWidget(p)

    # =====================================================================
    # ANOMALY DETECTION PAGE
    # =====================================================================
    def create_anomaly_page(self):
        p = QWidget()
        l = QVBoxLayout(p)

        # Top controls
        controls = QHBoxLayout()

        # Method selection
        controls.addWidget(QLabel("Method:"))
        self.anomaly_method = QComboBox()
        self.anomaly_method.addItems(["Isolation Forest", "Local Outlier Factor"])
        self.anomaly_method.currentTextChanged.connect(self.on_anomaly_method_changed)
        controls.addWidget(self.anomaly_method)

        # Parameters
        controls.addWidget(QLabel("Contamination:"))
        self.anomaly_contamination = QDoubleSpinBox()
        self.anomaly_contamination.setRange(0.01, 0.5)
        self.anomaly_contamination.setSingleStep(0.01)
        self.anomaly_contamination.setValue(0.08)
        controls.addWidget(self.anomaly_contamination)

        controls.addWidget(QLabel("n_estimators:"))
        self.anomaly_estimators = QSpinBox()
        self.anomaly_estimators.setRange(10, 500)
        self.anomaly_estimators.setSingleStep(10)
        self.anomaly_estimators.setValue(100)
        controls.addWidget(self.anomaly_estimators)

        controls.addStretch()
        l.addLayout(controls)

        # Advanced controls
        adv_controls = QHBoxLayout()
        self.cb_use_eif = QCheckBox("Use Extended Isolation Forest (EIF)")
        self.cb_use_eif.setToolTip("Use EIF algorithm if available")
        adv_controls.addWidget(self.cb_use_eif)

        self.cb_use_clr_anomaly = QCheckBox("Apply CLR before detection")
        self.cb_use_clr_anomaly.setChecked(True)
        adv_controls.addWidget(self.cb_use_clr_anomaly)

        adv_controls.addWidget(QLabel("Threshold multiplier:"))
        self.anomaly_threshold_mult = QDoubleSpinBox()
        self.anomaly_threshold_mult.setRange(0.1, 3.0)
        self.anomaly_threshold_mult.setSingleStep(0.1)
        self.anomaly_threshold_mult.setValue(1.0)
        adv_controls.addWidget(self.anomaly_threshold_mult)

        adv_controls.addStretch()
        l.addLayout(adv_controls)

        # Feature selection
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Features:"))
        self.anomaly_features = QListWidget()
        self.anomaly_features.setMaximumHeight(80)
        self.anomaly_features.setSelectionMode(QAbstractItemView.MultiSelection)
        fl.addWidget(self.anomaly_features)
        l.addLayout(fl)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_run_anomaly = styledbutton("Run Anomaly Detection", "anomaly", "primary")
        self.btn_run_anomaly.clicked.connect(self.run_anomaly_detection)
        btn_layout.addWidget(self.btn_run_anomaly)

        self.btn_export_anomaly = styledbutton("Export Results", "download", "success")
        self.btn_export_anomaly.clicked.connect(self.export_anomaly_results)
        btn_layout.addWidget(self.btn_export_anomaly)

        self.btn_export_anomaly_fig = styledbutton("Export Figure", "trend", "info")
        self.btn_export_anomaly_fig.clicked.connect(self.export_anomaly_figure)
        btn_layout.addWidget(self.btn_export_anomaly_fig)

        btn_layout.addStretch()
        l.addLayout(btn_layout)

        # Results display
        split = QSplitter(Qt.Horizontal)
        split.setSizes([700, 300])

        # Left: plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.anomaly_fig = plt.Figure(figsize=(10, 6))
        self.anomaly_canvas = FigureCanvas(self.anomaly_fig)
        plot_layout.addWidget(self.anomaly_canvas)

        split.addWidget(plot_widget)

        # Right: results table and summary
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(5, 0, 5, 0)

        self.anomaly_summary = QTextEdit()
        self.anomaly_summary.setReadOnly(True)
        self.anomaly_summary.setMaximumHeight(150)
        result_layout.addWidget(QLabel("📊 Summary:"))
        result_layout.addWidget(self.anomaly_summary)

        self.anomaly_table = QTableWidget()
        self.anomaly_table.setAlternatingRowColors(True)
        result_layout.addWidget(QLabel("📋 Results Table:"))
        result_layout.addWidget(self.anomaly_table)

        split.addWidget(result_widget)
        l.addWidget(split)

        self.stack.addWidget(p)

    def on_anomaly_method_changed(self, method):
        """Update UI based on selected anomaly detection method."""
        is_lof = method == "Local Outlier Factor"
        self.cb_use_eif.setVisible(not is_lof)

    # =====================================================================
    # ANOMALY DETECTION METHODS
    # =====================================================================
    def run_anomaly_detection(self):
        """Run anomaly detection using Isolation Forest or LOF."""
        if self.df is None:
            QMessageBox.warning(self, "Error", "Load data first")
            return

        # Get selected features
        selected = [item.text() for item in self.anomaly_features.selectedItems()]
        if not selected:
            # Use all numeric columns
            selected = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(selected) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 numeric features")
            return

        # Prepare data
        data = self.df[selected].copy()
        oxide_cols = selected

        try:
            # Configure detector
            config = AnomalyDetectorConfig(
                method="IsolationForest" if self.anomaly_method.currentText() == "Isolation Forest" else "LOF",
                contamination=self.anomaly_contamination.value(),
                n_estimators=self.anomaly_estimators.value(),
                use_eif=self.cb_use_eif.isChecked(),
                random_state=42
            )

            # Create detector
            detector = create_detector(config)

            # Create analyzer with correction service
            correction_service = CorrectionService(self.zero_strategy if self.zero_strategy else None)

            # Use CLR if selected
            transform_method = 'CLR' if self.cb_use_clr_anomaly.isChecked() else None

            analyzer = GeochemAnalyzer(
                correction_service=correction_service,
                transform_method=transform_method,
                detector=detector
            )

            # Set data and fit
            analyzer.set_data(data, oxide_cols)
            analyzer.fit_detector()

            # Get results
            result = analyzer.analyze()

            # Store results
            self.anomaly_results = result
            self.analyzer = analyzer
            self.detector = detector

            # Display results
            self.display_anomaly_results(result)

            self.log(f"Anomaly detection complete: {np.sum(result['labels'] == -1)} anomalies found")
            self.set_status(f"Anomaly detection: {np.sum(result['labels'] == -1)} anomalies found")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Anomaly detection failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.log(f"Anomaly detection error: {e}")

    def display_anomaly_results(self, result):
        """Display anomaly detection results."""
        # Clear figure
        self.anomaly_fig.clear()

        # Create subplots
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=self.anomaly_fig)
        ax1 = self.anomaly_fig.add_subplot(gs[0])
        ax2 = self.anomaly_fig.add_subplot(gs[1])

        # Plot anomaly scatter
        plot_anomaly_scatter(ax1, result['features'], result['labels'], result['scores'],
                            title="Anomaly Detection Results")

        # Plot score distribution
        threshold = None
        if hasattr(self.detector, 'threshold') and self.detector.threshold is not None:
            threshold = self.detector.threshold
        plot_anomaly_scores(ax2, result['scores'], threshold,
                           title="Anomaly Scores Distribution")

        self.anomaly_fig.tight_layout()
        self.anomaly_canvas.draw()

        # Update summary
        n_anomalies = np.sum(result['labels'] == -1)
        n_normals = np.sum(result['labels'] == 1)
        anomaly_ratio = n_anomalies / (n_anomalies + n_normals) if (n_anomalies + n_normals) > 0 else 0

        summary = f"""
        📊 Anomaly Detection Summary
        ================================
        • Method: {self.anomaly_method.currentText()}
        • Contamination: {self.anomaly_contamination.value():.2f}
        • Total samples: {len(result['labels'])}
        • Anomalies: {n_anomalies} ({anomaly_ratio*100:.1f}%)
        • Normal: {n_normals} ({(1-anomaly_ratio)*100:.1f}%)
        • Mean score: {np.mean(result['scores']):.3f}
        • Min score: {np.min(result['scores']):.3f}
        • Max score: {np.max(result['scores']):.3f}
        """

        if hasattr(self.detector, 'config'):
            summary += f"\n    • n_estimators: {self.detector.config.n_estimators}"
            if self.detector.config.use_eif:
                summary += "\n    • Using Extended Isolation Forest"

        self.anomaly_summary.setText(summary)

        # Update table
        df_result = result['features'].copy()
        df_result['Label'] = result['labels']
        df_result['Score'] = result['scores']

        # Show only first 100 rows
        display_df = df_result.head(100)

        self.anomaly_table.setRowCount(len(display_df))
        self.anomaly_table.setColumnCount(len(display_df.columns))
        self.anomaly_table.setHorizontalHeaderLabels(display_df.columns.tolist())

        for r in range(len(display_df)):
            for c in range(len(display_df.columns)):
                val = display_df.iloc[r, c]
                txt = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                item = QTableWidgetItem(txt)
                if display_df.columns[c] == 'Label' and val == -1:
                    item.setBackground(QColor(255, 200, 200))
                self.anomaly_table.setItem(r, c, item)

        self.anomaly_table.resizeColumnsToContents()

    def export_anomaly_results(self):
        """Export anomaly detection results to CSV."""
        if self.anomaly_results is None:
            QMessageBox.warning(self, "Error", "Run anomaly detection first")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Anomaly Results",
                                              "anomaly_results.csv", "CSV (*.csv)")
        if path:
            df_result = self.anomaly_results['features'].copy()
            df_result['Label'] = self.anomaly_results['labels']
            df_result['Score'] = self.anomaly_results['scores']
            df_result.to_csv(path, index=False)
            self.log(f"Anomaly results exported: {path}")

    def export_anomaly_figure(self):
        """Export anomaly detection figure."""
        if not hasattr(self, 'anomaly_fig'):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Anomaly Figure",
                                              "anomaly_detection.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.anomaly_fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Anomaly figure saved: {path}")

    def create_workflow_page(self):
        p = QWidget()
        l = QVBoxLayout(p)

        header = QLabel("📋 Coordinated Workflow Dashboard")
        header.setStyleSheet(f"font-size:18px;font-weight:bold;color:{Palette.PRIMARY};padding:10px;")
        l.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        workflow_widget = QWidget()
        workflow_layout = QVBoxLayout(workflow_widget)

        self.workflow_status = QTextEdit()
        self.workflow_status.setReadOnly(True)
        self.workflow_status.setStyleSheet("font-family: monospace; font-size: 12px; background: #f8f9fa;")
        self.workflow_status.setMinimumHeight(400)
        workflow_layout.addWidget(self.workflow_status)

        self.workflow_results = QTextEdit()
        self.workflow_results.setReadOnly(True)
        self.workflow_results.setMaximumHeight(150)
        workflow_layout.addWidget(QLabel("📈 Analysis Results Summary:"))
        workflow_layout.addWidget(self.workflow_results)

        scroll.setWidget(workflow_widget)
        l.addWidget(scroll)

        controls = QHBoxLayout()
        btn_refresh = styledbutton("Refresh Workflow", "refresh", "primary")
        btn_refresh.clicked.connect(self.update_workflow_page)
        controls.addWidget(btn_refresh)

        btn_report = styledbutton("Generate Full Report", "document", "success")
        btn_report.clicked.connect(self.generate_full_report)
        controls.addWidget(btn_report)

        controls.addStretch()
        l.addLayout(controls)

        self.stack.addWidget(p)

    def create_log_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        lc = QHBoxLayout()
        btn_clear = styledbutton("Clear", "trash", "danger")
        btn_clear.clicked.connect(self.clear_log)
        btn_export_log = styledbutton("Export", "download", "default")
        btn_export_log.clicked.connect(self.export_log)
        lc.addWidget(btn_clear)
        lc.addWidget(btn_export_log)
        lc.addStretch()
        l.addLayout(lc)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        l.addWidget(self.log_text)
        self.stack.addWidget(p)

    # =====================================================================
    # ZERO STRATEGY SELECTION
    # =====================================================================
    def choose_zero_strategy(self):
        dlg = ZeroReplacementDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            strategy = dlg.get_strategy()
            self.zero_strategy = strategy
            self.zero_strategy_label.setText(f"Strategy: {strategy.name()}")
            self.set_status(f"Zero strategy: {strategy.name()}")
            self.log(f"Zero strategy set: {strategy.name()}")

    # =====================================================================
    # CORRECTION METHODS
    # =====================================================================
    def apply_correction(self, method: str):
        """Apply closure correction to the data."""
        if self.df is None:
            QMessageBox.warning(self, "Error", "Load data first")
            return

        try:
            # Get oxide columns
            oxide_cols = [c for c in self.df.columns if
                         any(ox in c for ox in ['SiO2', 'Al2O3', 'MgO', 'CaO',
                                               'Fe2O3', 'Na2O', 'K2O', 'TiO2', 'MnO', 'P2O5'])]

            if not oxide_cols:
                QMessageBox.warning(self, "Error", "No oxide columns detected")
                return

            # Create dataset and service
            dataset = GeochemicalDataset(self.df, oxide_cols)
            service = CorrectionService(self.zero_strategy)

            # Apply correction
            corrected = service.correct(dataset, method)

            # Update dataframe
            self.df = corrected
            self.active_columns = list(corrected.columns)

            self.populate_vars()
            self.update_analysis_vars()
            self.update_clustering_vars()
            self.update_table()
            self.update_data_info()
            self.closure_status_label.setText(f"✅ Applied {method} correction")
            self.set_status(f"Applied {method} correction")
            self.log(f"Applied {method} correction")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_correction_dialog(self):
        """Show dialog to apply closure correction."""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded. Please upload data first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Apply Closure Correction")
        dlg.resize(400, 200)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Select correction method:"))
        method_combo = QComboBox()
        method_combo.addItems(["CLR", "ALR", "ILR"])
        layout.addWidget(method_combo)

        if self.zero_strategy is None:
            btn_zero = QPushButton("Choose Zero Strategy")
            btn_zero.clicked.connect(lambda: self.choose_zero_strategy())
            layout.addWidget(btn_zero)
            layout.addWidget(QLabel("⚠️ No zero strategy selected. Zeros will be replaced with 1e-10."))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(lambda: self.apply_correction(method_combo.currentText()))
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        dlg.exec_()

    def show_clr_interpretation(self):
        """Display CLR enrichment interpretation."""
        if self.df is None:
            QMessageBox.warning(self, "Error", "Load data first")
            return

        try:
            # Get oxide columns
            oxide_cols = [c for c in self.df.columns if
                         any(ox in c for ox in ['SiO2', 'Al2O3', 'MgO', 'CaO',
                                               'Fe2O3', 'Na2O', 'K2O', 'TiO2', 'MnO', 'P2O5'])]

            if not oxide_cols:
                QMessageBox.warning(self, "Error", "No oxide columns found")
                return

            # Prepare data
            data = self.df[oxide_cols].copy()
            if self.zero_strategy is not None:
                data = self.zero_strategy.replace(data)

            # Apply CLR with interpretation
            corrector = CLRCorrector()
            results = corrector.apply_with_enrichment(data, tolerance=0.05)

            # Build text
            text = "=" * 80 + "\n"
            text += "CLR ENRICHMENT INTERPRETATION\n"
            text += "=" * 80 + "\n\n"
            text += "💡 INTERPRETATION GUIDE:\n"
            text += "  • exp(CLR) > 1.05  → Element is ENRICHED\n"
            text += "  • exp(CLR) = 1.00  → Element is at AVERAGE abundance\n"
            text += "  • exp(CLR) < 0.95  → Element is DEPLETED\n\n"

            summary = results['summary']
            text += f"📊 Most enriched overall: {summary['most_enriched_overall']}\n"
            text += f"📊 Most depleted overall: {summary['most_depleted_overall']}\n\n"

            text += "📊 Classification Summary per Element:\n"
            for var, counts in summary['overall_classification_counts'].items():
                total = sum(counts.values())
                enriched_pct = (counts['enriched'] / total * 100) if total > 0 else 0
                depleted_pct = (counts['depleted'] / total * 100) if total > 0 else 0
                status = "↑" if enriched_pct > 60 else "↓" if depleted_pct > 60 else "≈"
                text += f"  {var}: {status}  enriched={enriched_pct:.0f}%, depleted={depleted_pct:.0f}%\n"

            text += "\n📊 Sample Interpretations (first 3 samples):\n"
            for interp in results['sample_interpretations'][:3]:
                text += f"\n📍 Sample: {interp['sample']}\n"
                text += f"   Most enriched: {interp['most_enriched']}\n"
                text += f"   Most depleted: {interp['most_depleted']}\n"
                text += f"   Enriched elements: {', '.join(interp['enriched_elements'][:5])}\n"
                text += f"   Depleted elements: {', '.join(interp['depleted_elements'][:5])}\n"
                text += f"   Classification: {interp['classification_summary']}\n"

            text += "\n" + "=" * 80 + "\n"

            dlg = QDialog(self)
            dlg.setWindowTitle("CLR Enrichment Interpretation")
            dlg.resize(700, 500)
            layout = QVBoxLayout(dlg)
            text_edit = QTextEdit()
            text_edit.setPlainText(text)
            text_edit.setReadOnly(True)
            text_edit.setFont(QFont("Courier New", 10))
            layout.addWidget(text_edit)
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dlg.accept)
            layout.addWidget(btn_close)
            dlg.exec_()

            self.log("CLR interpretation displayed")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def toggle_use_clr(self, checked):
        if checked:
            self.set_status("✅ Using CLR-corrected data for analysis")
        else:
            self.set_status("Using raw data")

    # =====================================================================
    # DATA OPERATIONS
    # =====================================================================
    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "CSV/Excel (*.csv *.xlsx *.xls);;All (*)")
        if not path:
            return

        try:
            self.filename = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()

            if ext == '.csv':
                for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        self.raw_df = pd.read_csv(path, encoding=enc)
                        break
                    except:
                        continue
            elif ext in ['.xlsx', '.xls']:
                self.raw_df = pd.read_excel(path, engine='openpyxl')

            if self.raw_df is None or self.raw_df.empty:
                raise ValueError("Cannot read file")

            self.raw_df.columns = [normalize_column_name(c) for c in self.raw_df.columns]
            self.df = self.raw_df.copy()
            self.active_columns = self.df.columns.tolist()

            self.log(f"Loaded {self.filename}: {self.df.shape}")
            self.set_status(f"Loaded: {self.filename}")

            self.populate_vars()
            self.update_analysis_vars()
            self.update_clustering_vars()
            self.update_anomaly_features()
            self.update_ratio_selector()
            self.update_table()
            self.update_data_info()

            self.stack.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_anomaly_features(self):
        """Update the anomaly feature list."""
        if self.df is not None:
            self.anomaly_features.clear()
            self.anomaly_features.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())

    def switch_data_view(self, mode):
        if mode == "Raw Data" and self.raw_df is not None:
            self.update_table(data=self.raw_df)
        elif mode == "Analysis Data":
            self.update_table(data=self.get_analysis_df())
        elif mode == "Corrected Data":
            if hasattr(self, 'df') and self.df is not None:
                self.update_table(data=self.df)
            else:
                self.update_table(data=self.df)
        else:
            self.update_table(data=self.df)

    def get_analysis_df(self):
        if self.df is None:
            return None
        if not self.active_columns:
            return self.df.copy()
        cols = [c for c in self.active_columns if c in self.df.columns]
        if self.index_column and self.index_column in self.df.columns and self.index_column not in cols:
            cols.insert(0, self.index_column)
        return self.df[cols].copy() if cols else self.df.copy()

    def update_table(self, data=None):
        df = data if data is not None else self.df
        if df is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        disp = df if self.show_all else df.head(self.display_rows).iloc[:, :min(self.display_cols, len(df.columns))]
        self.table.setRowCount(len(disp))
        self.table.setColumnCount(len(disp.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in disp.columns])
        for r in range(len(disp)):
            for c in range(len(disp.columns)):
                val = disp.iloc[r, c]
                txt = "" if pd.isna(val) else f"{val:.4g}" if isinstance(val, (float, np.floating)) else str(val)
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()

    def populate_vars(self):
        if self.df is not None:
            self.dist_var.clear()
            self.dist_var.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())

    def update_analysis_vars(self):
        if self.df is not None:
            self.analysis_vars.clear()
            self.analysis_vars.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())
            for i in range(min(10, self.analysis_vars.count())):
                self.analysis_vars.item(i).setSelected(True)

    def update_clustering_vars(self):
        if self.df is not None:
            self.cluster_vars.clear()
            self.cluster_vars.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())
            for i in range(min(10, self.cluster_vars.count())):
                self.cluster_vars.item(i).setSelected(True)

    def update_ratio_selector(self):
        if self.df is not None:
            self.ratio_selector.clear()
            self.ratio_selector.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())

    def update_data_info(self):
        if self.df is not None:
            self.data_info.setText(f"Rows:{self.df.shape[0]} Cols:{self.df.shape[1]} | Active:{len(self.active_columns)}")

    def update_workflow_page(self):
        text = "╔" + "═" * 68 + "╗\n"
        text += "║" + " " * 20 + "GEOCHEMICAL ANALYSIS WORKFLOW" + " " * 20 + "║\n"
        text += "╚" + "═" * 68 + "╝\n\n"

        text += "┌─ " + "━" * 65 + " ─┐\n"
        text += "│ 📁 DATA STATUS" + " " * 53 + "│\n"
        text += "├─" + "─" * 66 + "─┤\n"
        if self.filename:
            text += f"│   • File: {self.filename}" + " " * (66 - len(self.filename) - 10) + "│\n"
            text += f"│   • Rows: {len(self.df) if self.df is not None else 0}" + " " * (66 - len(str(len(self.df) if self.df is not None else 0)) - 10) + "│\n"
            text += f"│   • Columns: {len(self.df.columns) if self.df is not None else 0}" + " " * (66 - len(str(len(self.df.columns) if self.df is not None else 0)) - 12) + "│\n"
        else:
            text += "│   • No data loaded" + " " * 48 + "│\n"
        text += "└─" + "─" * 66 + "─┘\n\n"

        text += "┌─ " + "━" * 65 + " ─┐\n"
        text += "│ 🔬 ANOMALY DETECTION" + " " * 44 + "│\n"
        text += "├─" + "─" * 66 + "─┤\n"
        if self.anomaly_results is not None:
            n_anomalies = np.sum(self.anomaly_results['labels'] == -1)
            n_normals = np.sum(self.anomaly_results['labels'] == 1)
            text += f"│   • Method: {self.anomaly_method.currentText()}" + " " * (66 - len(self.anomaly_method.currentText()) - 11) + "│\n"
            text += f"│   • Anomalies: {n_anomalies}" + " " * (66 - len(str(n_anomalies)) - 13) + "│\n"
            text += f"│   • Normal: {n_normals}" + " " * (66 - len(str(n_normals)) - 11) + "│\n"
        else:
            text += "│   • Not run" + " " * 53 + "│\n"
        text += "└─" + "─" * 66 + "─┘\n\n"

        text += "┌─ " + "━" * 65 + " ─┐\n"
        text += "│ 🔧 PREPROCESSING" + " " * 48 + "│\n"
        text += "├─" + "─" * 66 + "─┤\n"
        if self.zero_strategy:
            text += f"│   • Zero strategy: {self.zero_strategy.name()}" + " " * (66 - len(self.zero_strategy.name()) - 19) + "│\n"
        else:
            text += "│   • Zero strategy: None" + " " * 44 + "│\n"
        text += "└─" + "─" * 66 + "─┘\n"

        self.workflow_status.setText(text)

    # =====================================================================
    # DATA CLEANING & FILTERING
    # =====================================================================
    def advanced_clean_data(self):
        if self.df is None:
            return
        dlg = AdvancedCleanDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            opts = {k: v.isChecked() for k, v in dlg.cbs.items()}
            self.df, report = clean_dataframe_advanced(self.df, opts)
            self.active_columns = [c for c in self.active_columns if c in self.df.columns]
            self.log(report)
            self.update_table()
            self.update_data_info()
            self.populate_vars()
            self.update_analysis_vars()
            self.update_clustering_vars()
            self.update_ratio_selector()
            self.update_anomaly_features()

    def open_filter_dialog(self):
        if self.df is None:
            return
        dlg = DataFilterDialog(self.df.columns.tolist(), self)
        if dlg.exec_() == QDialog.Accepted:
            self.apply_filter(*dlg.get_filter_config())

    def apply_filter(self, col, ftype, val, op):
        if col not in self.df.columns:
            return
        try:
            s = pd.to_numeric(self.df[col], errors='coerce')
            mask = pd.Series([True] * len(self.df))
            if ftype == "Outliers (IQR)":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                mask = (s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)
            elif ftype == "Percentile":
                thresh = s.quantile(val / 100.0)
                if op == "<":
                    mask = s < thresh
                elif op == ">":
                    mask = s > thresh
                else:
                    mask = s == thresh
            else:
                if op == "<":
                    mask = s < val
                elif op == ">":
                    mask = s > val
                else:
                    mask = s == val
            self.df = self.df[mask].reset_index(drop=True)
            self.log(f"Filtered: kept {mask.sum()}")
            self.update_table()
            self.populate_vars()
            self.update_anomaly_features()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def delete_data(self):
        if self.df is None:
            return
        dlg = DeleteDataDialog(self.df, self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                if dlg.rb_rows_range.isChecked():
                    parts = dlg.row_range.text().split('-')
                    if len(parts) == 2:
                        start, end = int(parts[0].strip()), int(parts[1].strip())
                        self.df = self.df.drop(self.df.index[start:end+1]).reset_index(drop=True)
                        self.log(f"Deleted rows {start} to {end}")
                elif dlg.rb_rows_condition.isChecked():
                    col = dlg.row_cond_col.currentText()
                    val = dlg.row_cond_val.text()
                    if col in self.df.columns:
                        self.df = self.df[self.df[col].astype(str) != val].reset_index(drop=True)
                        self.log(f"Deleted rows where {col} == {val}")
                elif dlg.rb_rows_all.isChecked():
                    self.df = self.df.iloc[0:0]
                    self.log("Deleted all rows")

                selected = [item.text() for item in dlg.col_list.selectedItems()]
                if selected:
                    self.df = self.df.drop(columns=selected, errors='ignore')
                    self.active_columns = [c for c in self.active_columns if c not in selected]
                    self.log(f"Deleted columns: {', '.join(selected)}")

                if dlg.cb_remove_from_app.isChecked():
                    self.df = None
                    self.raw_df = None
                    self.filename = None
                    self.active_columns = []
                    self.index_column = None
                    self.anomaly_results = None
                    self.last_geochem_data = {"alterationindices": {}, "elementratios": {}, "gradestats": {}}
                    self.last_analysis_report = ""
                    self.last_cluster_data = None
                    self.svm_params = None
                    self.last_svm_data = None

                    if hasattr(self, 'alteration_view'):
                        self.alteration_view.clear()
                    if hasattr(self, 'ratio_view'):
                        self.ratio_view.clear()
                    if hasattr(self, 'grade_view'):
                        self.grade_view.clear()
                    if hasattr(self, 'ana_report'):
                        self.ana_report.clear()
                    if hasattr(self, 'cluster_features_label'):
                        self.cluster_features_label.setText("🔍 Distinguishing features will appear here after running clustering")
                    if hasattr(self, 'svm_report'):
                        self.svm_report.clear()
                    if hasattr(self, 'closure_status_label'):
                        self.closure_status_label.setText("")
                    if hasattr(self, 'anomaly_summary'):
                        self.anomaly_summary.clear()
                    self.anomaly_fig.clear()
                    if hasattr(self, 'anomaly_canvas'):
                        self.anomaly_canvas.draw()

                    self.log("Removed all data from application")
                    self.set_status("All data removed")

                self.update_table()
                self.update_data_info()
                self.populate_vars()
                self.update_analysis_vars()
                self.update_clustering_vars()
                self.update_ratio_selector()
                self.update_anomaly_features()

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def global_refresh(self):
        try:
            self.set_status("Refreshing all views...")
            QApplication.processEvents()

            if self.df is not None:
                self.update_table()
                self.update_data_info()
                self.populate_vars()
                self.update_analysis_vars()
                self.update_clustering_vars()
                self.update_ratio_selector()
                self.update_anomaly_features()
                self.update_workflow_page()
                self.last_analysis_report = ""
                if hasattr(self, 'ana_report'):
                    self.ana_report.clear()
                self.log("All views refreshed successfully")
                self.set_status(f"Refreshed - {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            else:
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                self.data_info.setText("No data loaded")
                self.dist_var.clear()
                self.analysis_vars.clear()
                self.cluster_vars.clear()
                self.ratio_selector.clear()
                self.anomaly_features.clear()
                if hasattr(self, 'alteration_view'):
                    self.alteration_view.clear()
                if hasattr(self, 'ratio_view'):
                    self.ratio_view.clear()
                if hasattr(self, 'grade_view'):
                    self.grade_view.clear()
                if hasattr(self, 'ana_report'):
                    self.ana_report.clear()
                if hasattr(self, 'svm_report'):
                    self.svm_report.clear()
                if hasattr(self, 'closure_status_label'):
                    self.closure_status_label.setText("")
                if hasattr(self, 'anomaly_summary'):
                    self.anomaly_summary.clear()
                self.anomaly_fig.clear()
                if hasattr(self, 'anomaly_canvas'):
                    self.anomaly_canvas.draw()
                if hasattr(self, 'workflow_status'):
                    self.workflow_status.setText("No data loaded. Upload a geochemical dataset to start.")
                self.log("All views cleared - no data loaded")
                self.set_status("No data loaded - ready for upload")

            QApplication.processEvents()
        except Exception as e:
            self.log(f"Refresh error: {str(e)}")
            self.set_status("Refresh failed - try re-uploading data", error=True)

    def control_display(self):
        if self.df is None:
            return
        dlg = DisplayControlDialog(len(self.df), len(self.df.columns), self)
        if dlg.exec_() == QDialog.Accepted:
            self.show_all = dlg.cb_full.isChecked()
            self.display_rows = dlg.sb_rows.value()
            self.display_cols = dlg.sb_cols.value()
            self.update_table()

    def set_index_columns(self):
        if self.df is None:
            return
        dlg = IndexColumnDialog(self.df.columns.tolist(), self.index_column, self)
        if dlg.exec_() == QDialog.Accepted:
            self.index_column = dlg.index_combo.currentText() if dlg.index_combo.currentText() != "None" else None
            self.active_columns = [item.text() for item in dlg.active_list.selectedItems()]
            self.update_data_info()

    def rename_columns_dialog(self):
        if self.df is None:
            return
        old, ok = QInputDialog.getItem(self, "Rename", "Column:", self.df.columns.tolist(), 0, False)
        if ok and old:
            new, ok2 = QInputDialog.getText(self, "Rename", f"New name:")
            if ok2 and new:
                self.df.columns = [new if c == old else c for c in self.df.columns]
                self.update_table()
                self.update_anomaly_features()

    def calculate_element_ratio(self):
        if self.df is None:
            return
        nc = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(nc) < 2:
            return
        dlg = ElementRatioDialog(nc, self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                self.df[dlg.name_input.text()] = pd.to_numeric(
                    self.df[dlg.num_combo.currentText()], errors='coerce'
                ) / (pd.to_numeric(self.df[dlg.den_combo.currentText()], errors='coerce') + 1e-9)
                self.update_table()
                self.populate_vars()
                self.update_anomaly_features()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def reset_data(self):
        if self.raw_df is None:
            return
        if QMessageBox.question(self, "Reset", "Reset all?") == QMessageBox.Yes:
            self.df = self.raw_df.copy()
            self.active_columns = self.df.columns.tolist()
            self.index_column = None
            self.anomaly_results = None
            self.update_table()
            self.update_data_info()
            self.populate_vars()
            self.update_anomaly_features()
            self.log("Data reset")

    def unify_units(self):
        if self.df is None:
            return
        for c in self.df.columns:
            if "ptc" in c.lower():
                new = c.replace("ptc", "%")
                self.df.rename(columns={c: new}, inplace=True)
        self.update_table()
        self.update_anomaly_features()

    # =====================================================================
    # DISTRIBUTION PAGE
    # =====================================================================
    def show_plot_config_safe(self):
        dlg = PlotConfigDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.last_plot_config = dlg.get_selection()
            config_text = ", ".join(self.last_plot_config)
            self.plot_title_label.setText(f"📊 Distribution Plot: {config_text}")

    def update_distribution(self):
        if self.df is None:
            return
        var = self.dist_var.currentText()
        if not var:
            return
        sel = self.last_plot_config or ["Histogram", "PDF KDE"]
        data = pd.to_numeric(self.df[var], errors='coerce').dropna()
        if len(data) == 0:
            return

        self.dist_fig.clear()
        n = len(sel)
        cols = 2 if n > 1 else 1
        rows = (n + 1) // 2
        gs = gridspec.GridSpec(rows, cols, figure=self.dist_fig)

        for i, pt in enumerate(sel):
            ax = self.dist_fig.add_subplot(gs[i])
            if pt == "Histogram":
                ax.hist(data, bins=30, color=Palette.PRIMARY, alpha=0.7, edgecolor='white')
                ax.set_ylabel('Frequency')
            elif pt == "PDF KDE":
                sns.kdeplot(data, ax=ax, color=Palette.PRIMARY, fill=True, alpha=0.5)
                ax.set_ylabel('Density')
            elif pt == "CDF":
                ax.plot(np.sort(data), np.linspace(0, 1, len(data)), color=Palette.PRIMARY, linewidth=2)
                ax.set_ylabel('Cumulative Probability')
            elif pt == "Box Plot":
                ax.boxplot(data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor=Palette.PRIMARYLIGHT, color=Palette.PRIMARY),
                          whiskerprops=dict(color=Palette.PRIMARY),
                          capprops=dict(color=Palette.PRIMARY),
                          medianprops=dict(color=Palette.PRIMARYDARK, linewidth=2))
                ax.set_ylabel('Value')
            elif pt == "Q-Q Plot":
                scipystats.probplot(data, dist="norm", plot=ax)
                ax.get_lines()[0].set_color(Palette.PRIMARY)
                ax.get_lines()[1].set_color('red')
                ax.set_ylabel('Sample Quantiles')

            ax.set_xlabel(var)
            ax.set_title(pt)
            ax.grid(True, alpha=0.3)

        self.dist_fig.tight_layout()
        self.dist_canvas.draw()

        config_text = ", ".join(sel)
        self.plot_title_label.setText(f"📊 Distribution Plot: {config_text} for {var}")

    def export_distribution_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "distribution.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.dist_fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Distribution plot saved: {path}")

    # =====================================================================
    # ANALYSIS PAGE
    # =====================================================================
    def update_analysis(self):
        if self.df is None:
            return
        adf = self.get_analysis_df()
        if adf is None:
            return
        sel = [item.text() for item in self.analysis_vars.selectedItems()]
        if not sel:
            sel = adf.select_dtypes(include=[np.number]).columns.tolist()[:10]
        av = [v for v in sel if v in adf.columns]
        if not av:
            return
        nd = adf[av].select_dtypes(include=[np.number])
        if nd.shape[1] < 2:
            return
        atype = self.analysis_type.currentText()
        self.ana_fig.clear()
        report = ""

        if atype == "Correlation Matrix":
            ax = self.ana_fig.add_subplot(111)
            corr, im = plot_correlation_heatmap(ax, nd, title="Correlation Matrix")
            plt.colorbar(im, ax=ax, shrink=0.8)

            strong_corr = get_strong_correlations(corr)
            if strong_corr:
                report = "🔗 Strong correlations (|r|>0.7):\n" + "\n".join(
                    f"{p['pair']}: {p['correlation']:.2f}" for p in strong_corr
                )
            else:
                report = "No strong correlations found (|r|<0.7 for all pairs)"

        elif atype == "PCA Analysis":
            ax = self.ana_fig.add_subplot(111)
            pca_obj, scores = plot_pca_biplot(
                ax, nd,
                show_loadings=self.cb_show_pca.isChecked(),
                title="PCA Biplot"
            )
            if pca_obj is not None:
                report = pca_summary_text(pca_obj, nd.columns)
            else:
                report = "PCA could not be computed - insufficient data"

        elif atype == "Scatter Plot":
            ax = self.ana_fig.add_subplot(111)
            if nd.shape[1] >= 2:
                x_var = nd.columns[0]
                y_var = nd.columns[1] if nd.shape[1] >= 2 else nd.columns[0]
                colors = nd.iloc[:, 2].values if nd.shape[1] >= 3 else np.random.rand(len(nd))
                scatter = ax.scatter(nd.iloc[:, 0], nd.iloc[:, 1], c=colors, cmap='viridis',
                                    alpha=0.7, edgecolors='white', s=80)
                plt.colorbar(scatter, ax=ax, label='Color by value')
                ax.set_xlabel(nd.columns[0])
                ax.set_ylabel(nd.columns[1])
                ax.set_title(f"{nd.columns[0]} vs {nd.columns[1]}")
                corr_val = nd.iloc[:, 0].corr(nd.iloc[:, 1])
                ax.grid(True, alpha=0.3)
                report = f"📈 Scatter Plot:\n• {nd.columns[0]} vs {nd.columns[1]}\n• Pearson r = {corr_val:.4f}\n• R² = {corr_val**2:.4f}"

        self.ana_fig.tight_layout()
        self.ana_canvas.draw()
        self.ana_report.setText(report)
        self.last_analysis_report = f"{atype}:\n{report}"
        self.update_statistics()

    def export_analysis_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "analysis.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.ana_fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Analysis plot saved: {path}")

    def update_statistics(self):
        if self.df is None:
            return
        adf = self.get_analysis_df()
        stats = EnhancedStatisticsEngine.compute(adf) if adf is not None else pd.DataFrame()
        if stats.empty:
            return
        self.stats_table.setRowCount(len(stats))
        self.stats_table.setColumnCount(len(stats.columns))
        self.stats_table.setHorizontalHeaderLabels(stats.columns.tolist())
        for i in range(len(stats)):
            for j in range(len(stats.columns)):
                self.stats_table.setItem(i, j, QTableWidgetItem(str(stats.iloc[i, j])))
        self.stats_table.resizeColumnsToContents()

    def export_statistics(self):
        if self.df is None:
            return
        stats = EnhancedStatisticsEngine.compute(self.get_analysis_df())
        if stats.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "statistics.csv", "CSV (*.csv)")
        if path:
            stats.to_csv(path, index=False)
            self.log(f"Statistics exported: {path}")

    # =====================================================================
    # GEOCHEMISTRY PAGE
    # =====================================================================
    def calculate_alteration_only(self):
        if self.df is None:
            return
        adf = self.get_analysis_df()
        if adf is None:
            return
        selected = [k for k, v in self.alteration_cbs.items() if v.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one index")
            return
        try:
            results = EnhancedGeochemistryEngine.calculate_alteration_indices(adf, selected)
            self.last_geochem_data["alterationindices"] = results
            self.alteration_view.setHtml(self.dict_to_html("Alteration Indices", results))
            self.log("Alteration indices calculated")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def calculate_ratios_only(self):
        if self.df is None:
            return
        adf = self.get_analysis_df()
        if adf is None:
            return
        try:
            selected = [item.text() for item in self.ratio_selector.selectedItems()]
            results = EnhancedGeochemistryEngine.calculate_element_ratios(adf, selected)
            self.last_geochem_data["elementratios"] = results
            self.ratio_view.setHtml(self.dict_to_html("Element Ratios", results))
            self.log(f"Ratios: {len(results)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def calculate_grades_only(self):
        if self.df is None:
            return
        adf = self.get_analysis_df()
        if adf is None:
            return
        selected = [item.text() for item in self.grade_elements_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one element")
            return
        try:
            results = EnhancedGeochemistryEngine.calculate_grade_statistics(adf, selected, self.grade_threshold.value())
            self.last_geochem_data["gradestats"] = results
            self.grade_view.setHtml(self.dict_to_html("Grade Statistics", results, True))
            self.log("Grade stats calculated")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def dict_to_html(self, title, data, is_nested=False):
        h = [f'<h3 style="color:{Palette.PRIMARY};">{title}</h3>']
        if not data:
            h.append("<p>No data. Click Run to calculate.</p>")
            return "".join(h)
        h.append('<table border="1" style="width:100%;border-collapse:collapse;font-size:12px;">')
        if is_nested:
            fv = next(iter(data.values()))
            if isinstance(fv, dict):
                keys = list(fv.keys())
                h.append('<tr style="background:#f0f0f0;"><th>Element</th>' +
                        ''.join(f'<th>{k}</th>' for k in keys) + '</tr>')
                for k, v in data.items():
                    h.append(f'<tr><td><b>{k}</b></td>' +
                            ''.join(f'<td style="text-align:right;">{v.get(ik, "N/A")}</td>' for ik in keys) + '</tr>')
        else:
            h.append('<tr style="background:#f0f0f0;"><th>Parameter</th><th>Value</th></tr>')
            for k, v in data.items():
                h.append(f'<tr><td>{k}</td><td style="text-align:right;">{v}</td></tr>')
        h.append('</table>')
        return "".join(h)

    def export_geochem_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "geochem_results.xlsx", "Excel (*.xlsx)")
        if path:
            with pd.ExcelWriter(path, engine='openpyxl') as w:
                for k, data in self.last_geochem_data.items():
                    if data:
                        try:
                            df = pd.DataFrame.from_dict(data, orient='index') if isinstance(next(iter(data.values())), dict) else pd.DataFrame(list(data.items()), columns=['Param', 'Value'])
                            df.to_excel(w, sheet_name=k[:31], index=False)
                        except:
                            pass
            self.log(f"Geochemistry results exported: {path}")

    # =====================================================================
    # CLUSTERING PAGE
    # =====================================================================
    def toggle_cluster_params(self, method):
        self.kmeans_group.setVisible(method == "K-Means")
        self.dbscan_group.setVisible(method == "DBSCAN")

    def run_clustering(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Load data first")
            return

        adf = self.get_analysis_df()
        if adf is None:
            return

        sel = [item.text() for item in self.cluster_vars.selectedItems()]
        if not sel:
            sel = adf.select_dtypes(include=[np.number]).columns.tolist()
        else:
            sel = [c for c in sel if c in adf.columns and pd.api.types.is_numeric_dtype(adf[c])]

        if len(sel) < 2:
            QMessageBox.warning(self, "Error", "Need 2+ numeric columns")
            return

        X = adf[sel].dropna().values
        if len(X) == 0:
            QMessageBox.warning(self, "Error", "No complete cases")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        method = self.cluster_method.currentText()

        try:
            if method == "K-Means":
                k = self.kmeans_k.value()
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            else:
                eps = self.dbscan_eps.value()
                min_samples = self.dbscan_min_samples.value()
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            cluster_features = {}
            for lab in unique_labels:
                if lab == -1:
                    continue
                mask = labels == lab
                cluster_data = X_scaled[mask]
                if len(cluster_data) > 0:
                    means = cluster_data.mean(axis=0)
                    overall_means = X_scaled.mean(axis=0)
                    deviations = means - overall_means
                    top_idx = np.argsort(np.abs(deviations))[-5:][::-1]
                    cluster_features[lab] = [(sel[i], deviations[i]) for i in top_idx]

            pca = SklearnPCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            self.last_cluster_data = {
                'labels': labels,
                'X_pca': X_pca,
                'model': model,
                'selected_vars': sel,
                'df': adf[sel].dropna(),
                'cluster_features': cluster_features,
                'n_clusters': n_clusters,
                'method': method
            }

            self.cluster_fig.clear()

            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=self.cluster_fig)
            ax1 = self.cluster_fig.add_subplot(gs[0])
            ax2 = self.cluster_fig.add_subplot(gs[1])

            colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
            plot_clustering_scatter(ax1, X_pca, labels, n_clusters, method=method, colors=colors)
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

            if method == "K-Means" and hasattr(model, 'cluster_centers_'):
                centers_pca = pca.transform(model.cluster_centers_)
                ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
                          s=200, edgecolors='black', linewidths=2, label='Centers')

            plot_cluster_sizes(ax2, labels, colors=colors)

            self.cluster_fig.tight_layout()
            self.cluster_canvas.draw()

            self.show_cluster_stats(labels, X_scaled, sel)

            feature_text = "🔍 Distinguishing Features per Cluster:\n\n"
            for lab, features in cluster_features.items():
                feature_text += f"📌 Cluster {lab} (n={np.sum(labels == lab)}):\n"
                for var, dev in features:
                    direction = "↑ HIGH" if dev > 0 else "↓ LOW"
                    feature_text += f"   • {var}: {direction} (deviation: {dev:.2f})\n"
                feature_text += "\n"
            if not cluster_features:
                feature_text = "No clusters found with sufficient members for feature analysis."
            self.cluster_features_label.setText(feature_text)

            self.log(f"Clustering complete: {n_clusters} clusters found")
            self.set_status(f"Clustering: {n_clusters} clusters")

        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", str(e))
            self.log(f"Clustering error: {e}")

    def show_cluster_stats(self, labels, X_scaled, var_names):
        unique_labels = np.unique(labels)
        rows = []
        for cl in unique_labels:
            mask = labels == cl
            count = np.sum(mask)
            row = {'Cluster': f'Cluster {cl}' if cl != -1 else 'Noise', 'Count': count}
            if count > 0:
                cluster_data = X_scaled[mask]
                for i, var in enumerate(var_names[:5]):
                    row[var] = round(float(cluster_data[:, i].mean()), 2)
            else:
                for var in var_names[:5]:
                    row[var] = 'N/A'
            rows.append(row)
        stats_df = pd.DataFrame(rows)
        self.cluster_stats.setRowCount(len(stats_df))
        self.cluster_stats.setColumnCount(len(stats_df.columns))
        self.cluster_stats.setHorizontalHeaderLabels(stats_df.columns.tolist())
        for i in range(len(stats_df)):
            for j in range(len(stats_df.columns)):
                self.cluster_stats.setItem(i, j, QTableWidgetItem(str(stats_df.iloc[i, j])))
        self.cluster_stats.resizeColumnsToContents()

    def export_cluster_figure(self):
        if not hasattr(self, 'cluster_fig'):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "clustering.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.cluster_fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"Cluster figure saved: {path}")

    def export_cluster_assignments(self):
        if self.last_cluster_data is None:
            return
        data = self.last_cluster_data
        df_out = data['df'].copy()
        df_out['Cluster'] = data['labels']
        path, _ = QFileDialog.getSaveFileName(self, "Save", "clusters.csv", "CSV (*.csv)")
        if path:
            df_out.to_csv(path, index=False)
            self.log(f"Cluster assignments saved: {path}")

    # =====================================================================
    # SVM PAGE
    # =====================================================================
    def configure_svm(self):
        if self.df is None:
            return
        dlg = SVMDialog(self.df.columns.tolist(), self)
        if dlg.exec_() == QDialog.Accepted:
            self.svm_params = dlg.get_params()
            self.log("SVM configured")

    def run_svm(self):
        if self.svm_params is None or self.df is None:
            QMessageBox.warning(self, "Error", "Configure SVM first")
            return
        params = self.svm_params
        if not params['features']:
            QMessageBox.warning(self, "Error", "No features selected")
            return

        try:
            numeric_cols = self.get_numeric_cols()
            valid_features = [f for f in params['features'] if f in numeric_cols]
            if not valid_features:
                QMessageBox.warning(self, "Error", "No valid numeric features selected")
                return

            df = self.df[[params['target']] + valid_features].copy()
            df[params['target']] = df[params['target']].astype(str)
            for col in valid_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()

            if len(df) < 10:
                QMessageBox.warning(self, "Error", f"Only {len(df)} complete cases. Need at least 10.")
                return

            X = df[valid_features].values
            y_raw = df[params['target']].values

            unique_classes, class_counts = np.unique(y_raw, return_counts=True)
            if len(unique_classes) < 2:
                QMessageBox.warning(self, "Error", f"Target column '{params['target']}' has only {len(unique_classes)} class. Need at least 2.")
                return

            if np.any(class_counts < 2):
                small_mask = class_counts < 3
                if small_mask.any():
                    small_classes = unique_classes[small_mask]
                    y_clean = y_raw.copy()
                    for sc in small_classes:
                        y_clean[y_clean == sc] = 'Other'
                    self.log(f"Merged {len(small_classes)} small classes into 'Other'")
                    y_raw = y_clean
                    unique_classes, class_counts = np.unique(y_raw, return_counts=True)
                    if len(unique_classes) < 2:
                        QMessageBox.warning(self, "Error", "After merging small classes, still not enough classes.")
                        return

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            all_class_names = [str(c) for c in le.classes_]

            # FIX: split BEFORE scaling to avoid test-set leakage
            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=params['test_size'], random_state=42, stratify=y)
            except:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=params['test_size'], random_state=42)

            scaler = StandardScaler().fit(X_train_raw)
            X_train = scaler.transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
            X_scaled = scaler.transform(X)

            model = SVC(kernel=params['kernel'], C=params['C'],
                       degree=params['degree'] if params['kernel'] == 'poly' else 3,
                       random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_classes = np.unique(np.concatenate([y_test, y_pred]))
            test_class_names = [all_class_names[i] for i in test_classes if i < len(all_class_names)]

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=test_classes)
            report = classification_report(y_test, y_pred, labels=test_classes,
                                         target_names=test_class_names, zero_division=0)

            self.last_svm_data = {'accuracy': acc, 'report': report, 'class_names': test_class_names}

            self.svm_fig.clear()

            pca = SklearnPCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            X_test_pca = pca.transform(X_test)

            ax1 = self.svm_fig.add_subplot(121)
            ax2 = self.svm_fig.add_subplot(122)

            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            b_model = SVC(kernel=params['kernel'], C=params['C'],
                         degree=params['degree'] if params['kernel'] == 'poly' else 3,
                         random_state=42)
            b_model.fit(X_pca, y)
            Z = b_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            n_classes = len(test_classes)
            if n_classes <= 10:
                ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu, levels=n_classes)
            else:
                ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

            scatter = ax1.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                                 cmap=plt.cm.RdYlBu, edgecolors='black', s=60)
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax1.set_title(f"SVM Decision Boundary\nAccuracy: {acc:.2%}")
            if n_classes <= 10:
                ax1.legend(*scatter.legend_elements(), title="Classes", loc='upper right', fontsize=7)

            if cm.shape[0] <= 15:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                           xticklabels=test_class_names, yticklabels=test_class_names,
                           annot_kws={'size': 7})
            else:
                sns.heatmap(cm, cmap='Blues', ax=ax2)
            ax2.set_title("Confusion Matrix")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")

            self.svm_fig.tight_layout()
            self.svm_canvas.draw()

            self.svm_accuracy_table.setRowCount(1)
            self.svm_accuracy_table.setColumnCount(4)
            self.svm_accuracy_table.setHorizontalHeaderLabels(["Accuracy", "Precision", "Recall", "F1"])
            try:
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0, labels=test_classes)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0, labels=test_classes)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=test_classes)
            except:
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            for j, v in enumerate([acc, prec, rec, f1]):
                self.svm_accuracy_table.setItem(0, j, QTableWidgetItem(f"{v:.2%}"))
            self.svm_accuracy_table.resizeColumnsToContents()

            self.svm_report.setText(f"Classification Report:\n\n{report}")
            self.log(f"SVM accuracy: {acc:.3f}")
            self.set_status(f"SVM Acc: {acc:.2%}")

        except Exception as e:
            QMessageBox.critical(self, "SVM Error", f"{str(e)}\n\n{traceback.format_exc()}")
            self.log(f"SVM error: {e}")

    def visualize_overfitting(self):
        if self.svm_params is None or self.df is None:
            return
        params = self.svm_params
        if not params['features']:
            return
        valid_features = [f for f in params['features'] if f in self.get_numeric_cols()]
        if not valid_features:
            return

        try:
            df = self.df[[params['target']] + valid_features].copy()
            for col in valid_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            if len(df) == 0:
                return

            X = df[valid_features].values
            y_raw = df[params['target']].astype(str).values
            unique_classes, class_counts = np.unique(y_raw, return_counts=True)
            if len(unique_classes) < 2:
                return

            le = LabelEncoder()
            y = le.fit_transform(y_raw)

            # FIX: split BEFORE scaling to avoid test-set leakage
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=params['test_size'], random_state=42)
            scaler = StandardScaler().fit(X_train_raw)
            X_train = scaler.transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            C_range = np.logspace(-2, 3, 10)
            train_scores, test_scores = [], []

            for C in C_range:
                model = SVC(kernel=params['kernel'], C=C,
                           degree=params['degree'] if params['kernel'] == 'poly' else 3,
                           random_state=42)
                model.fit(X_train, y_train)
                train_scores.append(model.score(X_train, y_train))
                test_scores.append(model.score(X_test, y_test))

            self.svm_fig.clear()
            ax = self.svm_fig.add_subplot(111)
            ax.semilogx(C_range, train_scores, 'o-', label='Train', markersize=8, color=Palette.PRIMARY)
            ax.semilogx(C_range, test_scores, 's-', label='Test', markersize=8, color=Palette.DANGER)
            ax.set_xlabel('C (Regularization Parameter)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Overfitting Analysis - Train vs Test Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            best_idx = np.argmax(test_scores)
            optimal_C = C_range[best_idx]
            best_test_score = test_scores[best_idx]
            ax.axvline(optimal_C, color='green', linestyle='--', alpha=0.5,
                      label=f'Optimal C={optimal_C:.3f}')
            ax.legend()

            self.svm_fig.tight_layout()
            self.svm_canvas.draw()

            self.log(f"Overfitting analysis completed. Optimal C: {optimal_C:.3f}")

        except Exception as e:
            QMessageBox.critical(self, "Overfitting Analysis Error", str(e))

    def export_svm_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "svm_plot.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.svm_fig.savefig(path, dpi=300, bbox_inches='tight')
            self.log(f"SVM figure saved: {path}")

    # =====================================================================
    # EXPORT / LOG / HELP
    # =====================================================================
    def export_current_figure(self):
        idx = self.stack.currentIndex()
        if idx == 2:
            self.export_distribution_plot()
        elif idx == 3:
            self.export_analysis_plot()
        elif idx == 5:
            self.export_cluster_figure()
        elif idx == 6:
            self.export_svm_figure()
        elif idx == 7:
            self.export_anomaly_figure()

    def export_data_csv(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "data.csv", "CSV (*.csv);;Excel (*.xlsx)")
        if path:
            if path.endswith('.xlsx'):
                self.df.to_excel(path, index=False)
            else:
                self.df.to_csv(path, index=False)
            self.log(f"Data exported: {path}")

    def generate_full_report(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Upload data first")
            return

        adf = self.get_analysis_df()
        stats_df = EnhancedStatisticsEngine.compute(adf)

        html = f"""<html><head><title>{APPNAME} Report</title>
        <style>
            body{{font-family:'Segoe UI',sans-serif;margin:40px;color:#333;max-width:1200px;margin:auto;}}
            h1{{color:{Palette.PRIMARY};border-bottom:3px solid {Palette.PRIMARY};padding-bottom:10px;}}
            h2{{color:{Palette.PRIMARYDARK};margin-top:30px;border-left:4px solid {Palette.PRIMARY};padding-left:10px;}}
            table{{border-collapse:collapse;width:100%;margin:10px 0;}}
            th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}
            th{{background-color:{Palette.PRIMARYLIGHT};}}
            tr:nth-child(even){{background-color:#f9f9f9;}}
            .footer{{margin-top:50px;font-size:12px;color:#999;text-align:center;border-top:1px solid #E5E7EB;padding-top:20px;}}
            .workflow-section{{background:#F9FAFB;padding:15px;border-radius:8px;margin:10px 0;}}
            .highlight{{background:#EEF2FF;padding:8px;border-radius:4px;}}
        </style></head>
        <body>
            <h1>{APPNAME} {APPVERSION} - Full Report</h1>
            <p><b>File:</b> {self.filename}<br><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br><b>Developer:</b> {DEVELOPER} | {EMAIL}</p>

            <h2>Data Summary</h2>
            <table>
                <tr><td><b>Rows</b></td><td>{len(self.df)}</td></tr>
                <tr><td><b>Cols</b></td><td>{len(self.df.columns)}</td></tr>
                <tr><td><b>Numeric</b></td><td>{len(self.df.select_dtypes(include=[np.number]).columns)}</td></tr>
                <tr><td><b>Zero Strategy</b></td><td>{self.zero_strategy.name() if self.zero_strategy else 'None'}</td></tr>
            </table>"""

        # Anomaly Detection Results
        if self.anomaly_results is not None:
            n_anomalies = np.sum(self.anomaly_results['labels'] == -1)
            n_normals = np.sum(self.anomaly_results['labels'] == 1)
            html += f"""
            <h2>🔬 Anomaly Detection Results</h2>
            <div class="workflow-section">
                <table>
                    <tr><td><b>Method</b></td><td>{self.anomaly_method.currentText()}</td></tr>
                    <tr><td><b>Anomalies</b></td><td>{n_anomalies} ({n_anomalies/(n_anomalies+n_normals)*100:.1f}%)</td></tr>
                    <tr><td><b>Normal</b></td><td>{n_normals} ({n_normals/(n_anomalies+n_normals)*100:.1f}%)</td></tr>
                </table>
            </div>"""

        html += f"""
            <h2>Statistics</h2>
            {stats_df.to_html(index=False) if not stats_df.empty else '<p>No data</p>'}
            
            <h2>Analysis</h2>
            <pre>{self.last_analysis_report or 'Not performed'}</pre>
            
            <h2>Geochemistry</h2>
            {self.dict_to_html('Alteration', self.last_geochem_data.get('alterationindices', {}))}
            {self.dict_to_html('Ratios', self.last_geochem_data.get('elementratios', {}))}
            {self.dict_to_html('Grades', self.last_geochem_data.get('gradestats', {}), True)}
            
            <h2>Clustering</h2>
            <p>{f"Method: {self.last_cluster_data.get('method', 'N/A')}, Clusters: {self.last_cluster_data.get('n_clusters', 'N/A')}" if self.last_cluster_data else 'Not performed'}</p>
            
            <h2>SVM</h2>
            {f"<p><b>Accuracy:</b> {self.last_svm_data.get('accuracy', 'N/A'):.2%}</p><pre>{self.last_svm_data.get('report', '')}</pre>" if self.last_svm_data else '<p>Not performed</p>'}
            
            <div class='footer'>
                <p>{APPNAME} {APPVERSION} | {DEVELOPER} | {EMAIL}</p>
                <p>Generated using Coordinated Workflow with Isolation Forest Anomaly Detection</p>
            </div>
        </body></html>"""

        d = QDialog(self)
        d.setWindowTitle("Full Report")
        d.resize(900, 700)
        l = QVBoxLayout(d)
        v = QTextEdit()
        v.setHtml(html)
        v.setReadOnly(True)
        l.addWidget(v)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        btns.rejected.connect(d.reject)
        btns.button(QDialogButtonBox.Save).clicked.connect(lambda: self.save_report(html))
        l.addWidget(btns)
        d.exec_()

    def save_report(self, html):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save",
            f"report_{os.path.splitext(self.filename)[0]}.html" if self.filename else "report.html",
            "HTML (*.html)"
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.log(f"Report saved: {path}")

    def clear_log(self):
        self.log_text.clear()
        self.log("Log cleared")

    def export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "log.txt", "Text (*.txt)")
        if path:
            with open(path, 'w') as f:
                f.write(self.log_text.toPlainText())
            self.log(f"Log exported: {path}")

    def show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle(f"About {APPNAME}")
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"""
<b>{APPNAME} {APPVERSION}</b><br><br>
A comprehensive geochemical data analysis platform with Isolation Forest anomaly detection and coordinated workflow management.<br><br>

<b>Developer:</b> {DEVELOPER}<br>
<b>Email:</b> {EMAIL}<br>
<b>GitHub:</b> <a href='{GITHUB_URL}'>github.com/almouftygeo/elemental-society-analysis</a><br><br>

<b>Workflow Features:</b><br>
• Coordinated Workflow Management<br>
• Isolation Forest & LOF Anomaly Detection<br>
• Zero Replacement Strategies (Constant, Half-DL, Multiplicative)<br>
• CLR Enrichment Interpretation (exp(CLR))<br>
• CLR/ALR/ILR Closure Correction<br>
• Comprehensive Reporting<br><br>

<b>Analysis Features:</b><br>
• PCA with Loading Vectors<br>
• K-Means & DBSCAN Clustering<br>
• SVM Classification<br>
• Geochemistry Indices<br>
• Element Ratios<br><br>

<b>Tip:</b> Use the Anomaly Detection tab to find outliers in your geochemical data using Isolation Forest.
        """)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def get_numeric_cols(self):
        if self.df is None:
            return []
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def get_stylesheet(self):
        P = Palette
        return f"""
        QMainWindow{{background-color:{P.BG};}}
        QWidget{{font-family:"Segoe UI";font-size:12px;color:{P.TEXTPRIMARY};}}
        #Sidebar{{background-color:{P.SIDEBARBG};color:{P.SIDEBARTEXT};}}
        #BrandLabel{{color:{P.SIDEBARTEXTACTIVE};font-size:20px;font-weight:bold;padding:20px;}}
        QListWidget{{background:transparent;border:none;}}
        QListWidget::item{{padding:12px 20px;color:{P.SIDEBARTEXT};}}
        QListWidget::item:selected{{background-color:{P.PRIMARY};color:white;border-radius:4px;}}
        #Toolbar{{background-color:{P.SURFACE};padding:8px;border-bottom:1px solid {P.BORDER};}}
        #StatusPanel{{background-color:{P.PRIMARYLIGHT};color:{P.PRIMARY};padding:8px;font-weight:bold;}}
        QPushButton{{padding:8px 16px;border-radius:6px;border:1px solid {P.BORDER};background:{P.SURFACE};}}
        QPushButton[kind="primary"]{{background:{P.PRIMARY};color:white;border:none;}}
        QPushButton[kind="warning"]{{background:{P.WARNING};color:white;border:none;}}
        QPushButton[kind="success"]{{background:{P.SUCCESS};color:white;border:none;}}
        QPushButton[kind="danger"]{{background:{P.DANGER};color:white;border:none;}}
        QPushButton[kind="info"]{{background:{P.INFO};color:white;border:none;}}
        QPushButton:hover{{background:{P.PRIMARYLIGHT};}}
        QTableWidget{{background:white;gridline-color:#eee;border:1px solid {P.BORDER};}}
        QHeaderView::section{{background:#f9fafb;padding:6px;border:1px solid #eee;font-weight:bold;}}
        QGroupBox{{font-weight:bold;border:1px solid {P.BORDER};border-radius:6px;margin-top:10px;padding-top:10px;}}
        QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 5px;}}
        QTabWidget::pane{{border:1px solid {P.BORDER};border-radius:4px;}}
        QTabBar::tab{{padding:8px 16px;margin-right:2px;border:1px solid {P.BORDER};border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}}
        QTabBar::tab:selected{{background:{P.PRIMARYLIGHT};border-bottom:none;}}
        QTabBar::tab:hover{{background:{P.PRIMARYLIGHT};}}
        """


# =====================================================================
# DIALOG CLASSES
# =====================================================================
class PlotConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Visualizations")
        layout = QVBoxLayout(self)
        self.cbs = {}
        for name in ["Histogram", "PDF KDE", "CDF", "Box Plot", "Q-Q Plot"]:
            cb = QCheckBox(name)
            cb.setChecked(name in ["Histogram", "PDF KDE"])
            self.cbs[name] = cb
            layout.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_selection(self):
        return [k for k, v in self.cbs.items() if v.isChecked()]


class DataFilterDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Column:"))
        self.col_combo = QComboBox()
        self.col_combo.addItems(columns)
        layout.addWidget(self.col_combo)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Threshold", "Outliers (IQR)", "Percentile"])
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.type_combo)
        self.val_input = QDoubleSpinBox()
        self.val_input.setRange(-999999, 999999)
        self.val_input.setValue(95)
        layout.addWidget(QLabel("Value:"))
        layout.addWidget(self.val_input)
        self.op_combo = QComboBox()
        self.op_combo.addItems(["<", ">", "=="])
        layout.addWidget(QLabel("Operator:"))
        layout.addWidget(self.op_combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_filter_config(self):
        return self.col_combo.currentText(), self.type_combo.currentText(), self.val_input.value(), self.op_combo.currentText()


class AdvancedCleanDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Data Cleaning")
        layout = QVBoxLayout(self)
        self.cbs = {}
        for name, label in [
            ("remove_spaces", "Remove leading/trailing spaces"),
            ("remove_special", "Remove special characters"),
            ("standardize_delim", "Standardize delimiters (comma/period)"),
            ("drop_null_rows", "Drop rows with null values"),
            ("drop_null_cols", "Drop columns with >50% null values"),
            ("fill_numeric", "Fill numeric nulls with median"),
            ("fill_text", "Fill text nulls with 'Unknown'")
        ]:
            cb = QCheckBox(label)
            cb.setChecked(name in ["remove_spaces", "remove_special", "standardize_delim"])
            self.cbs[name] = cb
            layout.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class DeleteDataDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete Data Options")
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        t1 = QWidget()
        t1l = QVBoxLayout()
        self.rb_rows_range = QRadioButton("Delete by range (e.g., 0-10)")
        self.rb_rows_condition = QRadioButton("Delete by condition")
        self.rb_rows_all = QRadioButton("Delete all rows")
        self.rb_rows_range.setChecked(True)
        for rb in [self.rb_rows_range, self.rb_rows_condition, self.rb_rows_all]:
            t1l.addWidget(rb)
        self.row_range = QLineEdit("0-10")
        t1l.addWidget(self.row_range)
        self.row_cond_col = QComboBox()
        self.row_cond_col.addItems(df.columns.tolist())
        self.row_cond_val = QLineEdit()
        t1l.addWidget(QLabel("Column:"))
        t1l.addWidget(self.row_cond_col)
        t1l.addWidget(QLabel("Value to match:"))
        t1l.addWidget(self.row_cond_val)
        t1.setLayout(t1l)

        t2 = QWidget()
        t2l = QVBoxLayout()
        t2l.addWidget(QLabel("Select columns to delete (Ctrl+click for multiple):"))
        self.col_list = QListWidget()
        self.col_list.addItems(df.columns.tolist())
        self.col_list.setSelectionMode(QAbstractItemView.MultiSelection)
        t2l.addWidget(self.col_list)
        t2.setLayout(t2l)

        t3 = QWidget()
        t3l = QVBoxLayout()
        self.cb_remove_from_app = QCheckBox("⚠️ Completely remove all data from application")
        t3l.addWidget(self.cb_remove_from_app)
        t3l.addWidget(QLabel("This will clear all views, analyses, and results."))
        t3.setLayout(t3l)

        self.tabs.addTab(t1, "Rows")
        self.tabs.addTab(t2, "Columns")
        self.tabs.addTab(t3, "Global")
        layout.addWidget(self.tabs)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class DisplayControlDialog(QDialog):
    def __init__(self, total_rows, total_cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Control")
        layout = QVBoxLayout(self)
        g = QGridLayout()
        g.addWidget(QLabel("Rows to display:"), 0, 0)
        self.sb_rows = QSpinBox()
        self.sb_rows.setRange(10, max(10, total_rows))
        self.sb_rows.setValue(min(100, total_rows))
        g.addWidget(self.sb_rows, 0, 1)
        g.addWidget(QLabel("Columns to display:"), 1, 0)
        self.sb_cols = QSpinBox()
        self.sb_cols.setRange(1, max(1, total_cols))
        self.sb_cols.setValue(min(20, total_cols))
        g.addWidget(self.sb_cols, 1, 1)
        self.cb_full = QCheckBox("Show all data (may be slow for large datasets)")
        g.addWidget(self.cb_full, 2, 0, 1, 2)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class IndexColumnDialog(QDialog):
    def __init__(self, columns, current_index=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Index & Active Columns")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Set Index Column (preserved during numeric conversion):"))
        self.index_combo = QComboBox()
        self.index_combo.addItem("None")
        self.index_combo.addItems(columns)
        if current_index:
            idx = self.index_combo.findText(current_index)
            if idx >= 0:
                self.index_combo.setCurrentIndex(idx)
        layout.addWidget(self.index_combo)
        layout.addWidget(QLabel("Active Columns (included in analysis):"))
        self.active_list = QListWidget()
        self.active_list.addItems(columns)
        self.active_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(self.active_list.count()):
            self.active_list.item(i).setSelected(True)
        layout.addWidget(self.active_list)
        bl = QHBoxLayout()
        sa = QPushButton("Select All")
        sa.clicked.connect(lambda: [self.active_list.item(i).setSelected(True) for i in range(self.active_list.count())])
        da = QPushButton("Deselect All")
        da.clicked.connect(lambda: [self.active_list.item(i).setSelected(False) for i in range(self.active_list.count())])
        bl.addWidget(sa)
        bl.addWidget(da)
        layout.addLayout(bl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class ElementRatioDialog(QDialog):
    def __init__(self, numeric_cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Element Ratio")
        layout = QVBoxLayout(self)
        g = QGridLayout()
        g.addWidget(QLabel("Numerator:"), 0, 0)
        self.num_combo = QComboBox()
        self.num_combo.addItems(numeric_cols)
        g.addWidget(self.num_combo, 0, 1)
        g.addWidget(QLabel("Denominator:"), 1, 0)
        self.den_combo = QComboBox()
        self.den_combo.addItems(numeric_cols)
        g.addWidget(self.den_combo, 1, 1)
        g.addWidget(QLabel("New Column Name:"), 2, 0)
        self.name_input = QLineEdit("Ratio_1")
        g.addWidget(self.name_input, 2, 1)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class SVMDialog(QDialog):
    def __init__(self, df_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SVM Classification Settings")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Target Column (categorical with few unique values):"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(df_columns)
        layout.addWidget(self.target_combo)
        layout.addWidget(QLabel("Feature Columns (select NUMERIC columns, Ctrl+click for multiple):"))
        self.feature_list = QListWidget()
        self.feature_list.addItems(df_columns)
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.feature_list)
        g = QGridLayout()
        g.addWidget(QLabel("Kernel:"), 0, 0)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["linear", "poly", "rbf", "sigmoid"])
        self.kernel_combo.currentTextChanged.connect(self.toggle_degree)
        g.addWidget(self.kernel_combo, 0, 1)
        g.addWidget(QLabel("C (Regularization):"), 1, 0)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.01, 1000)
        self.c_spin.setValue(1.0)
        g.addWidget(self.c_spin, 1, 1)
        self.lbl_degree = QLabel("Degree (for poly kernel):")
        g.addWidget(self.lbl_degree, 2, 0)
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(2, 10)
        self.degree_spin.setValue(3)
        g.addWidget(self.degree_spin, 2, 1)
        g.addWidget(QLabel("Test Size:"), 3, 0)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.3)
        g.addWidget(self.test_size, 3, 1)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def toggle_degree(self, kernel):
        is_poly = kernel == "poly"
        self.lbl_degree.setVisible(is_poly)
        self.degree_spin.setVisible(is_poly)

    def get_params(self):
        target = self.target_combo.currentText()
        features = [item.text() for item in self.feature_list.selectedItems()]
        if target in features:
            features.remove(target)
        return {
            'target': target,
            'features': features,
            'kernel': self.kernel_combo.currentText(),
            'C': self.c_spin.value(),
            'degree': self.degree_spin.value(),
            'test_size': self.test_size.value()
        }


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APPNAME)
    w = ElementalSocietyAnalysis()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()