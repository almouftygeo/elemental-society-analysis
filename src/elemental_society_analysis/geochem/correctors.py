"""Closure correctors for compositional data (CLR, ALR, ILR)."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _helmert_submatrix(k: int) -> np.ndarray:
    """Helmert submatrix used for the ILR transformation."""
    H = np.zeros((k - 1, k))
    for i in range(1, k):
        H[i - 1, :i] = 1.0 / np.sqrt(i * (i + 1))
        H[i - 1, i] = -i / np.sqrt(i * (i + 1))
    return H


def classify_relative_value(value: float, tolerance: float = 0.05) -> str:
    """Classify exp(CLR) relative to the geometric mean."""
    lower, upper = 1 - tolerance, 1 + tolerance
    if value > upper:
        return "enriched"
    if value < lower:
        return "depleted"
    return "near_geometric_mean"


def format_enrichment_factor(value: float) -> str:
    """Format enrichment factor with a meaningful sign."""
    return f"+{value:.3f}x" if value >= 1 else f"{value:.3f}x"


class ClosureCorrector(ABC):
    """Abstract base for closure correctors."""

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def name(self) -> str:
        return self.__class__.__name__


class CLRCorrector(ClosureCorrector):
    """Centered Log-Ratio transformation with enrichment interpretation."""

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = data.to_numpy(dtype=float)
        if (vals <= 0).any():
            raise ValueError(
                "Non-positive values reached CLR. Zero policy must run BEFORE correction."
            )
        logv = np.log(vals)
        return pd.DataFrame(
            logv - logv.mean(axis=1, keepdims=True),
            index=data.index,
            columns=data.columns,
        )

    def apply_with_enrichment(self, data: pd.DataFrame, tolerance: float = 0.05) -> dict:
        """Apply CLR and return full interpretation dictionary."""
        clr = self.apply(data)
        enrichment = np.exp(clr)

        result_df = enrichment.copy()
        for col in result_df.columns:
            result_df[f"interpretation_{col}"] = result_df[col].apply(
                lambda x: classify_relative_value(x, tolerance=tolerance)
            )
            result_df[f"clr_{col}"] = clr[col]
            result_df[f"factor_{col}"] = result_df[col].apply(format_enrichment_factor)

        sample_interpretations = []
        for idx in clr.index:
            sample = clr.loc[idx]
            classification_counts = {
                "enriched": 0,
                "depleted": 0,
                "near_geometric_mean": 0,
            }
            for col in enrichment.columns:
                v = classify_relative_value(enrichment.loc[idx, col], tolerance=tolerance)
                classification_counts[v] += 1
            sample_interpretations.append(
                {
                    "sample": idx,
                    "enriched_elements": sample[sample > 0].index.tolist(),
                    "depleted_elements": sample[sample < 0].index.tolist(),
                    "most_enriched": sample.idxmax() if not sample.empty else None,
                    "most_depleted": sample.idxmin() if not sample.empty else None,
                    "enrichment_factors": enrichment.loc[idx].to_dict(),
                    "classification_summary": classification_counts,
                }
            )

        summary = {
            "clr_mean": clr.mean().to_dict(),
            "enrichment_mean": enrichment.mean().to_dict(),
            "most_enriched_overall": clr.mean().idxmax(),
            "most_depleted_overall": clr.mean().idxmin(),
            "range_clr": {col: (clr[col].min(), clr[col].max()) for col in clr.columns},
            "overall_classification_counts": {},
        }
        for col in enrichment.columns:
            counts = {"enriched": 0, "depleted": 0, "near_geometric_mean": 0}
            for idx in enrichment.index:
                v = classify_relative_value(enrichment.loc[idx, col], tolerance=tolerance)
                counts[v] += 1
            summary["overall_classification_counts"][col] = counts

        return {
            "clr": clr,
            "enrichment": enrichment,
            "result_with_interpretation": result_df,
            "sample_interpretations": sample_interpretations,
            "summary": summary,
        }


class ALRCorrector(ClosureCorrector):
    """Additive Log-Ratio transformation with parameterizable base."""

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
    """Isometric Log-Ratio transformation (Helmert basis)."""

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        closed = data.div(data.sum(axis=1), axis=0)
        clr = CLRCorrector().apply(closed)
        H = _helmert_submatrix(clr.shape[1])
        return pd.DataFrame(
            clr.to_numpy() @ H.T,
            index=clr.index,
            columns=[f"ilr_{i + 1}" for i in range(H.shape[0])],
        )
