"""Read geochemical CSVs using the project config."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import IOConfig

logger = logging.getLogger(__name__)

# Matches strings like "<0,3", "<10", ">5" (detection limits).
_DETECTION_LIMIT_RE = re.compile(r"^\s*[<>]=?\s*([0-9]+(?:[.,][0-9]+)?)\s*$")


def _clean_numeric_cell(value: object, decimal: str = ",") -> float:
    """Convert a raw cell to float, handling '<x' detection limits."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "NONE", "-"}:
        return np.nan

    match = _DETECTION_LIMIT_RE.match(text)
    if match:
        limit = match.group(1).replace(decimal, ".")
        # Convention: replace <DL with 2/3 of the detection limit
        return (2.0 / 3.0) * float(limit)

    # European decimal formatting
    text = text.replace(decimal, ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_dataframe(config: IOConfig, path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV and coerce numeric columns using the IO config."""

def load_dataframe(config: IOConfig, path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV and coerce numeric columns using the IO config."""
    csv_path = Path(path) if path else Path(config.raw_path)
    logger.info("Loading %s", csv_path)

    df = pd.read_csv(
        csv_path,
        encoding=config.encoding,
        sep=config.separator,
        decimal=config.decimal,
        na_values=["<LOD", "LOD"],
        keep_default_na=True,
    )

    # Drop footer/blank rows (rows without an ID are not samples)
    if "ID" in df.columns:
        before = len(df)
        df = df[df["ID"].notna()].reset_index(drop=True)
        if before - len(df):
            logger.info("Dropped %d row(s) with missing ID", before - len(df))

    # Resolve duplicate column names in one pass.
    # The raw CSV declares some headers twice (K_pct, Th_ppm).
    # pandas preserves both with the same label, which breaks
    # `df["K_pct"]` and `coverage["K_pct"]` later.
    # Strategy: rename the Nth occurrence of X to X__dupN, then
    # drop it if it is an exact copy of X on the shared non-NaN rows.
    seen: dict[str, int] = {}
    renamed: list[str] = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            renamed.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            renamed.append(c)
    df.columns = renamed

    _dup_re = re.compile(r"^(?P<base>.+?)(?:\.\d+|__dup\d+)$")
    for col in list(df.columns):
        m = _dup_re.match(col)
        if not m:
            continue
        base = m.group("base")
        if base not in df.columns:
            continue
        shared = df[[base, col]].dropna()
        if len(shared) and shared[base].equals(shared[col]):
            logger.info("Dropping exact duplicate column %s (== %s)", col, base)
            df = df.drop(columns=[col])
        else:
            logger.warning(
                "Column %s differs from %s on shared rows; keeping both",
                col, base,
            )

    id_cols = set(config.id_columns)
    coord_cols = set(config.coord_columns)

    for col in df.columns:
        if col in id_cols or col in coord_cols:
            continue
        df[col] = df[col].map(lambda v: _clean_numeric_cell(v, config.decimal))

    # Coordinates: also numeric, but keep them for spatial work
    for col in coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded shape=%s numeric=%d", df.shape,
                df.select_dtypes(include=[np.number]).shape[1])
    return df
    return df


def detect_oxide_columns(df: pd.DataFrame) -> list[str]:
    """Return oxide columns in wt% (ending in _pct) that look like major oxides."""
    candidates = [c for c in df.columns if c.endswith("_pct")]
    # Drop duplicate columns like K_pct.1 if present
    seen: dict[str, str] = {}
    unique: list[str] = []
    for col in candidates:
        base = col.split(".")[0]
        if base in seen:
            continue
        seen[base] = col
        unique.append(col)
    logger.info("Detected %d oxide columns: %s", len(unique), unique)
    return unique
