"""Read geochemical CSVs using the project config."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import IOConfig

logger = logging.getLogger(__name__)

# Matches strings like "<0,3", "<10", ">5", "<=0.5".
_DETECTION_LIMIT_RE = re.compile(r"^\s*[<>]=?\s*([0-9]+(?:[.,][0-9]+)?)\s*$")


def _clean_numeric_cell(
    value: object,
    decimal: str = ",",
) -> tuple[float, str | None, float]:
    """Convert a raw cell to (value, censor_flag, detection_limit).

    Censored values ('<0.5', '>10') return NaN for the value — the
    substitution (half-DL, multiplicative, etc.) happens downstream in
    the geochemical layer, where the strategy is chosen and recorded.
    """
    if pd.isna(value):
        return np.nan, None, np.nan
    if isinstance(value, (int, float)):
        return float(value), None, np.nan

    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "NONE", "-"}:
        return np.nan, None, np.nan

    match = _DETECTION_LIMIT_RE.match(text)
    if match:
        limit = float(match.group(1).replace(decimal, "."))
        flag = "BELOW_DL" if text.lstrip().startswith("<") else "ABOVE_DL"
        return np.nan, flag, limit

    normalized = text.replace(decimal, ".")
    try:
        return float(normalized), None, np.nan
    except ValueError:
        return np.nan, None, np.nan


def load_dataframe(config: IOConfig, path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV and coerce numeric columns using the IO config.

    Returns a DataFrame whose ``.attrs`` carries:
        ``censor_flags``      DataFrame of 'BELOW_DL' / 'ABOVE_DL' / None per cell
        ``detection_limits``  DataFrame of DL values per cell
        ``provenance``        dict with source file, row count, timestamp
    """
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
                col,
                base,
            )

    id_cols = set(config.id_columns)
    coord_cols = set(config.coord_columns)

    # Initialize the per-cell metadata BEFORE the loop.
    censor_flags = pd.DataFrame(index=df.index)
    detection_limits = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in id_cols or col in coord_cols:
            continue
        triplets = df[col].map(lambda v: _clean_numeric_cell(v, config.decimal))
        df[col] = triplets.map(lambda t: t[0])
        censor_flags[f"{col}_censor"] = triplets.map(lambda t: t[1])
        detection_limits[f"{col}_dl"] = triplets.map(lambda t: t[2])

    # Coordinates: also numeric, but keep them for spatial work
    for col in coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Loaded shape=%s numeric=%d",
        df.shape,
        df.select_dtypes(include=[np.number]).shape[1],
    )

    # Attach metadata so downstream code can access it without
    # changing the return type (kept as DataFrame for compatibility).
    df.attrs["censor_flags"] = censor_flags
    df.attrs["detection_limits"] = detection_limits
    df.attrs["provenance"] = {
        "source_file": str(csv_path),
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    return df


def detect_oxide_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted oxide columns in wt% (ending in _pct)."""
    candidates = [c for c in df.columns if c.endswith("_pct")]
    # Deduplicate any "X_pct" and "X_pct__dupN" pairs by base name.
    seen: dict[str, str] = {}
    for col in candidates:
        base = col.split(".")[0].split("__dup")[0]
        if base not in seen:
            seen[base] = col
    result = sorted(seen.values())
    logger.info("Detected %d oxide columns: %s", len(result), result)
    return result
