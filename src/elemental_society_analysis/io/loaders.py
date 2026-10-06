"""File-loading helpers for geochemical and spatial datasets."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file after confirming that it exists."""
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path, low_memory=False)
