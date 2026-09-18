"""Shared pytest fixtures for elemental-society-analysis."""
import sys
from pathlib import Path

# Make src/ importable during tests
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numeric_data():
    """Random numeric DataFrame for anomaly detector tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(100, 5)),
        columns=[f"c{i}" for i in range(5)],
    )


@pytest.fixture
def sample_composition():
    """Small geochemical composition table (5 oxides, 3 samples)."""
    return pd.DataFrame({
        "SiO2":  [65.2, 54.1, 72.0],
        "Al2O3": [15.1, 17.8, 12.4],
        "Fe2O3": [4.2, 8.5, 2.1],
        "MgO":   [2.1, 5.2, 0.8],
        "CaO":   [3.5, 7.1, 1.2],
    })

