"""Unit tests for closure correctors."""

import numpy as np
import pandas as pd
import pytest

from elemental_society_analysis.geochem.correctors import (
    ALRCorrector,
    CLRCorrector,
    ILRCorrector,
    classify_relative_value,
)


def test_clr_rows_sum_to_zero(sample_composition):
    clr = CLRCorrector().apply(sample_composition)
    np.testing.assert_allclose(clr.sum(axis=1).values, 0.0, atol=1e-10)


def test_clr_rejects_non_positive():
    bad = pd.DataFrame({"A": [1.0, 0.0], "B": [2.0, 3.0]})
    with pytest.raises(ValueError, match="Non-positive"):
        CLRCorrector().apply(bad)


def test_alr_uses_parameterized_base(sample_composition):
    alr = ALRCorrector(base="SiO2").apply(sample_composition)
    assert "SiO2" not in alr.columns


def test_ilr_has_n_minus_1_columns(sample_composition):
    ilr = ILRCorrector().apply(sample_composition)
    assert ilr.shape == (3, 4)


def test_classify_relative_value():
    assert classify_relative_value(1.10) == "enriched"
    assert classify_relative_value(0.90) == "depleted"
    assert classify_relative_value(1.00) == "near_geometric_mean"
