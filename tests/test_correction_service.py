"""
tests/test_correction_service.py
Unit tests for the CorrectionService (Half-DL, Constant replacement)
"""

import pytest
import pandas as pd
from src.services.correction_service import (
    CorrectionService,
    HalfDetectionLimit,
    ConstantReplacement,
)


class TestHalfDetectionLimit:
    """Test Half-DL zero replacement strategy."""

    def test_replace_zeros(self):
        """Test that zeros are replaced with DL/2."""
        dl_map = {"SIO2": 0.1, "FE2O3": 0.01}
        strategy = HalfDetectionLimit(dl_map)

        data = pd.DataFrame(
            {
                "SiO2": [72.0, 0.0, 65.0],
                "Fe2O3": [2.0, 0.0, 1.5],
                "CaO": [2.0, 1.5, 0.0],
            }
        )

        result = strategy.replace(data)

        assert result.iloc[1, 0] == 0.05  # SiO2 DL/2
        assert result.iloc[1, 1] == 0.005  # Fe2O3 DL/2
        assert result.iloc[2, 2] == 0.0  # No DL for CaO

    def test_case_insensitive_lookup(self):
        """Test that DL lookup is case-insensitive."""
        dl_map = {"sio2": 0.1, "fe2o3": 0.01}
        strategy = HalfDetectionLimit(dl_map)

        data = pd.DataFrame({"SiO2": [0.0, 72.0], "Fe2O3": [2.0, 0.0]})

        result = strategy.replace(data)

        assert result.iloc[0, 0] == 0.05
        assert result.iloc[1, 1] == 0.005

    def test_non_zero_values_unchanged(self):
        """Test that non-zero values are not modified."""
        dl_map = {"SIO2": 0.1, "FE2O3": 0.01}
        strategy = HalfDetectionLimit(dl_map)

        original = pd.DataFrame({"SiO2": [72.0, 65.0, 68.0], "Fe2O3": [2.0, 1.5, 0.8]})

        result = strategy.replace(original.copy())
        pd.testing.assert_frame_equal(original, result)


class TestConstantReplacement:
    """Test Constant zero replacement strategy."""

    def test_replace_zeros_with_constant(self):
        """Test that zeros are replaced with a constant value."""
        strategy = ConstantReplacement(value=1e-6)

        data = pd.DataFrame(
            {
                "SiO2": [72.0, 0.0, 65.0],
                "Fe2O3": [2.0, 0.0, 1.5],
                "CaO": [0.0, 1.5, 0.0],
            }
        )

        result = strategy.replace(data)

        assert result.iloc[1, 0] == 1e-6
        assert result.iloc[1, 1] == 1e-6
        assert result.iloc[0, 2] == 1e-6
        assert result.iloc[2, 2] == 1e-6
        assert result.iloc[0, 0] == 72.0
        assert result.iloc[0, 1] == 2.0

    def test_raises_error_for_negative_value(self):
        """Test that ConstantReplacement raises error for value <= 0."""
        with pytest.raises(ValueError, match="Replacement must be > 0"):
            ConstantReplacement(value=0)

        with pytest.raises(ValueError, match="Replacement must be > 0"):
            ConstantReplacement(value=-1e-6)


class TestCorrectionService:
    """Test the CorrectionService that orchestrates zero replacement."""

    def test_prepare_with_zero_strategy(self):
        """Test prepare() with a zero strategy."""
        dl_map = {"SIO2": 0.1}
        strategy = HalfDetectionLimit(dl_map)
        service = CorrectionService(zero_strategy=strategy)

        data = pd.DataFrame({"SiO2": [72.0, 0.0, 65.0], "Fe2O3": [2.0, 1.0, 0.0]})

        class MockDataset:
            def composition(self):
                return data

        result = service.prepare(MockDataset())

        assert result.iloc[1, 0] == 0.05
        assert result.iloc[2, 1] == 0.0
        assert result.iloc[0, 0] == 72.0

    def test_prepare_without_zero_strategy(self):
        """Test prepare() without a zero strategy."""
        service = CorrectionService(zero_strategy=None)

        data = pd.DataFrame({"SiO2": [72.0, 0.0, 65.0], "Fe2O3": [2.0, 1.0, 0.0]})

        class MockDataset:
            def composition(self):
                return data

        result = service.prepare(MockDataset())

        assert result.iloc[1, 0] == 1e-10
        assert result.iloc[2, 1] == 1e-10

    def test_raises_error_for_negative_values(self):
        """Test that prepare() raises error for negative values."""
        service = CorrectionService(zero_strategy=None)

        data = pd.DataFrame({"SiO2": [72.0, -5.0, 65.0], "Fe2O3": [2.0, 1.0, 0.0]})

        class MockDataset:
            def composition(self):
                return data

        with pytest.raises(ValueError, match="Negative values present"):
            service.prepare(MockDataset())
