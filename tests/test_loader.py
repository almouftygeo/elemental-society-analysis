"""Tests for the geochemical loader (current signature)."""

from pathlib import Path

import numpy as np
import pandas as pd

from elemental_society_analysis.geochem.loader import (
    _clean_numeric_cell,
    detect_oxide_columns,
    load_dataframe,
)


def test_clean_numeric_cell_plain_number():
    value, _, _ = _clean_numeric_cell("65.2")
    assert value == 65.2


def test_clean_numeric_cell_eu_decimal():
    value, _, _ = _clean_numeric_cell("65,2", decimal=",")
    assert value == 65.2


def test_clean_numeric_cell_below_dl():
    """Censored values return NaN + a flag + the DL — never an invented value.

    Substitution (half-DL, multiplicative, constant) happens downstream in
    the geochemical layer, where the strategy is chosen and recorded.
    """
    value, flag, dl = _clean_numeric_cell("<0.5")
    assert np.isnan(value)
    assert flag == "BELOW_DL"
    assert dl == 0.5


def test_clean_numeric_cell_above_dl():
    """Same contract for '>X' — NaN + ABOVE_DL flag + the DL."""
    value, flag, dl = _clean_numeric_cell(">10")
    assert np.isnan(value)
    assert flag == "ABOVE_DL"
    assert dl == 10.0


def test_clean_numeric_cell_na_variants():
    for raw in [None, np.nan, "", "NA", "null", "-"]:
        value, _, _ = _clean_numeric_cell(raw)
        assert np.isnan(value)


def test_load_dataframe_drops_missing_id_and_dedups(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "ID,SiO2_pct,CaO_pct,SiO2_pct\n"
        "S1,65.2,3.5,65.2\n"
        "S2,54.1,7.1,54.1\n"
        ",,,\n",  # footer row with missing ID
        encoding="utf-8",
    )

    class _Cfg:
        raw_path = str(csv)
        encoding = "utf-8"
        separator = ","
        decimal = "."
        id_columns = ("ID",)
        coord_columns = ()

    df = load_dataframe(_Cfg())

    # Footer dropped
    assert len(df) == 2
    # Duplicate column resolved (only one SiO2_pct remains)
    assert list(df.columns).count("SiO2_pct") == 1
    # Numeric conversion happened
    assert df["SiO2_pct"].dtype.kind == "f"


def test_detect_oxide_columns():
    df = pd.DataFrame(columns=["ID", "SiO2_pct", "Al2O3_pct", "K_pct__dup1", "CaO_ppm"])
    assert detect_oxide_columns(df) == ["Al2O3_pct", "SiO2_pct"]
