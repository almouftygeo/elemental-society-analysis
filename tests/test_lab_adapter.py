import pytest

from elemental_society_analysis.ingestion.lab_adapter import LabCSVAdapter
from elemental_society_analysis.ingestion.canonical import CensorFlag


def _write_csv(tmp_path, content):
    csv_file = tmp_path / "lab.csv"
    csv_file.write_text(content, encoding="utf-8")
    return str(csv_file)


def test_censored_below_dl_is_preserved(tmp_path):
    csv_file = _write_csv(tmp_path,
        "sample_id;sampling_point_id;calcium;silica;chloride;sample_time\n"
        "S001;SP-01;<0.5;180;1200;2026-10-30T09:00:00Z\n")
    results = LabCSVAdapter().parse(csv_file)
    ca = next(r for r in results if r.analyte == "calcium")
    assert ca.value is None
    assert ca.censor_flag == CensorFlag.BELOW_DL
    assert ca.detection_limit == 0.5
    assert ca.raw_string == "<0.5"


def test_normal_value_is_parsed(tmp_path):
    csv_file = _write_csv(tmp_path,
        "sample_id;sampling_point_id;calcium;silica;chloride;sample_time\n"
        "S001;SP-01;450;180;1200;2026-10-30T09:00:00Z\n")
    results = LabCSVAdapter().parse(csv_file)
    ca = next(r for r in results if r.analyte == "calcium")
    assert ca.value == 450.0
    assert ca.censor_flag == CensorFlag.NONE


def test_missing_value_flagged(tmp_path):
    csv_file = _write_csv(tmp_path,
        "sample_id;sampling_point_id;calcium;silica;chloride;sample_time\n"
        "S001;SP-01;NA;180;1200;2026-10-30T09:00:00Z\n")
    results = LabCSVAdapter().parse(csv_file)
    ca = next(r for r in results if r.analyte == "calcium")
    assert ca.value is None
    assert ca.censor_flag == CensorFlag.MISSING


def test_naive_timestamp_raises_error(tmp_path):
    csv_file = _write_csv(tmp_path,
        "sample_id;sampling_point_id;calcium;silica;chloride;sample_time\n"
        "S001;SP-01;450;180;1200;2026-10-30T09:00:00\n")
    with pytest.raises(ValueError, match="timezone-aware"):
        LabCSVAdapter().parse(csv_file)
