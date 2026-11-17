import json
from pathlib import Path

import pytest
import yaml

from elemental_society_analysis.core.run_maleg_pipeline import run


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    # Use tmp paths so the test doesn't touch real data
    config = {
        "ingestion": {"decimal": ","},
        "storage": {"raw_db_path": str(tmp_path / "raw.db")},
        "quality": {"sensor_limits": {"TEMP-01": [80, 180]}},
        "inputs": {
            "sensor_jsons": [],
            "lab_csv": str(tmp_path / "lab.csv"),
            "event_jsons": [],
        },
    }
    (tmp_path / "lab.csv").write_text(
        "sample_id;sampling_point_id;calcium;silica;chloride;sample_time\n"
        "S001;SP-01;450;180;1200;2026-10-30T09:00:00Z\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run(str(config_path))
    assert summary["counts"]["lab"] == 3  # one LabResult per analyte
    assert summary["counts"]["sensor"] == 0
    assert summary["counts"]["event"] == 0
