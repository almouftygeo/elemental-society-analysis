import json
import sqlite3
from datetime import datetime, timezone

import pytest

from elemental_society_analysis.ingestion.canonical import (
    CensorFlag, LabResult, QualityStatus, SensorRecord,
)
from elemental_society_analysis.quality.validator import (
    LabValidator, SensorValidator,
)
from elemental_society_analysis.storage.raw_repository import RawRepository


UTC = timezone.utc
NOW = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)


def make_sensor(value=145.2, sensor_id="TEMP-01"):
    return SensorRecord(
        measurement_id="M001", sensor_id=sensor_id, equipment_id="PIPE-01",
        value=value, unit="C", event_time=NOW, ingested_at=NOW,
        source_system="PLC_SCADA", raw_payload={"tag": sensor_id},
    )


def make_lab(censor=CensorFlag.NONE, value=450.0):
    return LabResult(
        result_id="R001", sample_id="S001", sampling_point_id="SP-01",
        analyte="calcium", raw_string="450", value=value, unit="mg/L",
        censor_flag=censor, detection_limit=0.5 if censor != CensorFlag.NONE else None,
        sample_time=NOW, result_available_at=NOW, ingested_at=NOW,
        quality_status=QualityStatus.VALID, quality_reason=None,
        source_system="LABORATORY", raw_payload={"sample_id": "S001"},
    )


def test_sensor_validator_rejects_below_limit():
    validator = SensorValidator({"TEMP-01": (80, 180)})
    result = validator.validate(make_sensor(value=50.0))
    assert result.status == "REJECTED"
    assert result.reason == "BELOW_SENSOR_LIMIT"


def test_sensor_validator_flags_above_range():
    validator = SensorValidator({"TEMP-01": (80, 180)})
    result = validator.validate(make_sensor(value=200.0))
    assert result.status == "SUSPECT"


def test_lab_validator_preserves_censored():
    result = LabValidator.validate(make_lab(censor=CensorFlag.BELOW_DL, value=None))
    assert result.status == "VALID"


def test_raw_repository_saves_sensor(tmp_path):
    db = tmp_path / "raw.db"
    repo = RawRepository(str(db))
    record = make_sensor()
    payload_id = repo.save_sensor(record)
    assert payload_id == "M001"

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT raw_json FROM raw_sensor_payload WHERE payload_id=?",
            ("M001",),
        ).fetchone()
    assert json.loads(row[0])["tag"] == "TEMP-01"


def test_raw_repository_is_append_only(tmp_path):
    db = tmp_path / "raw.db"
    repo = RawRepository(str(db))
    repo.save_sensor(make_sensor())
    # Saving the same ID again should not duplicate (INSERT OR IGNORE)
    repo.save_sensor(make_sensor())
    with sqlite3.connect(str(db)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM raw_sensor_payload WHERE payload_id='M001'"
        ).fetchone()[0]
    assert count == 1
