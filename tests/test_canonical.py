from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta

import pytest

from elemental_society_analysis.ingestion.canonical import (
    CensorFlag,
    LabResult,
    OperationalEvent,
    QualityStatus,
    SensorRecord,
)


UTC = timezone.utc

EVENT_TIME = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
INGESTED_AT = datetime(2026, 11, 12, 10, 1, tzinfo=UTC)


def make_sensor(**overrides):
    data = {
        "measurement_id": "M001",
        "sensor_id": "TEMP-01",
        "equipment_id": "PIPE-01",
        "value": 145.2,
        "unit": "C",
        "event_time": EVENT_TIME,
        "ingested_at": INGESTED_AT,
        "source_system": "PLC_SCADA",
        "raw_payload": {
            "tag": "TEMP-01",
            "value": 145.2,
            "unit": "C",
            "metadata": {"location": "separator"},
        },
    }
    data.update(overrides)
    return SensorRecord(**data)


def make_lab(**overrides):
    data = {
        "result_id": "R001",
        "sample_id": "S001",
        "sampling_point_id": "SP-01",
        "analyte": "calcium",
        "raw_string": "450",
        "value": 450.0,
        "unit": "mg/L",
        "censor_flag": CensorFlag.NONE,
        "detection_limit": None,
        "sample_time": EVENT_TIME,
        "result_available_at": INGESTED_AT,
        "ingested_at": INGESTED_AT,
        "quality_status": QualityStatus.VALID,
        "quality_reason": None,
        "source_system": "LABORATORY",
        "raw_payload": {"sample_id": "S001", "calcium": "450"},
    }
    data.update(overrides)
    return LabResult(**data)


def test_sensor_record_is_created():
    record = make_sensor()

    assert record.sensor_id == "TEMP-01"
    assert record.value == 145.2
    assert record.unit == "C"


def test_sensor_timestamps_are_normalized_to_utc():
    offset_time = datetime(
        2026, 11, 12, 11, 0,
        tzinfo=timezone(timedelta(hours=1)),
    )

    record = make_sensor(event_time=offset_time)

    assert record.event_time == EVENT_TIME
    assert record.event_time.utcoffset() == timedelta(0)


def test_naive_timestamp_is_rejected():
    naive_time = datetime(2026, 11, 12, 10, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        make_sensor(event_time=naive_time)


def test_sensor_record_is_immutable():
    record = make_sensor()

    with pytest.raises(FrozenInstanceError):
        record.value = 200.0


def test_raw_payload_is_deeply_immutable():
    record = make_sensor()

    with pytest.raises(TypeError):
        record.raw_payload["value"] = 200.0

    with pytest.raises(TypeError):
        record.raw_payload["metadata"]["location"] = "other"

    assert record.raw_payload["value"] == 145.2


def test_below_detection_limit_is_preserved():
    result = make_lab(
        raw_string="<0.5",
        value=None,
        censor_flag=CensorFlag.BELOW_DL,
        detection_limit=0.5,
    )

    assert result.value is None
    assert result.censor_flag == CensorFlag.BELOW_DL
    assert result.detection_limit == 0.5
    assert result.raw_string == "<0.5"


def test_censored_result_requires_detection_limit():
    with pytest.raises(ValueError, match="requires a detection limit"):
        make_lab(
            raw_string="<0.5",
            value=None,
            censor_flag=CensorFlag.BELOW_DL,
            detection_limit=None,
        )


def test_uncensored_result_requires_numeric_value():
    with pytest.raises(ValueError, match="must have a numeric value"):
        make_lab(
            raw_string="",
            value=None,
            censor_flag=CensorFlag.NONE,
        )


def test_operational_event_is_created():
    event = OperationalEvent(
        event_id="EV001",
        equipment_id="PIPE-01",
        event_type="ANTISCALANT_DOSING",
        event_time=EVENT_TIME,
        ingested_at=INGESTED_AT,
        parameters={"dose_ml": 25.0},
        source_system="OPERATOR_LOG",
        raw_payload={"dose_ml": 25.0},
    )

    assert event.event_type == "ANTISCALANT_DOSING"
    assert event.parameters["dose_ml"] == 25.0
