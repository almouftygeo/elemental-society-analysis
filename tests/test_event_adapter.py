import json

import pytest

from elemental_society_analysis.ingestion.event_adapter import EventAdapter


def test_valid_event_json_is_parsed():
    payload = {
        "event_id": "EV001",
        "equipment": "PIPE-01",
        "type": "ANTISCALANT_DOSING",
        "time": "2026-11-12T10:15:00Z",
        "dose_ml": 25.0,
    }
    event = EventAdapter().parse(payload)

    assert event.event_id == "EV001"
    assert event.equipment_id == "PIPE-01"
    assert event.event_type == "ANTISCALANT_DOSING"
    assert event.parameters["dose_ml"] == 25.0
    assert event.event_time.tzinfo is not None


def test_missing_field_raises_error():
    payload = {
        "event_id": "EV001",
        "equipment": "PIPE-01",
        "type": "ANTISCALANT_DOSING",
        # "time" is missing
    }
    with pytest.raises(ValueError, match="Missing required fields"):
        EventAdapter().parse(payload)


def test_string_json_is_also_accepted():
    raw = json.dumps({
        "event_id": "EV002",
        "equipment": "SEP-01",
        "type": "MAINTENANCE",
        "time": "2026-11-12T11:00:00Z",
    })
    event = EventAdapter().parse(raw)
    assert event.event_type == "MAINTENANCE"


def test_naive_timestamp_raises_error():
    payload = {
        "event_id": "EV003",
        "equipment": "PIPE-01",
        "type": "CALIBRATION",
        "time": "2026-11-12T12:00:00",  # no timezone
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        EventAdapter().parse(payload)
