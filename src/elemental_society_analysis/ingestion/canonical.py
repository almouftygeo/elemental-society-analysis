from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping


# ---------------------------------------------------------
# 1. Enumerations
# ---------------------------------------------------------

class CensorFlag(str, Enum):
    NONE = "NONE"
    BELOW_DL = "BELOW_DL"
    ABOVE_DL = "ABOVE_DL"
    MISSING = "MISSING"
    PARSE_ERROR = "PARSE_ERROR"


class QualityStatus(str, Enum):
    VALID = "VALID"
    SUSPECT = "SUSPECT"
    REJECTED = "REJECTED"


# ---------------------------------------------------------
# 2. Shared validation and immutability helpers
# ---------------------------------------------------------

def require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def require_finite(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def require_utc_datetime(value: datetime, field_name: str) -> datetime:
    """
    Reject timezone-naive timestamps.

    Convert timezone-aware timestamps to UTC so that
    downstream comparisons are consistent.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field_name} must be timezone-aware; "
            "naive timestamps are not accepted"
        )

    return value.astimezone(timezone.utc)


def freeze_json(value: Any) -> Any:
    """
    Recursively protect JSON-compatible payload data.

    Dictionaries become read-only mappings.
    Lists become tuples.
    """
    if isinstance(value, dict):
        return MappingProxyType({
            key: freeze_json(item)
            for key, item in value.items()
        })

    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)

    if isinstance(value, tuple):
        return tuple(freeze_json(item) for item in value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported raw payload value: {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """
    Convert protected payload data back into ordinary
    dictionaries and lists for JSON serialization.
    """
    if isinstance(value, Mapping):
        return {
            key: thaw_json(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]

    return value


# ---------------------------------------------------------
# 3. Sensor record
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SensorRecord:
    measurement_id: str
    sensor_id: str
    equipment_id: str

    value: float
    unit: str

    event_time: datetime
    ingested_at: datetime

    source_system: str
    raw_payload: Mapping[str, Any]

    # Optional result of later unit normalization
    normalized_value: float | None = None
    normalized_unit: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "measurement_id",
            "sensor_id",
            "equipment_id",
            "unit",
            "source_system",
        ):
            require_nonempty(
                getattr(self, field_name),
                field_name,
            )

        require_finite(self.value, "value")

        if self.normalized_value is not None:
            require_finite(
                self.normalized_value,
                "normalized_value",
            )

        if (self.normalized_value is None) != (
            self.normalized_unit is None
        ):
            raise ValueError(
                "normalized_value and normalized_unit "
                "must either both be provided or both be None"
            )

        if self.normalized_unit is not None:
            require_nonempty(
                self.normalized_unit,
                "normalized_unit",
            )

        if not isinstance(self.raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        object.__setattr__(
            self,
            "event_time",
            require_utc_datetime(self.event_time, "event_time"),
        )

        object.__setattr__(
            self,
            "ingested_at",
            require_utc_datetime(self.ingested_at, "ingested_at"),
        )

        object.__setattr__(
            self,
            "raw_payload",
            freeze_json(dict(self.raw_payload)),
        )


# ---------------------------------------------------------
# 4. Laboratory result
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LabResult:
    result_id: str
    sample_id: str
    sampling_point_id: str

    analyte: str
    raw_string: str

    value: float | None
    unit: str

    censor_flag: CensorFlag
    detection_limit: float | None

    sample_time: datetime
    result_available_at: datetime
    ingested_at: datetime

    quality_status: QualityStatus
    quality_reason: str | None

    source_system: str
    raw_payload: Mapping[str, Any]

    # Optional link to the equipment/process stream
    equipment_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "result_id",
            "sample_id",
            "sampling_point_id",
            "analyte",
            "unit",
            "source_system",
        ):
            require_nonempty(
                getattr(self, field_name),
                field_name,
            )

        if not isinstance(self.raw_string, str):
            raise TypeError("raw_string must be a string")

        if not isinstance(self.censor_flag, CensorFlag):
            raise TypeError("censor_flag must be a CensorFlag")

        if not isinstance(self.quality_status, QualityStatus):
            raise TypeError(
                "quality_status must be a QualityStatus"
            )

        if self.equipment_id is not None:
            require_nonempty(self.equipment_id, "equipment_id")

        if self.quality_reason is not None:
            require_nonempty(self.quality_reason, "quality_reason")

        if self.value is not None:
            require_finite(self.value, "value")

        if self.detection_limit is not None:
            require_finite(
                self.detection_limit,
                "detection_limit",
            )

            if self.detection_limit <= 0:
                raise ValueError(
                    "detection_limit must be greater than zero"
                )

        if self.censor_flag == CensorFlag.NONE:
            if self.value is None:
                raise ValueError(
                    "An uncensored result must have a numeric value"
                )

        elif self.censor_flag in (
            CensorFlag.BELOW_DL,
            CensorFlag.ABOVE_DL,
        ):
            if self.value is not None:
                raise ValueError(
                    "A censored result must not be represented "
                    "as an exact numeric value"
                )

            if self.detection_limit is None:
                raise ValueError(
                    "A censored result requires a detection limit "
                    "or reporting limit"
                )

        elif self.value is not None:
            raise ValueError(
                "Missing or parse-error results cannot have "
                "a numeric value"
            )

        if not isinstance(self.raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        object.__setattr__(
            self,
            "sample_time",
            require_utc_datetime(self.sample_time, "sample_time"),
        )

        object.__setattr__(
            self,
            "result_available_at",
            require_utc_datetime(
                self.result_available_at,
                "result_available_at",
            ),
        )

        object.__setattr__(
            self,
            "ingested_at",
            require_utc_datetime(self.ingested_at, "ingested_at"),
        )

        object.__setattr__(
            self,
            "raw_payload",
            freeze_json(dict(self.raw_payload)),
        )


# ---------------------------------------------------------
# 5. Operational event
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OperationalEvent:
    event_id: str
    equipment_id: str
    event_type: str

    event_time: datetime
    ingested_at: datetime

    parameters: Mapping[str, Any]
    source_system: str
    raw_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "event_id",
            "equipment_id",
            "event_type",
            "source_system",
        ):
            require_nonempty(
                getattr(self, field_name),
                field_name,
            )

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")

        if not isinstance(self.raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        object.__setattr__(
            self,
            "event_time",
            require_utc_datetime(self.event_time, "event_time"),
        )

        object.__setattr__(
            self,
            "ingested_at",
            require_utc_datetime(self.ingested_at, "ingested_at"),
        )

        object.__setattr__(
            self,
            "parameters",
            freeze_json(dict(self.parameters)),
        )

        object.__setattr__(
            self,
            "raw_payload",
            freeze_json(dict(self.raw_payload)),
        )
