import json
import uuid
from datetime import datetime, timezone

from .canonical import SensorRecord


class SensorAdapter:
    """Parses PLC/SCADA JSON into a canonical SensorRecord."""

    REQUIRED_FIELDS = {"tag", "value", "unit", "time"}

    def parse(self, raw: str | dict) -> SensorRecord:
        payload = json.loads(raw) if isinstance(raw, str) else raw

        missing = self.REQUIRED_FIELDS - payload.keys()
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        return SensorRecord(
            measurement_id=str(uuid.uuid4()),
            sensor_id=payload["tag"],
            equipment_id=payload.get("equipment", "UNKNOWN"),
            value=float(payload["value"]),
            unit=payload["unit"],
            event_time=self._parse_time(payload["time"]),
            ingested_at=datetime.now(timezone.utc),
            source_system="PLC_SCADA",
            raw_payload=payload,
        )

    @staticmethod
    def _parse_time(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                "timestamp must be timezone-aware; "
                "naive timestamps are not accepted"
            )
        return dt.astimezone(timezone.utc)
