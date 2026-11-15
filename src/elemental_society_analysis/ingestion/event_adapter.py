import json
import uuid
from datetime import datetime, timezone

from .canonical import OperationalEvent


class EventAdapter:
    """Parses operator/SCADA event JSON into a canonical OperationalEvent."""

    REQUIRED_FIELDS = {"event_id", "equipment", "type", "time"}

    def parse(self, raw: str | dict) -> OperationalEvent:
        payload = json.loads(raw) if isinstance(raw, str) else raw

        missing = self.REQUIRED_FIELDS - payload.keys()
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        return OperationalEvent(
            event_id=payload["event_id"],
            equipment_id=payload["equipment"],
            event_type=payload["type"],
            event_time=self._parse_time(payload["time"]),
            ingested_at=datetime.now(timezone.utc),
            parameters={k: v for k, v in payload.items()
                        if k not in self.REQUIRED_FIELDS},
            source_system="OPERATOR_LOG",
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
