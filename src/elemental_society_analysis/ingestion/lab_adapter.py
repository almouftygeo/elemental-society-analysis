import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .canonical import CensorFlag, LabResult, QualityStatus


_DETECTION_LIMIT_RE = re.compile(
    r"^\s*([<>]=?)\s*([0-9]+(?:[.,][0-9]+)?)\s*$"
)


class LabCSVAdapter:
    """Parses laboratory CSV into canonical LabResult records,
    preserving censored values."""

    def __init__(self, decimal: str = ","):
        self.decimal = decimal

    def parse(self, path: str) -> list[LabResult]:
        results = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                for analyte_col in ("calcium", "silica", "chloride"):
                    raw = row.get(analyte_col, "")
                    value, flag, dl = self._parse_value(raw)
                    results.append(LabResult(
                        result_id=str(uuid.uuid4()),
                        sample_id=row["sample_id"],
                        sampling_point_id=row.get("sampling_point_id", "UNKNOWN"),
                        analyte=analyte_col,
                        raw_string=raw,
                        value=value,
                        unit="mg/L",
                        censor_flag=flag,
                        detection_limit=dl,
                        sample_time=self._parse_time(row["sample_time"]),
                        result_available_at=self._parse_time(
                            row.get("result_available_at", row["sample_time"])
                        ),
                        ingested_at=datetime.now(timezone.utc),
                        quality_status=QualityStatus.VALID,
                        quality_reason=None,
                        source_system="LABORATORY",
                        raw_payload=dict(row),
                    ))
        return results

    def _parse_value(self, raw: str):
        text = raw.strip()
        if not text or text.upper() in {"NA", "N/A", "--", "?"}:
            return None, CensorFlag.MISSING, None
        m = _DETECTION_LIMIT_RE.match(text)
        if m:
            sign, num = m.groups()
            dl = float(num.replace(self.decimal, "."))
            flag = CensorFlag.BELOW_DL if "<" in sign else CensorFlag.ABOVE_DL
            return None, flag, dl
        try:
            return float(text.replace(self.decimal, ".")), CensorFlag.NONE, None
        except ValueError:
            return None, CensorFlag.PARSE_ERROR, None

    @staticmethod
    def _parse_time(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                "timestamp must be timezone-aware; "
                "naive timestamps are not accepted"
            )
        return dt.astimezone(timezone.utc)
