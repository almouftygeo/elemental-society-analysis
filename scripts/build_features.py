# scripts/build_features.py
"""
Build ML-ready features from the raw database.

The raw lab payload is stored in WIDE format (one row per sample,
one column per analyte). This script:
1. Parses censor flags from the wide row.
2. Aligns sensors, labs, and events with merge_asof.
3. Builds rolling and chemical features.
4. Writes ml_ready_features.csv and a manifest.
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


CENSOR_RE = r'^\s*([<>])\s*([0-9]+(?:[.,][0-9]+)?)\s*$'
ANALYTES = ["calcium", "silica", "chloride", "sulfate"]


def parse_censored(raw):
    """Return (value, censor_flag, detection_limit)."""
    if raw is None:
        return None, None, None
    raw = str(raw).strip()
    if raw.lower() in ("", "nan", "n/a", "na", "-", "--", "nd"):
        return None, None, None
    m = re.match(CENSOR_RE, raw)
    if m:
        op = m.group(1)
        dl = float(m.group(2).replace(",", "."))
        flag = "BELOW_DL" if op == "<" else "ABOVE_DL"
        return None, flag, dl
    try:
        return float(raw.replace(",", ".")), None, None
    except ValueError:
        return None, None, None


def load_sensors(conn):
    rows = conn.execute("SELECT raw_json FROM raw_sensor_payload").fetchall()
    records = [json.loads(r[0]) for r in rows]
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "tag": "sensor_id",
        "time": "event_time",
        "equipment": "equipment_id",
    })
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["parameter"] = df["sensor_id"].str.split("-").str[0].map({
        "TEMP": "temperature", "PRESS": "pressure",
        "FLOW": "flow", "PH": "ph", "COND": "conductivity",
    })
    return df


def load_labs(conn):
    """Load wide-format lab rows and parse censor flags."""
    rows = conn.execute("SELECT raw_json FROM raw_lab_payload").fetchall()
    records = [json.loads(r[0]) for r in rows]
    df = pd.DataFrame(records).drop_duplicates(subset=["sample_id"])

    for analyte in ANALYTES:
        if analyte not in df.columns:
            continue
        parsed = df[analyte].apply(parse_censored)
        df[analyte] = parsed.apply(lambda x: x[0])
        df[f"{analyte}_censor"] = parsed.apply(lambda x: x[1])
        df[f"{analyte}_dl"] = parsed.apply(lambda x: x[2])

    for col in ("sample_time", "result_available_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_events(conn):
    rows = conn.execute("SELECT raw_json FROM raw_event_payload").fetchall()
    records = [json.loads(r[0]) for r in rows]
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "time": "event_time",
        "equipment": "equipment_id",
        "type": "event_type",
    })
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def align(sensors, labs, events, max_lab_age_hours=72.0):
    sensor_wide = sensors.pivot_table(
        index="event_time", columns="parameter",
        values="value", aggfunc="mean",
    ).reset_index().sort_values("event_time").reset_index(drop=True)

    lab_wide = labs.sort_values("result_available_at").reset_index(drop=True)

    aligned = pd.merge_asof(
        sensor_wide.sort_values("event_time"),
        lab_wide,
        left_on="event_time",
        right_on="result_available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=max_lab_age_hours),
        suffixes=("", "_lab"),
    )

    aligned["hours_since_lab"] = (
        (aligned["event_time"] - aligned["result_available_at"])
        .dt.total_seconds() / 3600.0
    )
    aligned["lab_is_fresh"] = aligned["hours_since_lab"] < 24.0

    for event_type in ["ANTISCALANT_DOSING", "MAINTENANCE", "SAMPLING"]:
        subset = events[events["event_type"] == event_type][["event_time"]].copy()
        subset = subset.rename(columns={"event_time": f"{event_type}_time"})
        if subset.empty:
            aligned[f"hours_since_{event_type.lower()}"] = np.inf
            continue
        merged = pd.merge_asof(
            aligned.sort_values("event_time"),
            subset.sort_values(f"{event_type}_time"),
            left_on="event_time", right_on=f"{event_type}_time",
            direction="backward",
        )
        merged[f"hours_since_{event_type.lower()}"] = (
            (merged["event_time"] - merged[f"{event_type}_time"])
            .dt.total_seconds() / 3600.0
        ).fillna(np.inf)
        aligned = merged.drop(columns=[f"{event_type}_time"])

    return aligned


def add_features(df):
    df = df.set_index("event_time").sort_index()

    for col in ["temperature", "pressure", "flow", "ph"]:
        if col in df.columns:
            df[f"{col}_rolling_1h_mean"] = df[col].rolling("1h", closed="left").mean()
            df[f"{col}_rolling_1h_std"] = df[col].rolling("1h", closed="left").std()

    if "pressure" in df.columns:
        df["pressure_drop_1h"] = (
            df["pressure"] - df["pressure"].rolling("1h", closed="left").max()
        )

    df = df.reset_index()

    for analyte in ANALYTES:
        censor_col = f"{analyte}_censor"
        dl_col = f"{analyte}_dl"
        if analyte in df.columns and censor_col in df.columns and dl_col in df.columns:
            mask = df[censor_col] == "BELOW_DL"
            df.loc[mask, analyte] = df.loc[mask, dl_col] / 2.0

    if "calcium" in df.columns and "silica" in df.columns:
        df["ca_silica_ratio"] = df["calcium"] / (df["silica"] + 1e-9)

    dosing_col = "hours_since_antiscalant_dosing"
    if dosing_col in df.columns:
        df["is_post_dosing"] = (df[dosing_col] < 24).astype(int)

    return df


def main():
    print("=== main() started ===")
    db_path = "data/raw_db/maleg_raw.db"
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"DB: {db_path}, exists: {Path(db_path).exists()}")

    conn = sqlite3.connect(db_path)
    sensors = load_sensors(conn)
    labs = load_labs(conn)
    events = load_events(conn)
    conn.close()

    print(f"Sensors: {len(sensors)} rows, params: {sensors['parameter'].unique().tolist()}")
    print(f"Labs: {len(labs)} rows, analytes: {[a for a in ANALYTES if a in labs.columns]}")
    print(f"Events: {len(events)} rows, types: {events['event_type'].unique().tolist()}")

    aligned = align(sensors, labs, events)
    features = add_features(aligned)

    out_csv = out_dir / "ml_ready_features.csv"
    features.to_csv(out_csv, index=False)

    manifest = {
        "rows": len(features),
        "columns": len(features.columns),
        "column_names": features.columns.tolist(),
        "sensor_rows": len(sensors),
        "lab_rows": len(labs),
        "event_rows": len(events),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved {len(features)} rows x {len(features.columns)} columns")
    print(f"  CSV: {out_csv}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()