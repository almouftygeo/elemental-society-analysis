import json
import sqlite3
from pathlib import Path

import pandas as pd

from elemental_society_analysis.integration.feature_builder import FeatureBuilder


def load_from_raw_db(db_path: str):
    conn = sqlite3.connect(db_path)
    sensor_rows = conn.execute(
        "SELECT raw_json FROM raw_sensor_payload"
    ).fetchall()
    lab_rows = conn.execute(
        "SELECT raw_json FROM raw_lab_payload"
    ).fetchall()
    event_rows = conn.execute(
        "SELECT raw_json FROM raw_event_payload"
    ).fetchall()
    conn.close()

    sensor_df = pd.DataFrame([json.loads(r[0]) for r in sensor_rows])
    sensor_df = sensor_df.rename(columns={
        "tag": "sensor_id", "time": "event_time", "equipment": "equipment_id"
    })
    sensor_df["event_time"] = pd.to_datetime(sensor_df["event_time"])
    sensor_df["quality_status"] = "VALID"

    lab_df = pd.DataFrame([json.loads(r[0]) for r in lab_rows])
    lab_df = lab_df.rename(columns={
        "sample_time": "sample_time",
        "result_available_at": "result_available_at",
        "equipment": "equipment_id",
    })
    for col in ("sample_time", "result_available_at"):
        lab_df[col] = pd.to_datetime(lab_df[col])
    # Pivot: one row per sample with analyte columns
    lab_wide = lab_df.pivot_table(
        index=["sample_id", "sampling_point_id", "sample_time",
               "result_available_at", "equipment_id"],
        columns="analyte", values="value", aggfunc="first"
    ).reset_index()
    lab_wide = lab_wide.rename(columns={
        "sample_time": "sample_time",
        "result_available_at": "result_available_at",
    })

    event_df = pd.DataFrame([json.loads(r[0]) for r in event_rows])
    event_df = event_df.rename(columns={
        "time": "event_time", "equipment": "equipment_id", "type": "event_type"
    })
    event_df["event_time"] = pd.to_datetime(event_df["event_time"])

    return sensor_df, lab_wide, event_df


def main():
    db_path = "data/raw_db/maleg_raw.db"
    sensor_df, lab_df, event_df = load_from_raw_db(db_path)
    builder = FeatureBuilder(max_lab_age_hours=72.0)
    features = builder.build(sensor_df, lab_df, event_df)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "ml_ready_features.csv", index=False)
    print(features.head())
    print(f"Saved {len(features)} rows to {out / 'ml_ready_features.csv'}")


if __name__ == "__main__":
    main()
