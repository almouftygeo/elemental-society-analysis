import pandas as pd


class FeatureBuilder:
    """As-of joins + freshness + quality-aware feature engineering."""

    def __init__(self, max_lab_age_hours: float = 72.0):
        self.max_lab_age_hours = max_lab_age_hours

    def build(self, sensor_df: pd.DataFrame,
              lab_df: pd.DataFrame,
              event_df: pd.DataFrame) -> pd.DataFrame:
        # 1. As-of join: latest lab result available at each sensor time
        lab_sorted = lab_df.sort_values("result_available_at")
        merged = pd.merge_asof(
            sensor_df.sort_values("event_time"),
            lab_sorted,
            left_on="event_time",
            right_on="result_available_at",
            by="equipment_id",
            direction="backward",
        )

        # 2. Freshness features
        merged["hours_since_lab"] = (
            (merged["event_time"] - merged["sample_time"]).dt.total_seconds() / 3600
        )
        merged["lab_is_fresh"] = (
            merged["hours_since_lab"] <= self.max_lab_age_hours
        )

        # 3. Operational features
        merged = self._add_time_since_event(
            merged, event_df, "ANTISCALANT_DOSING", "hours_since_dosing"
        )

        # 4. Quality-aware filtering
        if "quality_status" in merged.columns:
            merged = merged[merged["quality_status"] != "REJECTED"]

        # 5. Rolling features (no future leakage: shift before rolling)
        merged = merged.set_index("event_time")
        for col in ("temperature", "pressure"):
            if col in merged.columns:
                merged[f"{col}_rolling_mean_1h"] = (
                    merged.groupby("equipment_id")[col]
                    .transform(lambda s: s.shift(1).rolling("1h").mean())
                )
        merged = merged.reset_index()
        return merged

    def _add_time_since_event(self, df, events, event_type, col_name):
        if events.empty:
            df[col_name] = float("nan")
            return df
        ev = events[events["event_type"] == event_type][
            ["equipment_id", "event_time"]
        ].rename(columns={"event_time": "last_event_time"}).sort_values("last_event_time")
        df = pd.merge_asof(
            df.sort_values("event_time"),
            ev,
            left_on="event_time",
            right_on="last_event_time",
            by="equipment_id",
            direction="backward",
        )
        df[col_name] = (
            (df["event_time"] - df["last_event_time"]).dt.total_seconds() / 3600
        )
        return df
