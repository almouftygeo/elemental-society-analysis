"""Inspect LOF anomalies against the original geochemical data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from elemental_society_analysis.geochem.loader import load_dataframe
from elemental_society_analysis.utils.config import load_config


def main() -> None:
    config = load_config()
    df = load_dataframe(config.io)

    labels_path = Path(config.io.output_dir) / "anomaly_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Run the pipeline first.")

    labels = pd.read_csv(labels_path)
    if "ID" not in labels.columns:
        raise ValueError(
            "anomaly_labels.csv has no ID column - rerun the pipeline "
            "so the ID is written alongside the anomaly labels."
        )

    merged = df.merge(
        labels[["ID", "anomaly_label", "anomaly_score"]],
        on="ID",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Merge produced no rows - check ID formatting / dtype.")

    flagged = merged[merged["anomaly_label"] == -1].copy()
    normal = merged[merged["anomaly_label"] == 1].copy()

    print("=" * 70)
    print(f"Flagged samples: {len(flagged)} of {len(merged)}")
    print("=" * 70)

    cols = ["ID", "Lat", "Long", "Name of Area", "Lithology", "Alteration"]
    cols = [c for c in cols if c in flagged.columns]
    print("\n--- Location & geology ---")
    print(flagged[cols].to_string(index=False))

    geochem = [
        c for c in labels.columns
        if c not in {"ID", "anomaly_label", "anomaly_score"}
    ]
    print("\n--- Geochemistry ---")
    print(flagged[["ID", *geochem]].to_string(index=False))

    print("\n--- Scores (most anomalous first) ---")
    print(
        flagged[["ID", "anomaly_score"]]
        .sort_values("anomaly_score")
        .to_string(index=False)
    )

    if not normal.empty and geochem:
        print("\n--- Comparison: mean (anomaly vs normal) ---")
        compare = pd.DataFrame({
            "anomaly_mean": flagged[geochem].mean(numeric_only=True),
            "normal_mean": normal[geochem].mean(numeric_only=True),
        })
        compare["ratio"] = (
            compare["anomaly_mean"]
            / compare["normal_mean"].replace(0, pd.NA)
        ).round(2)
        print(compare.round(3).to_string())


if __name__ == "__main__":
    main()
