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

    # Align index: labels.csv has the same rows as the core matrix (post-cleanup)
    # Rebuild the same core matrix to align IDs
    core_cols = [c for c in config.io.core_features if c in df.columns]
    core = df[core_cols].dropna(how="any")
    core = core.reset_index(drop=False).rename(columns={"index": "orig_idx"})

    # Merge with labels
    merged = core[["orig_idx"]].copy()
    merged["anomaly_label"] = labels["anomaly_label"].values
    merged["anomaly_score"] = labels["anomaly_score"].values
    merged = merged.merge(
        df.reset_index(drop=False).rename(columns={"index": "orig_idx"}),
        on="orig_idx",
        how="left",
    )

    flagged = merged[merged["anomaly_label"] == -1].copy()
    print("=" * 70)
    print(f"Flagged samples: {len(flagged)} of {len(merged)}")
    print("=" * 70)

    cols = ["ID", "Lat", "Long", "Name of Area", "Lithology", "Alteration"]
    cols = [c for c in cols if c in flagged.columns]
    print("\n--- Location & geology ---")
    print(flagged[cols].to_string(index=False))

    geochem = ["Fe_pct", "Ca_pct", "Mg_pct", "Rb_ppm", "Cs_ppm",
               "Nb_ppm", "Ta_ppm", "U_ppm", "Th_ppm"]
    geochem = [c for c in geochem if c in flagged.columns]
    print("\n--- Geochemistry ---")
    print(flagged[geochem].to_string(index=False))

    print("\n--- Scores ---")
    print(flagged[["ID", "anomaly_score"]].sort_values("anomaly_score").to_string(index=False))

    # Compare against the "normal" population
    normal = merged[merged["anomaly_label"] == 1]
    print("\n--- Comparison: mean (anomaly vs normal) ---")
    compare = pd.DataFrame({
        "anomaly_mean": flagged[geochem].mean(),
        "normal_mean": normal[geochem].mean(),
    })
    compare["ratio"] = (compare["anomaly_mean"] /
                        compare["normal_mean"].replace(0, pd.NA)).round(2)
    print(compare.round(3).to_string())


if __name__ == "__main__":
    main()
