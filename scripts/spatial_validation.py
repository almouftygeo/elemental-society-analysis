"""Spatial validation: do LOF anomalies cluster near known Li-clay districts?

Inputs (written by run_pipeline + build_li_clay_dataset):
    reports/usgs_li_clay/anomaly_labels.csv   -- one row per sample, anomaly flag
    data/processed/usgs_li_clay_subset.csv    -- ID, Lat, Long + 17 features

Outputs:
    reports/usgs_li_clay/anomaly_map.png      -- map: anomalies vs reference sites
    stdout enrichment table                   -- per-district anomaly rate vs global
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # render to file; no display needed
import matplotlib.pyplot as plt
import json 
import pandas as pd

LABELS = Path("reports/usgs_li_clay/anomaly_labels.csv")
SUBSET = Path("data/processed/usgs_li_clay_subset.csv")
OUT_PNG = Path("reports/usgs_li_clay/anomaly_map.png")
METRICS = Path("reports/usgs_li_clay/metrics.json")


# Li-clay districts of the McDermitt volcanic field (OR/NV border)
SITES = {
    "Thacker Pass": (41.677, -117.871),
    "McDermitt Caldera": (42.000, -117.900),
}
RADIUS_DEG = 0.5               # ~55 km square window at this latitude

LABEL_COLS = ["anomaly_label", "is_anomaly", "anomaly", "label", "outlier", "is_outlier"]


def norm_anomaly(s: pd.Series) -> pd.Series:
    """Normalize whatever label convention the pipeline used to bool."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        uniq = set(pd.unique(s.dropna()))
        if uniq.issubset({-1, 1}):
            return s == -1                      # sklearn: -1 = anomaly
        return s.fillna(0).astype(bool)         # 0/1 flag
    low = s.astype(str).str.strip().str.lower()
    return low.isin({"true", "yes", "y", "anomaly", "outlier", "-1"})


def main() -> None:
    lab = pd.read_csv(LABELS)
    sub = pd.read_csv(SUBSET)

    sub["ID"] = sub["ID"].astype(str).str.strip()

    lab_col = next((c for c in LABEL_COLS if c in lab.columns), None)

    if lab_col is not None:
        print(
            f"Label column: '{lab_col}', "
            f"unique values: {sorted(pd.unique(lab[lab_col]))[:10]}"
        )
        lab["anom"] = norm_anomaly(lab[lab_col])

    elif "anomaly_score" in lab.columns:
        n_anom = json.loads(METRICS.read_text())["anomaly_count"]
        thr = lab["anomaly_score"].nsmallest(n_anom).max()
        lab["anom"] = lab["anomaly_score"] <= thr
        print(f"No flag column -> derived: LOWEST {n_anom} of {len(lab)} "
              f"by anomaly_score (threshold {thr:.6g}); "
              f"sklearn decision_function: most negative = anomaly")

    else:
        raise SystemExit(
            f"No usable label column in {LABELS}: {list(lab.columns)}"
        )

    # -- join coords: by ID if possible, else positional (order kept: 0 imputed) --
    if "ID" in lab.columns:
        lab["ID"] = lab["ID"].astype(str).str.strip()
        merged = lab.merge(sub[["ID", "Lat", "Long"]], on="ID", how="left")
        print(f"Joined on ID: {merged['Lat'].notna().sum()} of {len(merged)} matched")
    else:
        print("WARNING: labels have no ID column -> positional join (order preserved)")
        merged = lab.copy()
        merged["Lat"] = sub["Lat"].to_numpy()[: len(merged)]
        merged["Long"] = sub["Long"].to_numpy()[: len(merged)]

    n0 = len(merged)
    merged = merged.dropna(subset=["Lat", "Long"])
    if len(merged) < n0:
        print(f"Dropped {n0 - len(merged)} row(s) without coordinates")

    rate = merged["anom"].mean()
    print(f"\nGlobal anomaly rate: {merged['anom'].sum()} / {len(merged)} = {rate:.3%}\n")

    print(f"{'site':<20}{'samples':>8}{'anomalies':>11}{'rate':>9}{'enrich':>9}")
    for name, (slat, slon) in SITES.items():
        near = merged[(merged["Lat"].sub(slat).abs() <= RADIUS_DEG)
                      & (merged["Long"].sub(slon).abs() <= RADIUS_DEG)]
        n, k = len(near), int(near["anom"].sum())
        r = k / n if n else float("nan")
        e = (r / rate) if n and rate else float("nan")
        r_s = f"{r:.2%}" if n else "--"
        e_s = f"{e:.2f}x" if n and rate else "--"
        print(f"{name:<20}{n:>8}{k:>11}{r_s:>9}{e_s:>9}")

    # -- map --
    fig, ax = plt.subplots(figsize=(9, 8))
    normal = merged[~merged["anom"]]
    anomal = merged[merged["anom"]]
    ax.scatter(normal["Long"], normal["Lat"], s=6, c="0.75", alpha=0.4,
               label=f"normal (n={len(normal)})")
    ax.scatter(anomal["Long"], anomal["Lat"], s=16, c="crimson", alpha=0.85,
               edgecolors="darkred", linewidths=0.3,
               label=f"anomaly (n={len(anomal)})")
    for name, (slat, slon) in SITES.items():
        ax.scatter([slon], [slat], marker="*", s=350, c="black", zorder=5)
        ax.annotate(name, (slon, slat), textcoords="offset points",
                    xytext=(8, 6), fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude (WGS84)")
    ax.set_ylabel("Latitude (WGS84)")
    ax.set_title(f"USGS Li-clay subset - LOF k=50 c=0.05 - "
                 f"{len(anomal)} anomalies in {len(merged)} samples")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nMap saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
