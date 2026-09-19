"""Chondrite-normalized REE diagnostics for the Tokaj dataset.

Reads data/raw/all_new_data.csv via the project loader and writes:
  reports/ree_patterns.csv     per-sample REE diagnostics
  reports/ree_ranked.csv       samples ranked by HREE fraction

Usage:
  python scripts/ree_diagnostics.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from elemental_society_analysis.geochem.loader import load_dataframe
from elemental_society_analysis.utils.config import load_config


# Chondrite values (McDonough & Sun 1995), ppm.
CHONDRITE = {
    "La": 0.237, "Ce": 0.613, "Pr": 0.0927, "Nd": 0.457, "Sm": 0.148,
    "Eu": 0.0563, "Gd": 0.199, "Tb": 0.0361, "Dy": 0.246, "Ho": 0.0552,
    "Er": 0.160, "Tm": 0.0243, "Yb": 0.162, "Lu": 0.0245, "Y": 1.57,
}
LREE = ["La", "Ce", "Pr", "Nd", "Sm", "Eu"]
HREE = ["Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"]
ALL15 = LREE + HREE


def _ree_columns(df: pd.DataFrame) -> list[str]:
    return [f"{e}_ppm" for e in ALL15 if f"{e}_ppm" in df.columns]


def compute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample chondrite-normalized REE metrics.

    Only rows with complete REE suites are kept.
    """
    ree_cols = _ree_columns(df)
    if len(ree_cols) != len(ALL15):
        raise ValueError(
            f"Expected {len(ALL15)} REE columns, found {len(ree_cols)}"
        )

    core = df[df[ree_cols].notna().all(axis=1)].copy()
    if core.empty:
        raise ValueError("No samples with complete REE suites.")

    # Chondrite-normalized values
    for e in ALL15:
        core[f"{e}_N"] = core[f"{e}_ppm"] / CHONDRITE[e]

    lree_cols = [f"{e}_ppm" for e in LREE]
    hree_cols = [f"{e}_ppm" for e in HREE]

    core["SREE_ppm"] = core[ree_cols].sum(axis=1)
    core["LREE_ppm"] = core[lree_cols].sum(axis=1)
    core["HREE_ppm"] = core[hree_cols].sum(axis=1)
    core["HREE_frac"] = core["HREE_ppm"] / core["SREE_ppm"]
    core["Y_share"] = core["Y_ppm"] / core["HREE_ppm"]

    core["LaYb_N"] = core["La_N"] / core["Yb_N"]
    core["GdYb_N"] = core["Gd_N"] / core["Yb_N"]
    core["LaSm_N"] = core["La_N"] / core["Sm_N"]
    core["Eu_Eu*"] = core["Eu_N"] / np.sqrt(core["Sm_N"] * core["Gd_N"])
    core["Ce_Ce*"] = core["Ce_N"] / np.sqrt(core["La_N"] * core["Pr_N"])

    return core


def print_summary(core: pd.DataFrame) -> None:
    print("=" * 78)
    print(f"REE diagnostics — {len(core)} samples with complete 15-element REE suite")
    print("=" * 78)

    cols = ["ID", "Name of Area", "SREE_ppm", "HREE_frac", "Y_share",
            "LaYb_N", "GdYb_N", "Eu_Eu*", "Ce_Ce*"]
    cols = [c for c in cols if c in core.columns]

    print("\n--- 10 highest HREE fraction ---")
    top = core.sort_values("HREE_frac", ascending=False).head(10)
    print(top[cols].round(3).to_string(index=False))

    print("\n--- 10 strongest negative Eu anomaly (lowest Eu/Eu*) ---")
    eu = core.sort_values("Eu_Eu*").head(10)
    print(eu[cols].round(3).to_string(index=False))

    print("\n--- 10 highest La/Yb_N (most LREE-enriched) ---")
    lree = core.sort_values("LaYb_N", ascending=False).head(10)
    print(lree[cols].round(3).to_string(index=False))

    print("\n--- population summary ---")
    print(
        core[["SREE_ppm", "HREE_frac", "Y_share",
              "LaYb_N", "GdYb_N", "Eu_Eu*", "Ce_Ce*"]]
        .describe().round(3).to_string()
    )


def main() -> None:
    config = load_config()
    df = load_dataframe(config.io)

    core = compute_diagnostics(df)

    out_dir = Path(config.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["ID", "Name of Area", "Lithology", "Alteration",
            "SREE_ppm", "LREE_ppm", "HREE_ppm", "HREE_frac", "Y_share",
            "LaYb_N", "GdYb_N", "LaSm_N", "Eu_Eu*", "Ce_Ce*"]
    cols = [c for c in cols if c in core.columns]

    patterns_path = out_dir / "ree_patterns.csv"
    core[cols].to_csv(patterns_path, index=False)
    print(f"\nWrote {len(core)} samples -> {patterns_path}")

    ranked = core.sort_values(["HREE_frac", "SREE_ppm"], ascending=False)
    ranked_path = out_dir / "ree_ranked.csv"
    ranked[cols].to_csv(ranked_path, index=False)
    print(f"Wrote ranked table -> {ranked_path}")

    print_summary(core)


if __name__ == "__main__":
    main()
