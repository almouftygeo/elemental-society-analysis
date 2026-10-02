"""Build the Li-clay subset from the Earth MRI geochemical compilation.

Schema: EMRI columns follow <Element>_<unit>_<METHOD>. Each sample is analyzed
by a subset of methods, so per-element method selection is coverage-driven.

Geochemistry rule: prefer TOTAL-digestion methods for the CLR pipeline.
Partial leaches (ICPH2O = water leach) only measure the soluble fraction
and would distort closure; use them only as a flagged fallback.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/external/usgs_ngdb/EMRI_Data.csv")
OUT = Path("data/processed/usgs_li_clay_subset.csv")

# Canonical pipeline features -> (element, unit) targets
CANONICAL = {
    "Si_pct": ("Si", "pct"),  "Al_pct": ("Al", "pct"),  "K_pct": ("K", "pct"),
    "Fe_pct": ("Fe", "pct"),  "Ca_pct": ("Ca", "pct"),
    "Mg_pct": ("Mg", "pct"),  "Li_ppm": ("Li", "ppm"),  "Rb_ppm": ("Rb", "ppm"),
    "Cs_ppm": ("Cs", "ppm"),  "Nb_ppm": ("Nb", "ppm"),  "Ta_ppm": ("Ta", "ppm"),
    "Sn_ppm": ("Sn", "ppm"),  "Be_ppm": ("Be", "ppm"),  "Ga_ppm": ("Ga", "ppm"),
    "Tl_ppm": ("Tl", "ppm"),  "U_ppm": ("U", "ppm"),    "Th_ppm": ("Th", "ppm"),
}

TOTAL_METHODS = ["ICMS4A", "ICP4A", "ICPMS4A", "ICPMS", "ICP", "XRF",
                 "XRFFE", "XRFPE", "NAA", "INAA", "FUS", "TOTA", "TOT4A"]
LEACH_METHODS = ["H2O", "AR", "ARA", "ICPH2O", "ICPCAL", "ICPBS", "ICPAA", "ICPSDS"]
LEACH_SET = set(LEACH_METHODS)

COL_RE = re.compile(r"^([A-Z][a-z]?)_(ppb|ppm|pct|mgkg)_(.+)$")
ID_CANDIDATES = ["ID", "Lab_ID", "Sample_ID", "lab_id", "sample_id",
                 "IGSN", "igsn", "SS_ID", "Site_ID", "Record_ID"]
COORD_CANDIDATES = {"Lat": ["lat", "latitude"], "Long": ["lon", "long", "longitude"]}

TO_PPM = {"ppb": 1e-3, "ppm": 1.0, "mgkg": 1.0}


def find_id_column(df: pd.DataFrame) -> str:
    for cand in ID_CANDIDATES:
        if cand in df.columns:
            return cand
    hits = [c for c in df.columns if re.search(r"(id|igsn)", c, re.I)]
    raise SystemExit(
        f"No ID column found. Candidates tried: {ID_CANDIDATES}.\n"
        f"Closest matches: {hits[:10]} -- extend ID_CANDIDATES and rerun."
    )


def main() -> None:
    print(f"Loading {RAW} ...")
    df = pd.read_csv(RAW, low_memory=False)
    print(f"Shape: {df.shape[0]} samples x {df.shape[1]} columns")

    id_col = find_id_column(df)
    print(f"ID column detected: '{id_col}'")

    # -- Parse every element/unit/method column and build a coverage table --
    records = []
    for col in df.columns:
        m = COL_RE.match(col)
        if m:
            records.append({"col": col, "element": m.group(1),
                            "unit": m.group(2), "method": m.group(3)})
    parsed = pd.DataFrame(records)
    print(f"\nParsed {len(parsed)} geochemical columns "
          f"({df.shape[1] - len(parsed)} non-geochemical ignored)")

    # -- Coverage report for target elements only --
    print("\n=== Coverage per element x method (target suite) ===")
    chosen: dict[str, tuple[str, str, str]] = {}   # canonical -> (col, src_unit, method)
    for canon, (elem, unit) in CANONICAL.items():
        rows = parsed[parsed["element"] == elem].copy()
        if rows.empty:
            print(f"{canon:8s} -- element NOT PRESENT in EMRI")
            continue
        rows["coverage"] = rows["col"].map(lambda c: df[c].notna().mean())
        rows["is_total"] = ~rows["method"].isin(LEACH_SET)
        rows = rows.sort_values(["is_total", "coverage"],
                                ascending=[False, False])
        best = rows.iloc[0]
        flag = "OK " if best["is_total"] else "LEACH-FALLBACK"
        print(f"{canon:8s} -> {best['col']:22s} cov={best['coverage']:.2f}  [{flag}]"
              f"   (alts: {', '.join(rows['col'].head(3))})")
        if best["coverage"] >= 0.30:               # minimum viability threshold
            chosen[canon] = (best["col"], best["unit"], best["method"])
        else:
            print(f"{canon:8s} -- best coverage {best['coverage']:.2f} < 0.30, dropped")

    # -- Assemble the subset --
    out = pd.DataFrame({"ID": df[id_col].astype(str).str.strip()})
    for coord, keys in {"Lat": ("lat",), "Long": ("lon", "long")}.items():
        hits = [c for c in df.columns if c.lower().startswith(keys)]
        if hits:
            out[coord] = df[hits[0]]
            print(f"Coordinate {coord} <- '{hits[0]}'")
        else:
            print(f"WARNING: no column found for {coord}")
    for canon, (col, src_unit, _method) in chosen.items():
        values = pd.to_numeric(df[col], errors="coerce")
        # USGS convention: negative value = censored below |value| (below DL).
        # Same 2/3-of-detection-limit rule the project loader applies to "<x".
        values = values.mask(values < 0, (2.0 / 3.0) * values.abs())
        if src_unit in TO_PPM and canon.endswith("_ppm"):
            values = values * TO_PPM[src_unit]     # ppb/mgkg -> ppm
        out[canon] = values

    out = out.dropna(subset=["ID"]).drop_duplicates(subset="ID")
    feats = [c for c in CANONICAL if c in out.columns]
    before = len(out)
    out = out.dropna(subset=feats).reset_index(drop=True)
    print(f"Complete-case filter: kept {len(out)} of {before} rows")
    print(f"\nFinal subset: {out.shape[0]} samples, "
          f"{sum(c in out.columns for c in CANONICAL)} features (+ID/coords)")
    

    feats = [c for c in CANONICAL if c in out.columns]
    print("\n=== Final feature coverage ===")
    print(out[feats].notna().mean().sort_values(ascending=False).to_string())
    if "Li_ppm" in out.columns:
        print("\n=== Li_ppm distribution ===")
        print(out["Li_ppm"].describe().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
