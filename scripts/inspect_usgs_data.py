"""First-pass inspection of the local USGS geochemical CSV download."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "external" / "usgs_ngdb"

csvs = sorted(
    RAW.glob("*.csv"),
    key=lambda path: path.stat().st_size,
    reverse=True,
)

if not csvs:
    raise FileNotFoundError(
        f"No CSV files found in {RAW}.\n"
        "Copy the USGS CSV to data/external/usgs_ngdb/ first."
    )

print(f"Data folder: {RAW}")
print("\nCSV files found:")

for csv_path in csvs[:5]:
    size_mb = csv_path.stat().st_size / 1_000_000
    print(f"  {csv_path.name}: {size_mb:.1f} MB")

main_csv = csvs[0]
print(f"\nLoading largest CSV: {main_csv.name}")

df = pd.read_csv(main_csv, low_memory=False)

print(f"\nShape: {df.shape}")
print(f"\nAll columns ({len(df.columns)}):")

for column in df.columns:
    print(f"  {column}")
