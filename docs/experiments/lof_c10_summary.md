# LOF sensitivity experiment: 10% contamination

## Fixed settings

- Dataset: `data/raw/all_new_data.csv`
- Cleaning: missing-ID removal, duplicate-column removal, coverage filtering, and median imputation
- Features: 13 core geochemical variables
- Transform: CLR
- Detector: Local Outlier Factor
- Contamination: 0.10

## Results

| n_neighbors | Anomaly count | Anomaly IDs |
|---:|---:|---|
| 5 | 2 | KR-1; 240621-3-B |
| 12 | 3 | KR-1; 230621-1-C3; 240621-3-A |
| 15 | 3 | PT-1; 2MS; 3MS-A |
| 22 | 3 | PT-1; 230621-1-C1; 230621-1-C3 |

## Interpretation

PT-1 is the leading Li-oriented target: it appears at n_neighbors 15
and 22 under 10% contamination and was the only anomaly at the strict
2% threshold for n_neighbors 15.

KR-1 is a repeated local-scale anomaly at n_neighbors 5 and 12.
230621-1-C3 is a repeated compositional anomaly at n_neighbors 12 and 22
but has low Li and requires detection-limit and geological-context checks.

These are sensitivity-screening results, not proof of mineralization.
Validate targets using spatial position, lithology, alteration, sample
medium, analytical detection limits, and an independent anomaly detector.
