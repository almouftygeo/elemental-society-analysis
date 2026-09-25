# LOF sensitivity experiment: 2% contamination

## Fixed settings

- Dataset: `data/raw/all_new_data.csv`
- Cleaning: missing-ID removal, duplicate-column removal, median imputation
- Features: 13 core geochemical variables
- Transform: CLR
- Detector: Local Outlier Factor
- Contamination: 0.02

## Results

| n_neighbors | Anomaly count | Anomaly IDs |
|---:|---:|---|
| 5 | 0 | None |
| 12 | 0 | None |
| 15 | 1 | PT-1 |
| 22 | 0 | None |

## Interpretation

At a strict 2% contamination threshold and 28 samples, only the
`n_neighbors = 15` configuration identified PT-1. This is a
parameter-supported multivariate target, not proof of mineralization.
Geological and spatial validation are required.
