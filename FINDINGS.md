# Findings - v0.2.0-public-data-baseline

## Pipeline
USGS NGDB subset -> 13,096 complete-case samples x 17 features
(6 majors + 11 trace: Li, Rb, Cs, Nb, Ta, Sn, Be, Ga, Tl, U, Th)
-> row-normalized 6-oxide subcomposition closure -> CLR
-> LOF (k=50, contamination=0.05) -> 619 anomalies (4.73%).

## Result 1 - Detection lift (VALIDATED)
Among 11,989 coordinate-bearing samples, Li-rich material is flagged at
53-58% recall at every threshold tested, an ~11x enrichment over the
5.01% base flag rate (binomial p ~ 1e-13 at >=500 ppm).
The unsupervised pipeline concentrates its anomaly budget on Li-rich
clay with zero supervised information.

## Result 2 - District blindness (MEASURED, not assumed)
Windows fully containing Thacker Pass and the McDermitt Caldera
(40x30 to 60x35 km) hold only 146 NGDB samples, max Li = 131 ppm:
the national mapping-grade database contains no ore-grade clay there.
In-window flag rate 1.37% vs 5.06% outside (0.27x, binomial p=0.02):
local-density methods flag isolated lithological outliers and are
structurally blind to dense mineralized clusters - where those
clusters exist in the data. Here they do not.

## Result 3 - Mechanism of misses (OPEN HYPOTHESIS)
Within the 30 Li-rich (>=500 ppm) samples, missed ones have smaller
median 10-NN distance (0.029 vs 0.045 deg), consistent with
density-masking, but IQRs overlap heavily and group means invert
(0.119 vs 0.087 deg). Suggestive, not decisive at n=30.
Regional contrast: Arizona cell (34N, -114..-112W; Artillery Peak /
Walapai country) 10/13 flagged vs Mojave playa corridor (34N,
-118..-116W) 1/9 flagged. v0.3 priority: cluster-aware detection
(flag rare clusters, not only isolated points).

## Artifacts
- configs/experiments/usgs_li_clay_lof_k50_c05.yaml  (experiment config)
- reports/usgs_li_clay/metrics.json                  (n=13096, k=50, c=0.05)
- reports/usgs_li_clay/anomaly_labels.csv            (per-sample labels+scores)
- reports/usgs_li_clay/anomaly_map.png               (national anomaly map)
- reports/usgs_li_clay/li_threshold_flag_rates.txt   (recall/enrichment table)
- scripts/spatial_validation.py                      (district validation)
- scripts/build_li_clay_dataset.py                   (subset builder)

## Known limitations
- 1,107 samples (8.5%) lack coordinates; excluded from spatial analyses.
- Censored-value ties (Th 1000.111, Be 3.333, Tl 0.333 = 2/3*DL artifacts)
  create artificial dense clouds; proper below-DL imputation is v0.3.
- is_closed=false is expected: 6-oxide subcomposition (partial-sum CV 14.6%).
- Pipeline hardcodes reports/ output dir, ignoring config io.output_dir.
