AGENTS.md — Elemental Society Analysis: Project Memory & Operating Manual
Purpose: This file is the single source of truth for the project’s history,
conventions, and current state. Any AI assistant (or human) reading this should
be able to continue work without re-discovering anything. Last updated: 2026-10-03.

1. PROJECT IDENTITY
Repo: https://github.com/almouftygeo/elemental-society-analysis
Owner/machine: AlmouftyGeo (Windows, Git Bash / MINGW64, VS Code, conda env elemental-society-analysis)
Goal: Unsupervised geochemical anomaly detection for lithium-clay exploration,
applied to public USGS data (NGDB), with spatial validation against known districts.
Active branch: feat/public-data-loader-v2
Current release: tag v0.2.0-public-data-baseline (pushed to GitHub 2026-10-02)
2. CURRENT STATE (v0.2.0 — VALIDATED & SEALED)
Pipeline
Entry: python -m elemental_society_analysis.core.run_pipeline <config.yaml>
Run command used for the baseline:
python -m elemental_society_analysis.core.run_pipeline configs/experiments/usgs_li_clay_lof_k50_c05.yaml
Flow: raw CSV → loader (dedupe, drop missing-ID, dup columns) → median-impute
→ drop low-coverage features (<80%) → row-normalize 6-oxide subcomposition
→ CLR transform → LOF (k=50, contamination=0.05) → labels + scores + metrics.
Dataset: data/processed/usgs_li_clay_subset.csv — 13,096 samples × 17 features
(6 majors: Si, Al, K, Fe, Ca, Mg pct; 11 traces: Li, Rb, Cs, Nb, Ta, Sn, Be, Ga, Tl, U, Th ppm).
Output: 619 anomalies (4.73%). Note: pipeline writes to hardcoded reports/,
ignoring config io.output_dir (known defect). Artifacts moved/kept in
reports/usgs_li_clay/.
Headline results (see FINDINGS.md for full detail)
Detection lift ~11× (VALIDATED): Li-rich samples (≥200/500/1000 ppm) flagged
at 53–58% recall vs 5.01% base rate; binomial p ~ 1e-13 at ≥500 ppm.
Hotspot: Arizona cell (34N, −114…−112W — Artillery Peak/Walapai Li-clay country),
10/13 flagged. Mojave corridor (34N, −118…−116W): only 1/9 flagged.
District blindness (MEASURED): windows fully containing Thacker Pass and
McDermitt Caldera hold only 146 NGDB samples, max Li = 131 ppm (no public ore
samples). Flag rate in-window 1.37% vs 5.06% out (0.27×, p=0.02).
Root cause: NGDB is mapping-grade (~1 sample per few hundred km²); ore zones
(few km²) are absent from the data. NOT a pipeline defect.
Density-masking hypothesis (OPEN): missed Li-rich samples have smaller median
10-NN distance (0.029 vs 0.045°) but IQRs overlap; means invert (n=30, indecisive).
Mann-Whitney U test optional, not yet run.
Key facts to remember
anomaly_score = sklearn LOF decision_function → more NEGATIVE = more anomalous.
anomaly_label: −1 = anomaly, +1 = normal.
Labels CSV columns: ID, <17 features…, anomaly_label, anomaly_score>.
is_closed=false is EXPECTED: only 6 of 11 oxides present → partial-sum CV 14.6%.
Ratios preserved under CLR subcomposition; documented limitation.
Censored-value artifacts: Th_ppm=1000.111, Be=3.333, Tl=0.333 (⅔·DL ties)
create artificial dense clouds — affects LOF neighborhoods.
1,107 samples (8.5%) lack coordinates; excluded from spatial work.
Two duplicate-coordinate samples exist (kNN min distance = 0).
3. FILE MAP
configs/experiments/usgs_li_clay_lof_k50_c05.yaml # baseline experiment config
configs/default.yaml # Tokaj-era defaults
data/processed/usgs_li_clay_subset.csv # 13,096×20 subset (ID,Lat,Long+17)
reports/usgs_li_clay/metrics.json # n=13096, k=50, c=0.05, 619 anomalies
reports/usgs_li_clay/anomaly_labels.csv # per-sample labels + scores
reports/usgs_li_clay/anomaly_map.png # national anomaly map
reports/usgs_li_clay/li_threshold_flag_rates.txt # recall/enrichment evidence
scripts/build_li_clay_dataset.py # builds subset from NGDB
scripts/spatial_validation.py # district windows + map + enrichment
scripts/inspect_anomalies.py, scripts/ree_scorer.py # Tokaj-era utilities
FINDINGS.md # one-page results document
src/elemental_society_analysis/
core/run_pipeline.py # CLI: config path arg (patched)
geochem/loader.py # loading/cleaning logs
models/detectors.py # LOF wrapper; decision_function passthrough

## 4. HISTORY OF THE CAMPAIGN (defects → fixes → habits)

Chronological; each entry is a lesson that shaped project conventions:

1. **Missing builder script** → rebuilt `build_li_clay_dataset.py`; habit: locate-before-edit,
   re-derive from data when source is lost.
2. **Silent config swallowing:** pipeline hardcoded `configs/default.yaml`; passing a
   config path was ignored (Tokaj 28-sample defaults ran instead of k=50/c=0.05).
   → Fix: `sys.argv[1]` CLI patch in `run_pipeline.py`. Lesson: verify the config was
   CONSUMED (echo n_neighbors/contamination in output), never trust the command alone.
3. **Stale-data trap (twice):** validation script read the old 28-row Tokaj labels from
   `reports/usgs_li_clay/` while fresh outputs sat at `reports/` root.
   → Fix: regenerate-then-move (no branches in instructions); loud diagnostics
   (`WARNING: no known label column`, `Joined on ID: 0 of 28`) caught it.
   Lesson: gates must check CONTENT (row counts via `wc -l`, key values via grep),
   never mere existence.
4. **Label column mismatch:** pipeline writes `anomaly_label`/`anomaly_score`, not
   `is_anomaly`. → Added `anomaly_label` first in `LABEL_COLS`; derived-flag fallback
   from `anomaly_score` via metrics.json `anomaly_count`.
5. **Inverted score direction:** initial fallback assumed higher score = anomaly;
   user's heredoc inspection (top/bottom 10 vs labels) PROVED lower = anomalous.
   → Fallback flipped to `nsmallest`. Lesson: never assume score direction; verify
   against `anomaly_label` before deriving flags.
6. **District test decoded as data-density story:** 0.27× enrichment initially looked
   like method failure; investigation showed NGDB contains no ore-zone samples.
   → Pre-registered interpretation table turned a "negative result" into a finding.

## 5. PROJECT CONVENTIONS (the culture — enforce these)

- **Loud, never silent:** every fallback prints what it assumed; every drop counts rows;
  `nan` beats a fabricated number.
- **Bind path first:** always `code <path>` before pasting into an editor.
- **Locate before edit:** `Ctrl+F` for exact strings; `grep -nEi` to find line numbers
  before editing; edit whole lines, never approximations.
- **Compile gate:** `python -m py_compile <file>` after every edit — silence = pass.
- **Content gates:** `wc -l` for row counts, `grep` for key values, `head -1` for schema.
  Existence checks (`ls`) are NOT gates.
- **Pre-registered interpretation:** decide what each possible result means BEFORE
  running the test; include the "failure" branches and what they'd teach.
- **Inspect before trust:** heredocs (`python - <<'PY' … PY`) for quick data checks;
  archive evidence with `| tee reports/...`.
- **One command, one gate, then commit:** `git add` → commit with finding-carrying
  message → push. Tags pushed separately (`git push origin <tag>`).
- **Self-serve bias:** give the user the exact command/edit; if ambiguous, ask for the
  raw output rather than guessing.

## 6. v0.3 BACKLOG (prioritized, evidence-based)

1. **Cluster-aware detection** (top priority — Result 3): flag rare CLUSTERS, not only
   isolated points (HDBSCAN or LOF variant). This is the McDermitt/Mojave fix:
   dense local clusters of Li-rich samples escape LOF.
2. **Below-DL imputation:** replace ⅔·DL censoring ties (Th 1000.111, Be 3.333,
   Tl 0.333) that create artificial LOF-dense clouds.
3. **km-based windows** in `scripts/spatial_validation.py`: 0.5° ≈ 111 km made the two
   district windows identical (146 samples counted twice). Use haversine, r≈25 km.
4. **Coordinate recovery:** 1,107 samples (8.5%) lack Lat/Long; recover or document.
5. **output_dir honoring:** pipeline hardcodes `reports/`, ignores config `io.output_dir`.
6. **Targeted Li-clay datasets** (state surveys, company drills) where sampling density
   can resolve ore zones — NGDB fundamentally cannot (Result 2).
7. Optional: Mann-Whitney U on kNN distances (flagged vs missed Li-rich, n=30) to pin
   Result 3's sentence; scipy already available.
8. Data-quality: two samples share exact coordinates.

## 7. QUICK-START FOR A FRESH CONVERSATION / AGENT

bash

environment
conda activate elemental-society-analysis
cd ~/elemental-society-analysis

re-run baseline (5 s, deterministic)
python -m elemental_society_analysis.core.run_pipeline configs/experiments/usgs_li_clay_lof_k50_c05.yaml
mv -f reports/metrics.json reports/usgs_li_clay/ && mv -f reports/anomaly_labels.csv reports/usgs_li_clay/

verify (content gates)
grep -o ‘“n_samples_after_cleaning”: [0-9]*’ reports/usgs_li_clay/metrics.json # expect 13096
wc -l < reports/usgs_li_clay/anomaly_labels.csv # expect 13097

spatial validation + map
python scripts/spatial_validation.py


Reading results: global rate ≈ 5.01%; label column `anomaly_label` (−1=anomaly);
score direction: lower = more anomalous; `is_closed=false` is expected.

## 8. AGENT AUTOMATION NOTES

- The pipeline is deterministic (same data+config → same output); safe to re-run.
- All analysis scripts are runnable standalone from repo root with relative paths.
- Report artifacts are regenerable; the git-tracked source of truth is code+configs+FINDINGS.md.
- For a future agent: read FINDINGS.md for results, this file for context/conventions,
  `git log --oneline` for the change history, and never assume — verify score
  direction, data freshness, and config consumption every session.
  