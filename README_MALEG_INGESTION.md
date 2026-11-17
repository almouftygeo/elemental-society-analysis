# MALEG Ingestion Pipeline

A reproducible, testable data-ingestion pipeline for the MALEG project:
three sources (sensors, laboratory, operational events) → canonical models →
validation → immutable raw storage → time-aware integration → ML-ready features.

## Architecture

```text
Source-specific data
  Sensors (T, P, flow, pH, conductivity)
  Laboratory (ICP-MS, IC, GC)
  Events (sampling, maintenance, dosing)
        ↓
Source-specific adapters
  SensorAdapter · LabCSVAdapter · EventAdapter
        ↓
Canonical data model
  SensorRecord · LabResult · OperationalEvent
        ↓
Quality validation
  VALID / SUSPECT / REJECTED (+ reason)
        ↓
Immutable raw storage (SQLite, append-only)
        ↓
Time-aware integration
  as-of joins · freshness · rolling features
        ↓
ML-ready feature table
```

## Key design decisions

- **Canonical data model** — one internal contract, independent of source format
- **Immutable records** — frozen dataclasses + frozen nested payloads
- **Censoring preserved** — `<0.5` stays `BELOW_DL`, not zero
- **Four timestamps** — event, sample, availability, ingestion (leakage prevention)
- **Append-only raw DB** — no UPDATE/DELETE; corrections are versioned
- **Time-aware joins** — `merge_asof` with `direction='backward'`
- **Shift-before-roll** — rolling windows use `shift(1)` to avoid look-ahead

## How to run

```bash
# 1. Activate venv
source .venv/Scripts/activate

# 2. Run tests
python -m pytest tests/ -v

# 3. Run the pipeline end-to-end
python -m elemental_society_analysis.core.run_maleg_pipeline \
  --config configs/maleg_run_001.yaml
```

## Test coverage

- Canonical model (9 tests): UTC, immutability, censoring rules
- SensorAdapter (4 tests): valid JSON, missing fields, string input, naive timestamps
- LabCSVAdapter (4 tests): censoring, normal values, missing, naive timestamps
- EventAdapter (4 tests): valid events, missing fields, string input, naive timestamps
- Quality + storage (5 tests): limits, censoring, append-only, idempotent inserts
- Feature builder (4 tests): no future leakage, fresh joins, dosing timing, rolling
- Pipeline runner (1 test): end-to-end smoke test

**Total: 31 tests, all passing.**

## Interview explanation (30 seconds)

> “I built a multi-source ingestion pipeline for MALEG: three adapters convert
> sensor, laboratory, and event data into canonical records. I enforce scientific
> integrity—censored lab values preserved, four distinct timestamps, immutable
> records—and prevent leakage with time-aware as-of joins and shift-before-roll
> features. Everything is validated, stored immutably, and tied together by a
> YAML-driven pipeline runner with end-to-end tests.”
