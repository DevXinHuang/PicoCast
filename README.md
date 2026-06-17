# NEXRAD Picoballoon Research MVP

This repository is a Python-first batch research scaffold for detecting and
tracking weak picoballoon-like targets in archived NEXRAD Level II radar data.
It follows the project brief in `Practical Plan for Detecting and Tracking
Picoballoons with NEXRAD in the United States.pdf`: use Level II as the source
of truth, start with interpretable rejection and persistence logic, validate on
archive data, and defer real-time operations until the science workflow is
stable.

## Quickstart

Use Python 3.11 or newer. On this machine, `/usr/bin/python3` is Python 3.9.6,
so create the virtual environment with an explicit Python 3.12 interpreter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `python3.12` is not on your path, use another Python 3.11+ interpreter.

Run the synthetic smoke workflow:

```bash
nexrad-pico fetch --site KTEST --start 2026-01-01T00:00:00Z --end 2026-01-01T00:10:00Z --out data/raw --synthetic
nexrad-pico features --input data/raw --out data/features
nexrad-pico detect --features data/features --out data/candidates
nexrad-pico evaluate --candidates data/candidates --truth data/truth
```

Run tests:

```bash
pytest
ruff check .
```

## Repository Layout

- `src/nexrad_picoballoon/` contains ingest, decode, feature extraction,
  detection, tracking, wind context, evaluation, and CLI code.
- `tests/` contains synthetic fixture tests that do not download radar data.
- `docs/` contains data, roadmap, limitations, and workflow notes.
- `notebooks/` contains starter notebooks for sample volume exploration and
  candidate replay.
- `data/` is for local raw radar files, derived features, candidate outputs, and
  optional truth data. Its contents are ignored by git.

## Data Strategy

The MVP uses NEXRAD Level II data as the primary input. Level III products,
including VAD wind profile and hydrometeor classification, are useful as
context layers but are not part of the first detector's source of truth.

Real Level II decoding is routed through optional radar dependencies such as
`xradar`, Py-ART, and wradlib. Install them when working with operational radar
files:

```bash
python -m pip install -e ".[dev,radar,notebooks]"
```

The default tests and CLI smoke path use a tiny synthetic xarray fixture so the
repository remains fast to clone, test, and review.

## Detection Approach

The first detector is intentionally interpretable. It builds per-gate features,
rejects obvious weather and clutter, scores weak non-stationary targets with
dual-pol texture context, groups compact detections, and links candidates across
scan times. Supervised ML, multi-radar fusion, PostGIS, and real-time SNS/S3
ingest are follow-on phases.

## Validation Cases

The first validation case is `cases/k7uaz_20260322`, built from K7UAZ
launch-day telemetry on March 22, 2026. Generate the clean expected track and
manifest from the raw CSV:

```bash
python scripts/build_case_from_csv.py cases/k7uaz_20260322/config.yaml
```

Create a track preview plot:

```bash
python scripts/plot_case_track.py cases/k7uaz_20260322/config.yaml
```

List the KEMX Level II files for the known flight window without downloading:

```bash
python scripts/download_nexrad_case.py cases/k7uaz_20260322/config.yaml --dry-run
```

Download and index the KEMX Level II files:

```bash
python scripts/download_nexrad_case.py cases/k7uaz_20260322/config.yaml
```

Match each indexed radar scan time to the expected balloon position:

```bash
python scripts/match_track_to_radar_scans.py cases/k7uaz_20260322/config.yaml
```

Render quick reflectivity preview plots with expected-position overlays:

```bash
python scripts/preview_nexrad_case.py cases/k7uaz_20260322/config.yaml
```

Preview plotting requires Py-ART, so install the radar extra first:

```bash
python -m pip install -e ".[dev,radar]"
```

The validation workflow uses known balloon time/position/altitude before any
blind detection. Later NEXRAD Level II comparisons should ask whether compact
non-weather radar candidates appear near these expected balloon positions.

## NOAA Data Attribution

When distributing derived results from NOAA/NEXRAD data, attribute NOAA/NWS as
the source for original radar data and do not imply NOAA endorsement. Do not
present modified or derived products as unaltered NOAA data.
