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

End-of-day candidate triage for KEMX:

```bash
python scripts/extract_near_track_gates.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
python scripts/cluster_near_track_gates.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
python scripts/score_near_track_candidates.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
python scripts/plot_top_candidates.py cases/k7uaz_20260322/config.yaml --radar-site KEMX --top-n 10
python scripts/write_candidate_report.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
```

This workflow produces near-track radar candidate CSVs, plots, and a cautious
human-readable report under `cases/k7uaz_20260322/outputs/candidates/KEMX/`.
It is for candidate inspection only and does not confirm a balloon association.

### Geospatial Mapping

Export GeoJSON layers and create interactive HTML maps for visual inspection of
balloon track, Maidenhead grid-square regions, radar candidates, and KEMX radar
geometry on real basemaps:

```bash
python scripts/export_case_geojson.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
python scripts/make_interactive_map.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
python scripts/make_candidate_validation_map.py cases/k7uaz_20260322/config.yaml --radar-site KEMX --rank 1
python scripts/make_sequence_map.py cases/k7uaz_20260322/config.yaml --radar-site KEMX --ranks 1 3 6 7 9
```

Open the maps in your browser:

```bash
open cases/k7uaz_20260322/outputs/maps/interactive_candidate_map.html
open cases/k7uaz_20260322/outputs/maps/rank_01_validation_map.html
```

> **Note:** Balloon position is estimated from Maidenhead grid-square centers,
> not exact GPS. Candidate offsets are relative to the grid-center estimate.

### Altitude-First Validation

Because the balloon horizontal position comes from coarse Maidenhead grid squares (~5-7 km wide), but the telemetry altitude is precise, we run an altitude-first validation to rescore candidates:

```bash
python scripts/validate_altitude_consistency.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
```

This promotes candidates whose radar gate altitude strongly matches the balloon telemetry altitude, generating updated scores, histograms, and mismatch plots in `outputs/candidates/KEMX/altitude_validation/`.

### Candidate Trajectory Fitting

Finally, test whether the top altitude-consistent candidates can form a physically plausible, continuous trajectory without impossible jumps:

```bash
python scripts/fit_candidate_path.py cases/k7uaz_20260322/config.yaml --radar-site KEMX --start-time 2026-03-22T20:00:00Z --end-time 2026-03-22T20:30:00Z
python scripts/make_path_dashboard.py cases/k7uaz_20260322/config.yaml --radar-site KEMX
```

This outputs a smoothed trajectory GeoJSON and an interactive dashboard integrating the GIS map and an altitude vs. time progression chart.

## NOAA Data Attribution

When distributing derived results from NOAA/NEXRAD data, attribute NOAA/NWS as
the source for original radar data and do not imply NOAA endorsement. Do not
present modified or derived products as unaltered NOAA data.
