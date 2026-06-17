# Roadmap

## Alpha: Data Foundation

- Local fixture and Level II file discovery.
- Decode abstraction around xarray-compatible radar datasets.
- Per-gate feature schema and metadata sanity checks.
- Notebook quicklooks.

## Beta: First Detector

- Rejector-first candidate scoring.
- Sweep-aware feature extraction.
- Candidate grouping and simple track association.
- Analyst replay notebook.

## Gamma: Validation

- Synthetic injection harness.
- Archive benchmark sets with hard negatives.
- Metrics for detection probability, false alarms per radar-hour, track
  continuity, latency, and position error.

## Delta: Operations

- Real-time S3/SNS ingest.
- Wind-aware Kalman or IMM tracking.
- Multi-radar association.
- Alert scoring, dashboard UI, and deployment runbook.
