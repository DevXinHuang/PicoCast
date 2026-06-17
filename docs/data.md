# Data Workflow

## Primary Input

Use NEXRAD Level II data as the source of truth for weak-target research. Level
II provides the base moments and dual-pol variables needed for custom filtering,
temporal association, and feature extraction.

## Local Layout

- `data/raw/`: local Level II files or synthetic fixtures.
- `data/features/`: derived per-gate Parquet feature tables.
- `data/candidates/`: candidate detections, tracks, and evaluation summaries.
- `data/truth/`: optional release telemetry or curated labels.

Only `.gitkeep` placeholders are committed. Raw radar files and generated
outputs stay local.

## Optional Radar Dependencies

Install `.[radar]` before decoding real Level II files. The MVP does not parse
Level II binary data directly; it routes decoding through mature radar
libraries and keeps custom parsers out of scope.
