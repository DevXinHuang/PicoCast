# Contributing

This project is a research MVP. Prefer small, reviewable changes that keep the
batch pipeline reproducible and the detector interpretable.

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Use optional extras only when needed:

```bash
python -m pip install -e ".[radar,notebooks]"
```

## Pull Requests

- Include tests for new detector, feature, or tracking behavior.
- Keep raw NEXRAD files and generated Parquet/GeoJSON outputs out of git.
- Document any new assumptions about Level II fields, scan timing, or target
  scoring.
- Treat balloon detections as probabilistic research signals, not operational
  aviation guidance.
