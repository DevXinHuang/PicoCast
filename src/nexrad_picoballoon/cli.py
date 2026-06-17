"""Command-line interface for the batch research MVP."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from nexrad_picoballoon.decode import open_volume
from nexrad_picoballoon.detect import detect_candidates, write_candidates
from nexrad_picoballoon.evaluate import summarize_candidates, write_summary
from nexrad_picoballoon.features import extract_gate_features, write_feature_table
from nexrad_picoballoon.ingest import list_input_files, write_synthetic_volume
from nexrad_picoballoon.tracking import build_tracks, write_tracks

app = typer.Typer(help="Batch research tools for NEXRAD picoballoon candidates.")


@app.command()
def fetch(
    site: str = typer.Option(..., help="NEXRAD site identifier, for example KTLX."),
    start: str = typer.Option(..., help="ISO start time."),
    end: str = typer.Option(..., help="ISO end time. Reserved for archive retrieval hooks."),
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/raw"),
    synthetic: Annotated[
        bool, typer.Option(help="Write a synthetic fixture instead of downloading.")
    ] = False,
) -> None:
    """Fetch archive data or create a synthetic fixture."""

    del end
    start_time = _parse_time(start)
    if not synthetic:
        msg = (
            "Archive download hooks are not enabled in the MVP CLI yet. "
            "Use --synthetic for smoke tests or place Level II files in data/raw."
        )
        raise typer.BadParameter(msg)
    path = write_synthetic_volume(site=site, start=start_time, out_dir=out)
    typer.echo(f"Wrote {path}")


@app.command()
def features(
    input: Annotated[
        Path, typer.Option(help="Input directory containing local files.")
    ] = Path("data/raw"),
    out: Annotated[Path, typer.Option(help="Feature output directory.")] = Path(
        "data/features"
    ),
) -> None:
    """Decode input files and write feature tables."""

    files = list_input_files(input)
    if not files:
        raise typer.BadParameter(f"No supported input files found in {input}")
    for path in files:
        dataset = open_volume(path)
        table = extract_gate_features(dataset, source_path=path)
        out_path = out / f"{path.stem}_features.parquet"
        write_feature_table(table, out_path)
        typer.echo(f"Wrote {out_path}")


@app.command()
def detect(
    features: Annotated[Path, typer.Option(help="Feature directory.")] = Path("data/features"),
    out: Annotated[Path, typer.Option(help="Candidate output directory.")] = Path(
        "data/candidates"
    ),
    min_score: Annotated[float, typer.Option(help="Minimum gate score.")] = 0.65,
) -> None:
    """Run the interpretable detector and write detections/tracks."""

    files = sorted(features.glob("*_features.parquet"))
    if not files:
        raise typer.BadParameter(f"No feature files found in {features}")
    frames = [pd.read_parquet(path) for path in files]
    table = pd.concat(frames, ignore_index=True)
    candidates = detect_candidates(table, min_score=min_score)
    candidate_path = out / "candidate_detections.parquet"
    write_candidates(candidates, candidate_path)
    tracks = build_tracks(candidates)
    track_path = out / "candidate_tracks.parquet"
    write_tracks(tracks, track_path)
    typer.echo(f"Wrote {candidate_path}")
    typer.echo(f"Wrote {track_path}")


@app.command()
def evaluate(
    candidates: Annotated[Path, typer.Option(help="Candidate output directory.")] = Path(
        "data/candidates"
    ),
    truth: Annotated[Path, typer.Option(help="Optional truth data directory.")] = Path(
        "data/truth"
    ),
) -> None:
    """Write a Markdown evaluation summary."""

    candidate_path = candidates / "candidate_detections.parquet"
    if not candidate_path.exists():
        raise typer.BadParameter(f"Missing {candidate_path}")
    table = pd.read_parquet(candidate_path)
    summary = summarize_candidates(table, truth_dir=truth)
    out_path = candidates / "evaluation_summary.md"
    write_summary(summary, out_path)
    typer.echo(f"Wrote {out_path}")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
