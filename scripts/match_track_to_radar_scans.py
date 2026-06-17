#!/usr/bin/env python
"""Match indexed NEXRAD scan times to a known PicoCAST expected track."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCAN_TRACK_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "scan_local_path",
    "expected_lat_deg",
    "expected_lon_deg",
    "expected_alt_km",
    "expected_alt_m",
    "time_offset_min",
    "inside_track_window",
]


def load_config(config_path: Path) -> dict:
    """Read a PicoCAST case YAML config."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def format_utc(value: pd.Timestamp) -> str:
    """Format a timestamp as UTC ISO with Z suffix."""

    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def interpolate_track_at_scans(expected_track: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    """Interpolate expected balloon position to each radar scan time."""

    track = expected_track.copy()
    scan_table = scans.copy()
    track["time_utc_dt"] = pd.to_datetime(track["time_utc"], utc=True, errors="raise")
    scan_table["scan_time_utc_dt"] = pd.to_datetime(
        scan_table["scan_time_utc"], utc=True, errors="raise"
    )

    track = track.sort_values("time_utc_dt")
    scan_table = scan_table.sort_values("scan_time_utc_dt")
    track_seconds = timestamp_seconds(track["time_utc_dt"])
    scan_seconds = timestamp_seconds(scan_table["scan_time_utc_dt"])

    output = scan_table.copy()
    for source_column, output_column in [
        ("lat_deg", "expected_lat_deg"),
        ("lon_deg", "expected_lon_deg"),
        ("alt_km", "expected_alt_km"),
        ("alt_m", "expected_alt_m"),
    ]:
        output[output_column] = np.interp(
            scan_seconds,
            track_seconds,
            track[source_column].to_numpy(dtype=float),
        )

    start = track["time_utc_dt"].iloc[0]
    end = track["time_utc_dt"].iloc[-1]
    inside = output["scan_time_utc_dt"].between(start, end, inclusive="both")
    output["inside_track_window"] = inside

    nearest_index = np.searchsorted(track_seconds, scan_seconds, side="left")
    nearest_index = np.clip(nearest_index, 0, len(track_seconds) - 1)
    previous_index = np.clip(nearest_index - 1, 0, len(track_seconds) - 1)
    choose_previous = (
        np.abs(scan_seconds - track_seconds[previous_index])
        <= np.abs(scan_seconds - track_seconds[nearest_index])
    )
    nearest_index = np.where(choose_previous, previous_index, nearest_index)
    nearest_track_seconds = track_seconds[nearest_index]
    output["time_offset_min"] = (scan_seconds - nearest_track_seconds) / 60.0

    output["scan_time_utc"] = output["scan_time_utc_dt"].map(format_utc)
    return output


def build_scan_track_matches(config_path: Path) -> pd.DataFrame:
    """Create the scan-to-track match table for a case config."""

    config = load_config(config_path)
    case_dir = config_path.parent
    case_id = str(config["case_id"])
    radar_site = str(config["nexrad"]["primary_radar_site"])
    track_path = case_dir / "expected_track.csv"
    index_path = case_dir / "nexrad" / radar_site / "index" / "nexrad_files.csv"

    expected_track = pd.read_csv(track_path)
    scans = pd.read_csv(index_path)
    scans = scans[scans["download_error"].fillna("") == ""].copy()
    matches = interpolate_track_at_scans(expected_track, scans)
    matches["case_id"] = case_id
    matches["radar_site"] = radar_site
    matches["scan_filename"] = matches["filename"]
    matches["scan_local_path"] = matches["local_path"]
    return matches[SCAN_TRACK_COLUMNS]


def timestamp_seconds(values: pd.Series) -> np.ndarray:
    """Convert UTC pandas timestamps to Unix seconds as floats."""

    return values.map(lambda value: value.timestamp()).to_numpy(dtype=float)


def write_scan_track_matches(config_path: Path) -> Path:
    """Write scan_track_matches.csv for the configured case."""

    config = load_config(config_path)
    radar_site = str(config["nexrad"]["primary_radar_site"])
    output_path = config_path.parent / "nexrad" / radar_site / "index" / "scan_track_matches.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_scan_track_matches(config_path).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()
    output_path = write_scan_track_matches(args.config)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
