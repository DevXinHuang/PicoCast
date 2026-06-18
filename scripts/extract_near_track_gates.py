#!/usr/bin/env python
"""Extract radar gates near the expected K7UAZ balloon track."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    SEARCH_WINDOWS,
    candidates_dir,
    case_id_from_config,
    horizontal_distance_km,
    load_config,
    matches_path,
    radar_site_from_config,
)

NEAR_TRACK_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "search_window",
    "expected_lat_deg",
    "expected_lon_deg",
    "expected_alt_m",
    "gate_lat_deg",
    "gate_lon_deg",
    "gate_alt_m",
    "horizontal_distance_km",
    "vertical_distance_m",
    "sweep_number",
    "elevation_deg",
    "azimuth_deg",
    "range_km",
    "reflectivity_dbz",
    "velocity_ms",
    "spectrum_width_ms",
    "cross_correlation_ratio",
    "field_names_available",
    "source_file",
]

SUMMARY_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "search_window",
    "expected_lat_deg",
    "expected_lon_deg",
    "expected_alt_m",
    "n_nearby_gates",
    "n_valid_reflectivity_gates",
    "max_reflectivity_dbz",
    "mean_reflectivity_dbz",
    "p95_reflectivity_dbz",
    "max_velocity_ms",
    "min_velocity_ms",
    "mean_cross_correlation_ratio",
    "min_horizontal_distance_km",
    "min_vertical_distance_m",
    "has_possible_return",
    "notes",
]


def sweep_numbers_by_ray(radar) -> np.ndarray:
    """Return sweep number for each ray in a Py-ART radar object."""

    sweep_numbers = np.zeros(radar.nrays, dtype=int)
    starts = radar.sweep_start_ray_index["data"]
    ends = radar.sweep_end_ray_index["data"]
    for sweep_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
        sweep_numbers[int(start) : int(end) + 1] = sweep_number
    return sweep_numbers


def field_values(
    radar,
    field_name: str,
    ray_index: np.ndarray,
    gate_index: np.ndarray,
) -> np.ndarray:
    """Extract field values as floats, using NaN for masked/missing data."""

    if field_name not in radar.fields:
        return np.full(ray_index.shape, np.nan, dtype=float)
    values = radar.fields[field_name]["data"][ray_index, gate_index]
    if isinstance(values, ma.MaskedArray):
        return values.astype(float).filled(np.nan)
    return np.asarray(values, dtype=float)


def extract_scan_near_track_gates(
    radar,
    match_row: pd.Series,
    *,
    case_id: str,
    radar_site: str,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """Extract near-track gates and summaries for one radar scan."""

    gate_lat = radar.gate_latitude["data"]
    gate_lon = radar.gate_longitude["data"]
    gate_alt = radar.gate_altitude["data"]
    expected_lat = float(match_row["expected_lat_deg"])
    expected_lon = float(match_row["expected_lon_deg"])
    expected_alt_m = float(match_row["expected_alt_m"])
    horizontal_distance = horizontal_distance_km(gate_lat, gate_lon, expected_lat, expected_lon)
    vertical_distance = np.abs(gate_alt - expected_alt_m)

    sweeps = sweep_numbers_by_ray(radar)
    field_names = ";".join(sorted(radar.fields.keys()))
    azimuth = radar.azimuth["data"]
    elevation = radar.elevation["data"]
    ranges_km = radar.range["data"] / 1000.0
    gate_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for window_name, window in SEARCH_WINDOWS.items():
        mask = (
            (horizontal_distance <= window["horizontal_radius_km"])
            & (vertical_distance <= window["vertical_radius_km"] * 1000.0)
        )
        ray_index, gate_index = np.where(mask)
        reflectivity = field_values(radar, "reflectivity", ray_index, gate_index)
        velocity = field_values(radar, "velocity", ray_index, gate_index)
        spectrum_width = field_values(radar, "spectrum_width", ray_index, gate_index)
        rhohv = field_values(radar, "cross_correlation_ratio", ray_index, gate_index)

        frame = pd.DataFrame(
            {
                "case_id": case_id,
                "radar_site": radar_site,
                "scan_time_utc": match_row["scan_time_utc"],
                "scan_filename": match_row["scan_filename"],
                "search_window": window_name,
                "expected_lat_deg": expected_lat,
                "expected_lon_deg": expected_lon,
                "expected_alt_m": expected_alt_m,
                "gate_lat_deg": gate_lat[ray_index, gate_index],
                "gate_lon_deg": gate_lon[ray_index, gate_index],
                "gate_alt_m": gate_alt[ray_index, gate_index],
                "horizontal_distance_km": horizontal_distance[ray_index, gate_index],
                "vertical_distance_m": vertical_distance[ray_index, gate_index],
                "sweep_number": sweeps[ray_index],
                "elevation_deg": elevation[ray_index],
                "azimuth_deg": azimuth[ray_index],
                "range_km": ranges_km[gate_index],
                "reflectivity_dbz": reflectivity,
                "velocity_ms": velocity,
                "spectrum_width_ms": spectrum_width,
                "cross_correlation_ratio": rhohv,
                "field_names_available": field_names,
                "source_file": match_row["scan_local_path"],
            },
            columns=NEAR_TRACK_COLUMNS,
        )
        gate_frames.append(frame)
        summary_rows.append(
            summarize_scan_window(
                frame,
                case_id=case_id,
                radar_site=radar_site,
                match_row=match_row,
                search_window=window_name,
            )
        )

    return gate_frames, summary_rows


def summarize_scan_window(
    frame: pd.DataFrame,
    *,
    case_id: str,
    radar_site: str,
    match_row: pd.Series,
    search_window: str,
) -> dict:
    """Summarize one scan/search-window extraction."""

    valid_reflectivity = frame["reflectivity_dbz"].dropna()
    valid_velocity = frame["velocity_ms"].dropna()
    valid_rhohv = frame["cross_correlation_ratio"].dropna()
    notes: list[str] = []
    if not is_truthy(match_row.get("inside_track_window", True)):
        notes.append("outside telemetry window")
    if frame.empty:
        notes.append("no gates in search volume")
    elif valid_reflectivity.empty:
        notes.append("nearby gates but reflectivity masked")

    return {
        "case_id": case_id,
        "radar_site": radar_site,
        "scan_time_utc": match_row["scan_time_utc"],
        "scan_filename": match_row["scan_filename"],
        "search_window": search_window,
        "expected_lat_deg": float(match_row["expected_lat_deg"]),
        "expected_lon_deg": float(match_row["expected_lon_deg"]),
        "expected_alt_m": float(match_row["expected_alt_m"]),
        "n_nearby_gates": int(len(frame)),
        "n_valid_reflectivity_gates": int(len(valid_reflectivity)),
        "max_reflectivity_dbz": value_or_nan(valid_reflectivity.max()),
        "mean_reflectivity_dbz": value_or_nan(valid_reflectivity.mean()),
        "p95_reflectivity_dbz": value_or_nan(valid_reflectivity.quantile(0.95)),
        "max_velocity_ms": value_or_nan(valid_velocity.max()),
        "min_velocity_ms": value_or_nan(valid_velocity.min()),
        "mean_cross_correlation_ratio": value_or_nan(valid_rhohv.mean()),
        "min_horizontal_distance_km": value_or_nan(frame["horizontal_distance_km"].min()),
        "min_vertical_distance_m": value_or_nan(frame["vertical_distance_m"].min()),
        "has_possible_return": bool(len(valid_reflectivity) > 0),
        "notes": "; ".join(notes),
    }


def value_or_nan(value) -> float:
    """Return a float or NaN for empty aggregate values."""

    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except TypeError:
        return float("nan")


def is_truthy(value) -> bool:
    """Parse boolean-ish CSV values."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def extract_case_gates(config_path: Path, radar_site: str | None = None) -> tuple[Path, Path]:
    """Extract near-track gates for one case/radar site."""

    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            'Gate extraction requires Py-ART. Install it with: python -m pip install -e ".[radar]"'
        ) from exc

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_id = case_id_from_config(config)
    matches = pd.read_csv(matches_path(config_path, site))
    output_dir = candidates_dir(config_path, site)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for _, match_row in matches.iterrows():
        local_path = Path(str(match_row["scan_local_path"]))
        if not local_path.exists():
            summary_rows.extend(missing_scan_summary(case_id, site, match_row))
            continue
        radar = pyart.io.read_nexrad_archive(str(local_path))
        scan_frames, scan_summaries = extract_scan_near_track_gates(
            radar,
            match_row,
            case_id=case_id,
            radar_site=site,
        )
        gate_frames.extend(scan_frames)
        summary_rows.extend(scan_summaries)
        print(f"Processed {match_row['scan_filename']}")

    gates = (
        pd.concat(gate_frames, ignore_index=True)
        if gate_frames
        else pd.DataFrame(columns=NEAR_TRACK_COLUMNS)
    )
    summaries = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    gates_path = output_dir / "near_track_gates.csv"
    summary_path = output_dir / "scan_gate_summary.csv"
    gates.to_csv(gates_path, index=False)
    summaries.to_csv(summary_path, index=False)
    print(f"Wrote {gates_path}")
    print(f"Wrote {summary_path}")
    return gates_path, summary_path


def missing_scan_summary(case_id: str, radar_site: str, match_row: pd.Series) -> list[dict]:
    """Return summary rows for a missing scan file."""

    rows = []
    for window_name in SEARCH_WINDOWS:
        rows.append(
            {
                "case_id": case_id,
                "radar_site": radar_site,
                "scan_time_utc": match_row["scan_time_utc"],
                "scan_filename": match_row["scan_filename"],
                "search_window": window_name,
                "expected_lat_deg": float(match_row["expected_lat_deg"]),
                "expected_lon_deg": float(match_row["expected_lon_deg"]),
                "expected_alt_m": float(match_row["expected_alt_m"]),
                "n_nearby_gates": 0,
                "n_valid_reflectivity_gates": 0,
                "max_reflectivity_dbz": np.nan,
                "mean_reflectivity_dbz": np.nan,
                "p95_reflectivity_dbz": np.nan,
                "max_velocity_ms": np.nan,
                "min_velocity_ms": np.nan,
                "mean_cross_correlation_ratio": np.nan,
                "min_horizontal_distance_km": np.nan,
                "min_vertical_distance_m": np.nan,
                "has_possible_return": False,
                "notes": "missing scan file",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    args = parser.parse_args()
    extract_case_gates(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
