#!/usr/bin/env python
"""Build a PicoCAST validation case from launch-day telemetry CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

EXPECTED_COLUMNS = [
    "case_id",
    "point_id",
    "time_local",
    "time_utc",
    "lat_deg",
    "lon_deg",
    "alt_km",
    "alt_m",
    "maidenhead_grid",
    "vertical_speed_m_min",
    "speed_kmh",
    "computed_speed_kmh",
    "voltage_v",
    "temperature_c",
    "rx_reports",
    "avg_rx_frequency_hz",
    "max_rx_distance_km",
    "max_snr_db",
    "source_row",
]

REQUIRED_SOURCE_COLUMNS = {
    "#": "source_row",
    "Local Time": "time_local",
    "Grid": "maidenhead_grid",
    "Lat (°)": "lat_deg",
    "Lon (°)": "lon_deg",
    "Altitude (km)": "alt_km",
    "Vertical Speed (m/min)": "vertical_speed_m_min",
    "Speed (km/h)": "speed_kmh",
    "Computed Speed (km/h)": "computed_speed_kmh",
    "Voltage (V)": "voltage_v",
    "Temperature (°C)": "temperature_c",
    "# RX Reports": "rx_reports",
    "Average RX Frequency (Hz)": "avg_rx_frequency_hz",
    "Max RX Distance (km)": "max_rx_distance_km",
    "Max SNR (dB)": "max_snr_db",
}

NUMERIC_COLUMNS = [
    "lat_deg",
    "lon_deg",
    "alt_km",
    "vertical_speed_m_min",
    "speed_kmh",
    "computed_speed_kmh",
    "voltage_v",
    "temperature_c",
    "rx_reports",
    "avg_rx_frequency_hz",
    "max_rx_distance_km",
    "max_snr_db",
    "source_row",
]


def load_config(config_path: Path) -> dict:
    """Read a case YAML config."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def build_expected_track(config_path: Path) -> pd.DataFrame:
    """Build and validate the clean expected track table."""

    config = load_config(config_path)
    case_dir = config_path.parent
    case_id = str(config["case_id"])
    timezone_name = str(config.get("timezone", "America/Phoenix"))
    source_csv = case_dir / str(config["source_csv"])
    local_tz = ZoneInfo(timezone_name)

    raw = pd.read_csv(source_csv)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")

    track = raw.rename(columns=REQUIRED_SOURCE_COLUMNS)[list(REQUIRED_SOURCE_COLUMNS.values())]
    for column in NUMERIC_COLUMNS:
        track[column] = pd.to_numeric(track[column], errors="coerce")

    local_times = pd.to_datetime(track["time_local"], errors="raise")
    localized = local_times.dt.tz_localize(local_tz)
    track["time_local"] = localized.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    track["time_utc"] = localized.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    track["case_id"] = case_id
    track["point_id"] = [f"{case_id}_{idx:03d}" for idx in range(1, len(track) + 1)]
    track["alt_m"] = track["alt_km"] * 1000.0

    track = track[EXPECTED_COLUMNS]
    track = track.sort_values("time_utc").reset_index(drop=True)
    validate_expected_track(track)
    return track


def validate_expected_track(track: pd.DataFrame) -> None:
    """Run basic validation checks for the launch-day expected track."""

    if len(track) != 12:
        raise ValueError(f"Expected 12 telemetry rows, found {len(track)}.")
    if track["lat_deg"].isna().any() or track["lon_deg"].isna().any():
        raise ValueError("Latitude and longitude must not be missing.")
    if not (track["alt_m"].round(6) == (track["alt_km"] * 1000.0).round(6)).all():
        raise ValueError("Altitude meters must equal altitude kilometers times 1000.")
    parsed_utc = pd.to_datetime(track["time_utc"], utc=True, errors="coerce")
    if parsed_utc.isna().any():
        raise ValueError("time_utc contains invalid timestamps.")
    if not parsed_utc.is_monotonic_increasing:
        raise ValueError("Rows must be sorted chronologically.")


def build_manifest(config_path: Path, track: pd.DataFrame) -> pd.DataFrame:
    """Build the one-row case manifest from the expected track."""

    config = load_config(config_path)
    utc_times = pd.to_datetime(track["time_utc"], utc=True)
    row = {
        "case_id": config["case_id"],
        "date_local": config["date_local"],
        "start_time_local": track["time_local"].iloc[0],
        "end_time_local": track["time_local"].iloc[-1],
        "start_time_utc": utc_times.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time_utc": utc_times.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_points": len(track),
        "lat_min": track["lat_deg"].min(),
        "lat_max": track["lat_deg"].max(),
        "lon_min": track["lon_deg"].min(),
        "lon_max": track["lon_deg"].max(),
        "alt_min_km": track["alt_km"].min(),
        "alt_max_km": track["alt_km"].max(),
        "primary_radar_site": config["primary_radar_site"],
        "source_csv": config["source_csv"],
    }
    return pd.DataFrame([row])


def write_case_outputs(config_path: Path) -> tuple[Path, Path]:
    """Generate expected_track.csv and manifest.csv for a case."""

    case_dir = config_path.parent
    track = build_expected_track(config_path)
    expected_track_path = case_dir / "expected_track.csv"
    manifest_path = case_dir / "manifest.csv"
    track.to_csv(expected_track_path, index=False)
    build_manifest(config_path, track).to_csv(manifest_path, index=False)
    return expected_track_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()
    expected_track_path, manifest_path = write_case_outputs(args.config)
    print(f"Wrote {expected_track_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
