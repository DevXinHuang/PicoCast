import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path("scripts/build_case_from_csv.py")
spec = importlib.util.spec_from_file_location("build_case_from_csv", SCRIPT_PATH)
assert spec and spec.loader
build_case_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_case_module)
build_expected_track = build_case_module.build_expected_track
build_manifest = build_case_module.build_manifest


def test_k7uaz_expected_track_basic_checks():
    config_path = Path("cases/k7uaz_20260322/config.yaml")
    track = build_expected_track(config_path)

    assert list(track.columns) == [
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
    assert len(track) == 12
    assert not track[["lat_deg", "lon_deg"]].isna().any().any()
    assert (track["alt_m"] == track["alt_km"] * 1000.0).all()
    assert pd.to_datetime(track["time_utc"], utc=True).is_monotonic_increasing
    assert track["time_utc"].iloc[0] == "2026-03-22T19:18:00Z"


def test_k7uaz_manifest_summary():
    config_path = Path("cases/k7uaz_20260322/config.yaml")
    track = build_expected_track(config_path)
    manifest = build_manifest(config_path, track)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["case_id"] == "k7uaz_20260322"
    assert row["n_points"] == 12
    assert row["primary_radar_site"] == "KEMX"
    assert row["start_time_local"] == "2026-03-22T12:18:00-0700"
    assert row["start_time_utc"] == "2026-03-22T19:18:00Z"
    assert row["lat_min"] == 32.23
    assert row["lon_max"] == -109.79
