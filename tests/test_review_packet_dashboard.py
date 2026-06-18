"""Tests for the presentation-ready review dashboard data layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.make_review_packet_dashboard import (
    build_dashboard_data,
    compute_candidate_segment_speeds,
    compute_grid_center_speeds,
    render_dashboard_html,
    telemetry_gap_flags,
    write_dashboard,
)


def test_grid_center_speed_calculation():
    track = pd.DataFrame(
        {
            "time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T21:00:00Z"],
            "lat_deg": [0.0, 0.0],
            "lon_deg": [0.0, 1.0],
        }
    )

    result = compute_grid_center_speeds(track)

    assert result.loc[0, "grid_center_speed_kmh"] != result.loc[0, "grid_center_speed_kmh"]
    assert result.loc[1, "grid_center_speed_kmh"] == pytest.approx(111.2, abs=0.3)
    assert result.loc[1, "is_gap_from_previous"]


def test_candidate_segment_speed_calculation():
    points = pd.DataFrame(
        {
            "tracklet_id": ["KEMX_T001", "KEMX_T001"],
            "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T21:00:00Z"],
            "cluster_lat_deg": [0.0, 0.0],
            "cluster_lon_deg": [0.0, 1.0],
        }
    )

    result = compute_candidate_segment_speeds(points)

    assert result.loc[1, "candidate_segment_speed_kmh"] == pytest.approx(111.2, abs=0.3)


def test_telemetry_gap_flags():
    track = pd.DataFrame(
        {
            "time_utc": [
                "2026-03-22T19:48:00Z",
                "2026-03-22T21:18:00Z",
            ],
            "maidenhead_grid": ["DM42pi", "DM42uh"],
        }
    )

    gaps = telemetry_gap_flags(track)

    assert len(gaps) == 1
    assert gaps[0]["gap_min"] == 90.0
    assert gaps[0]["start_grid"] == "DM42pi"


def write_dashboard_fixture(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    discovery_dir = case_dir / "outputs" / "discovery"
    review_dir.mkdir(parents=True)

    config = {
        "case_id": "test_case",
        "case_name": "Test Case",
        "date_local": "2026-03-22",
        "timezone": "America/Phoenix",
        "mapping": {"range_rings_km": [50, 100]},
        "radar_sites": {
            "KEMX": {"lat": 31.89365, "lon": -110.63025, "alt_m": 1621.2},
            "KIWA": {"lat": 33.28923, "lon": -111.66991, "alt_m": 434.6},
        },
    }
    config_path = case_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pd.DataFrame(
        {
            "time_utc": [
                "2026-03-22T19:48:00Z",
                "2026-03-22T21:18:00Z",
            ],
            "lat_deg": [32.35, 32.31],
            "lon_deg": [-110.71, -110.29],
            "alt_m": [4280.0, 11280.0],
            "speed_kmh": [30.0, 33.0],
            "vertical_speed_m_min": [62.0, 78.0],
            "maidenhead_grid": ["DM42pi", "DM42uh"],
        }
    ).to_csv(case_dir / "expected_track.csv", index=False)

    pd.DataFrame(
        [
            {
                "review_rank": 1,
                "item_type": "cross_radar_association",
                "item_id": "A001",
                "radar_sites": "KEMX;KIWA",
                "tracklet_ids": "KEMX_T004;KIWA_T001",
                "family_ids": "KEMX_F001;KIWA_F001",
                "review_priority_score": 0.86,
                "review_reason": "strong cross-radar candidate association",
                "plot_path": "plots/review_rank_01_A001.png",
            }
        ]
    ).to_csv(review_dir / "tracklet_review_queue.csv", index=False)

    pd.DataFrame(
        [
            {
                "association_id": "A001",
                "time_overlap_min": 8.8,
                "median_horizontal_difference_km": 9.8,
                "median_altitude_difference_m": 144.0,
                "association_label": "strong_cross_radar_candidate",
            }
        ]
    ).to_csv(review_dir / "cross_radar_review_queue.csv", index=False)

    pd.DataFrame(
        {
            "tracklet_id": ["KEMX_T004", "KEMX_T004", "KIWA_T001", "KIWA_T001"],
            "radar_site": ["KEMX", "KEMX", "KIWA", "KIWA"],
            "scan_time_utc": [
                "2026-03-22T20:00:00Z",
                "2026-03-22T20:10:00Z",
                "2026-03-22T20:00:00Z",
                "2026-03-22T20:10:00Z",
            ],
            "cluster_lat_deg": [32.34, 32.35, 32.32, 32.33],
            "cluster_lon_deg": [-110.70, -110.65, -110.82, -110.80],
            "cluster_alt_m": [5200.0, 5900.0, 5100.0, 5800.0],
            "expected_alt_m": [5200.0, 5900.0, 5200.0, 5900.0],
            "max_reflectivity_dbz": [-5.0, -4.5, 2.0, 3.0],
            "distance_to_track_corridor_km": [2.0, 3.0, 4.0, 5.0],
            "nearest_grid": ["DM42pi", "DM42pi", "DM42pi", "DM42pi"],
        }
    ).to_csv(discovery_dir / "plausible_tracklet_points.csv", index=False)

    return config_path


def test_dashboard_data_shape_and_default_rank(tmp_path):
    config_path = write_dashboard_fixture(tmp_path)

    data = build_dashboard_data(config_path, top_n=1)

    assert data["case"]["top_candidate"] == "A001"
    assert len(data["review_items"]) == 1
    assert data["review_items"][0]["tracklet_ids"] == ["KEMX_T004", "KIWA_T001"]
    assert data["review_items"][0]["stats"]["median_abs_vertical_mismatch_m"] >= 0
    assert data["telemetry_gaps"][0]["gap_min"] == 90.0
    assert data["review_items"][0]["points"][0]["expected_speed_kmh"] is not None


def test_dashboard_html_uses_cautious_language(tmp_path):
    config_path = write_dashboard_fixture(tmp_path)
    data = build_dashboard_data(config_path, top_n=1)

    html = render_dashboard_html(data)

    assert "Candidate radar feature, not a detection claim" in html
    assert "Horizontal balloon position comes from Maidenhead grid centers, not GPS" in html
    assert "Speed comparison is approximate" in html
    assert "detected balloon" not in html
    assert "confirmed detection" not in html


def test_dashboard_writer_creates_obvious_index(tmp_path):
    config_path = write_dashboard_fixture(tmp_path)

    dashboard_path, data_path, index_path = write_dashboard(config_path, top_n=1)

    assert dashboard_path.name == "review_packet_dashboard.html"
    assert data_path.name == "review_packet_dashboard_data.json"
    assert index_path.name == "index.html"
    assert index_path.exists()
    assert "K7UAZ Candidate Radar Feature Review" in index_path.read_text(encoding="utf-8")
