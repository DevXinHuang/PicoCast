#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 9: Unit tests for regional discovery mode algorithms."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from scripts.discover_regional_balloon_like_clusters import (
    distance_to_polyline_km,
)
from scripts.download_regional_nexrad import (
    calculate_beam_elevation_deg,
)
from scripts.link_regional_tracklets import (
    segment_bearing_deg,
)
from scripts.match_regional_scans_to_telemetry import (
    interpolate_balloon_telemetry,
)


def test_visibility_elevation_calculation():
    # Target height same as radar: elevation should be 0 (except curvature correction)
    el = calculate_beam_elevation_deg(1000.0, 1000.0, 10000.0)
    assert el < 0.0  # curvature pushes target down relative to local horizontal
    
    # Target height much higher: elevation should be positive
    el = calculate_beam_elevation_deg(1000.0, 6000.0, 100000.0)
    assert 0.0 < el < 10.0


def test_distance_to_polyline():
    # Define a simple track segment from (32.0, -110.0) to (32.0, -109.0)
    track_lats = np.array([32.0, 32.0])
    track_lons = np.array([-110.0, -109.0])
    
    # Gate exactly on the segment
    g_lats = np.array([32.0])
    g_lons = np.array([-109.5])
    
    d = distance_to_polyline_km(g_lats, g_lons, track_lats, track_lons)
    assert abs(d[0]) < 0.1
    
    # Gate off the segment (perpendicular distance)
    # 0.1 degrees north of the midpoint
    g_lats = np.array([32.1])
    g_lons = np.array([-109.5])
    d = distance_to_polyline_km(g_lats, g_lons, track_lats, track_lons)
    # 0.1 deg lat is approx 11.13 km
    assert 11.0 < d[0] < 11.3


def test_segment_bearing():
    # Due North
    b = segment_bearing_deg(32.0, -110.0, 33.0, -110.0)
    assert abs(b - 0.0) < 1.0 or abs(b - 360.0) < 1.0
    
    # Due East
    b = segment_bearing_deg(32.0, -110.0, 32.0, -109.0)
    assert abs(b - 90.0) < 1.0


def test_telemetry_interpolation():
    t0 = datetime(2026, 3, 22, 19, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 22, 19, 10, 0, tzinfo=UTC)
    
    points = [
        {"time_utc": t0, "lat": 32.0, "lon": -110.0, "alt_m": 5000.0, "maidenhead_grid": "DM42"},
        {"time_utc": t1, "lat": 32.1, "lon": -109.9, "alt_m": 6000.0, "maidenhead_grid": "DM43"},
    ]
    
    # Interpolate exactly in the middle
    t_mid = t0 + timedelta(minutes=5)
    res = interpolate_balloon_telemetry(points, t_mid)
    
    assert res is not None
    assert abs(res["lat"] - 32.05) < 1e-4
    assert abs(res["lon"] - (-109.95)) < 1e-4
    assert abs(res["alt_m"] - 5500.0) < 1e-1
    assert res["maidenhead_grid"] == "DM43"  # nearest neighbor is DM43 (midpoint defaults to p1)


def test_cross_radar_association_mock(tmp_path):
    from scripts.associate_cross_radar_tracklets import evaluate_association
    
    t1_start = datetime(2026, 3, 22, 20, 0, 0, tzinfo=UTC)
    
    t1_summary = pd.Series({
        "radar_site": "KEMX",
        "tracklet_id": "KEMX_T001",
        "telemetry_match_score": 0.8,
        "tracklet_label": "telemetry_consistent_candidate_tracklet",
    })
    
    t2_summary = pd.Series({
        "radar_site": "KIWA",
        "tracklet_id": "KIWA_T001",
        "telemetry_match_score": 0.75,
        "tracklet_label": "telemetry_consistent_candidate_tracklet",
    })
    
    # Build overlapping points
    times = [t1_start + timedelta(minutes=i) for i in range(21)]
    
    t1_points = pd.DataFrame({
        "time_dt": times,
        "cluster_lat_deg": [32.0 + 0.005 * i for i in range(21)],
        "cluster_lon_deg": [-110.0 + 0.005 * i for i in range(21)],
        "cluster_alt_m": [5000.0 + 50.0 * i for i in range(21)],
        "balloon_like_cluster_score": [0.8] * 21,
    })
    
    # T2 is close (e.g. 2 km offset, 100m alt offset)
    t2_points = pd.DataFrame({
        "time_dt": times,
        "cluster_lat_deg": [32.0 + 0.005 * i + 0.018 for i in range(21)],  # 0.018 deg is ~2 km
        "cluster_lon_deg": [-110.0 + 0.005 * i for i in range(21)],
        "cluster_alt_m": [5000.0 + 50.0 * i + 100.0 for i in range(21)],
        "balloon_like_cluster_score": [0.75] * 21,
    })
    
    res = evaluate_association(t1_summary, t1_points, t2_summary, t2_points)
    assert res is not None
    assert res["association_label"] == "strong_cross_radar_candidate"
    assert res["median_horizontal_difference_km"] < 5.0
    assert res["median_altitude_difference_m"] == 100.0


def test_tracklet_plausibility_filtering():
    import pandas as pd

    from scripts.filter_plausible_tracklets import evaluate_tracklet_quality
    
    # 1. Test rejected_too_short (n_points < 4 or duration < 15)
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 3,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    t_pts = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z", "2026-03-22T20:20:00Z"],
        "cluster_lat_deg": [32.0, 32.05, 32.1],
        "cluster_lon_deg": [-110.0, -109.95, -109.9],
        "distance_to_track_corridor_km": [5.0, 5.0, 5.0],
        "inside_or_near_grid_corridor": [True, True, True],
        "tracklet_id": ["KEMX_T001"] * 3,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_too_short"
    assert res["status"] == "rejected"

    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 10.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    t_pts = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:02:00Z", "2026-03-22T20:04:00Z", "2026-03-22T20:06:00Z", "2026-03-22T20:10:00Z"],
        "cluster_lat_deg": [32.0, 32.01, 32.02, 32.03, 32.05],
        "cluster_lon_deg": [-110.0, -109.99, -109.98, -109.97, -109.95],
        "distance_to_track_corridor_km": [5.0, 5.0, 5.0, 5.0, 5.0],
        "inside_or_near_grid_corridor": [True, True, True, True, True],
        "tracklet_id": ["KEMX_T001"] * 5,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_too_short"
    assert res["status"] == "rejected"

    # 2. Test rejected_altitude_mismatch (median > 750 or max > 2000)
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 800.0,
        "max_abs_vertical_mismatch_m": 1200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_altitude_mismatch"
    assert res["status"] == "rejected"

    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 2100.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_altitude_mismatch"
    assert res["status"] == "rejected"

    # 3. Test rejected_speed_jump / limits
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 3.0,
        "max_segment_speed_kmh": 5.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_speed_jump"

    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 110.0,
        "max_segment_speed_kmh": 120.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_speed_jump"

    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 160.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_speed_jump"

    # Single segment speed dominance
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 4,
        "duration_min": 30.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 20.0,
        "max_segment_speed_kmh": 120.0,
        "path_smoothness_score": 0.8,
    })
    t_pts_dom = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z", "2026-03-22T20:20:00Z", "2026-03-22T20:30:00Z"],
        "cluster_lat_deg": [32.0, 32.01, 32.02, 32.72],
        "cluster_lon_deg": [-110.0, -110.0, -110.0, -110.0],
        "distance_to_track_corridor_km": [5.0, 5.0, 5.0, 5.0],
        "inside_or_near_grid_corridor": [True, True, True, True],
        "tracklet_id": ["KEMX_T001"] * 4,
    })
    res = evaluate_tracklet_quality(row, t_pts_dom, t_pts_dom)
    assert res["quality_label"] == "rejected_speed_jump"
    assert "dominates" in res["reject_reason"]

    # 4. Test rejected_not_near_telemetry_corridor (corridor_fraction < 0.5)
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    t_pts_off = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:02:00Z", "2026-03-22T20:04:00Z", "2026-03-22T20:06:00Z", "2026-03-22T20:10:00Z"],
        "cluster_lat_deg": [32.0, 32.01, 32.02, 32.03, 32.05],
        "cluster_lon_deg": [-110.0, -109.99, -109.98, -109.97, -109.95],
        "distance_to_track_corridor_km": [5.0, 50.0, 60.0, 70.0, 5.0],
        "inside_or_near_grid_corridor": [True, False, False, False, True],
        "tracklet_id": ["KEMX_T001"] * 5,
    })
    res = evaluate_tracklet_quality(row, t_pts_off, t_pts_off)
    assert res["quality_label"] == "rejected_not_near_telemetry_corridor"

    # 5. Test rejected_spaghetti_tracklet (path_smoothness_score < 0.4)
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.3,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "rejected_spaghetti_tracklet"

    # 6. Test excellent_plausible_tracklet
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "radar_site": "KEMX",
        "n_points": 5,
        "duration_min": 20.0,
        "median_abs_vertical_mismatch_m": 100.0,
        "max_abs_vertical_mismatch_m": 200.0,
        "median_segment_speed_kmh": 40.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.8,
    })
    res = evaluate_tracklet_quality(row, t_pts, t_pts)
    assert res["quality_label"] == "excellent_plausible_tracklet"
    assert res["status"] == "plausible"


def test_spaghetti_score_calculation():
    import pandas as pd

    from scripts.filter_plausible_tracklets import compute_spaghetti_score
    
    # Base case: perfect smoothness, no speed jump, no reversals, no overlaps, zero distance
    row = pd.Series({
        "tracklet_id": "KEMX_T001",
        "median_segment_speed_kmh": 30.0,
        "max_segment_speed_kmh": 30.0,
        "path_smoothness_score": 1.0,
    })
    t_pts = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z"],
        "cluster_lat_deg": [32.0, 32.01],
        "cluster_lon_deg": [-110.0, -110.0],
        "distance_to_track_corridor_km": [0.0, 0.0],
        "tracklet_id": ["KEMX_T001", "KEMX_T001"],
    })
    score = compute_spaghetti_score(row, t_pts, t_pts)
    assert abs(score - 9.677) < 1e-2

    # Case with penalties
    row2 = pd.Series({
        "tracklet_id": "KEMX_T001",
        "median_segment_speed_kmh": 10.0,
        "max_segment_speed_kmh": 50.0,
        "path_smoothness_score": 0.6,
    })
    # Reversal: North then South
    t_pts2 = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z", "2026-03-22T20:20:00Z"],
        "cluster_lat_deg": [32.0, 32.1, 32.0],
        "cluster_lon_deg": [-110.0, -110.0, -110.0],
        "distance_to_track_corridor_km": [10.0, 20.0, 10.0],
        "tracklet_id": ["KEMX_T001"] * 3,
    })
    other_pts = pd.DataFrame({
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:50:00Z"],
        "cluster_lat_deg": [32.0, 32.5],
        "cluster_lon_deg": [-110.0, -110.0],
        "distance_to_track_corridor_km": [10.0, 10.0],
        "tracklet_id": ["KEMX_T002"] * 2,
    })
    all_pts = pd.concat([t_pts2, other_pts], ignore_index=True)
    
    score2 = compute_spaghetti_score(row2, t_pts2, all_pts)
    assert abs(score2 - 182.1212) < 1e-2


def test_filter_script_smoketest(tmp_path):
    import sys
    from unittest.mock import patch

    import yaml
    
    case_dir = tmp_path / "cases" / "k7uaz_20260322"
    case_dir.mkdir(parents=True)
    
    config_data = {
        "discovery": {
            "radar_sites_primary": ["KEMX"],
            "radar_sites_secondary": [],
        },
        "radar_sites": {
            "KEMX": {"lat": 32.0, "lon": -110.0, "alt_m": 1000.0},
        }
    }
    
    config_file = case_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
        
    nexrad_dir = case_dir / "nexrad"
    nexrad_dir.mkdir()
    geom_df = pd.DataFrame({
        "radar_site": ["KEMX"],
        "geometry_status": ["include"],
    })
    geom_df.to_csv(nexrad_dir / "regional_radar_geometry.csv", index=False)
    
    out_dir = case_dir / "outputs" / "discovery"
    kemx_out_dir = out_dir / "KEMX"
    kemx_out_dir.mkdir(parents=True)
    
    raw_tracklets_df = pd.DataFrame({
        "tracklet_id": ["KEMX_T001"],
        "radar_site": ["KEMX"],
        "n_points": [5],
        "duration_min": [20.0],
        "median_abs_vertical_mismatch_m": [100.0],
        "max_abs_vertical_mismatch_m": [200.0],
        "median_segment_speed_kmh": [40.0],
        "max_segment_speed_kmh": [50.0],
        "path_smoothness_score": [0.8],
    })
    raw_tracklets_df.to_csv(kemx_out_dir / "candidate_tracklets.csv", index=False)
    
    points_df = pd.DataFrame({
        "tracklet_id": ["KEMX_T001"] * 5,
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:05:00Z", "2026-03-22T20:10:00Z", "2026-03-22T20:15:00Z", "2026-03-22T20:20:00Z"],
        "cluster_lat_deg": [32.0, 32.01, 32.02, 32.03, 32.04],
        "cluster_lon_deg": [-110.0, -109.99, -109.98, -109.97, -109.96],
        "distance_to_track_corridor_km": [5.0] * 5,
        "inside_or_near_grid_corridor": [True] * 5,
    })
    points_df.to_csv(kemx_out_dir / "tracklet_points.csv", index=False)
    
    assoc_df = pd.DataFrame({
        "association_id": ["A001"],
        "radar_sites": ["KEMX;KEMX"],
        "tracklet_ids": ["KEMX_T001;KEMX_T001"],
        "n_radars": [1],
        "time_overlap_min": [10.0],
        "n_overlap_samples": [2],
        "median_altitude_difference_m": [100.0],
        "median_horizontal_difference_km": [2.5],
        "mean_telemetry_consistency_score": [0.8],
        "cross_radar_score": [0.85],
        "association_label": ["strong_cross_radar_candidate"],
        "notes": ["mock association"],
    })
    assoc_df.to_csv(out_dir / "cross_radar_tracklet_associations.csv", index=False)
    
    from scripts.filter_plausible_tracklets import main as filter_main
    
    test_args = ["filter_plausible_tracklets.py", str(config_file), "--primary-sites"]
    
    with patch("shutil.copy"):
        with patch.object(sys, "argv", test_args):
            filter_main()
            
    assert (out_dir / "plausible_tracklets.csv").exists()
    assert (out_dir / "plausible_tracklet_points.csv").exists()
    assert (out_dir / "tracklet_quality_diagnostics.csv").exists()
    assert (out_dir / "plausible_cross_radar_associations.csv").exists()


def test_dashboard_creation_smoketest(tmp_path):
    import sys
    from unittest.mock import patch

    import yaml
    
    case_dir = tmp_path / "cases" / "k7uaz_20260322"
    case_dir.mkdir(parents=True)
    
    config_data = {
        "discovery": {
            "radar_sites_primary": ["KEMX"],
            "radar_sites_secondary": [],
        },
        "radar_sites": {
            "KEMX": {"lat": 32.0, "lon": -110.0, "alt_m": 1000.0},
        }
    }
    
    config_file = case_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
        
    nexrad_dir = case_dir / "nexrad"
    nexrad_dir.mkdir()
    geom_df = pd.DataFrame({
        "radar_site": ["KEMX"],
        "geometry_status": ["include"],
    })
    geom_df.to_csv(nexrad_dir / "regional_radar_geometry.csv", index=False)
    
    expected_track_df = pd.DataFrame({
        "lat_deg": [32.0, 32.1],
        "lon_deg": [-110.0, -109.9],
        "maidenhead_grid": ["DM42", "DM42"],
        "time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z"],
        "alt_m": [5000.0, 5100.0],
    })
    expected_track_df.to_csv(case_dir / "expected_track.csv", index=False)
    
    out_dir = case_dir / "outputs" / "discovery"
    kemx_out_dir = out_dir / "KEMX"
    kemx_out_dir.mkdir(parents=True)
    
    diag_df = pd.DataFrame({
        "tracklet_id": ["KEMX_T001"],
        "radar_site": ["KEMX"],
        "status": ["plausible"],
        "quality_label": ["excellent_plausible_tracklet"],
        "spaghetti_score": [12.5],
        "reject_reason": [""],
    })
    diag_df.to_csv(out_dir / "tracklet_quality_diagnostics.csv", index=False)
    
    points_df = pd.DataFrame({
        "tracklet_id": ["KEMX_T001", "KEMX_T001"],
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z"],
        "cluster_lat_deg": [32.0, 32.05],
        "cluster_lon_deg": [-110.0, -109.95],
        "cluster_alt_m": [5000.0, 5200.0],
        "expected_alt_m": [5100.0, 5100.0],
        "signed_vertical_m": [-100.0, 100.0],
        "max_reflectivity_dbz": [15.0, 18.0],
        "balloon_like_cluster_score": [0.8, 0.85],
    })
    points_df.to_csv(kemx_out_dir / "tracklet_points.csv", index=False)
    
    assoc_df = pd.DataFrame({
        "association_id": ["A001"],
        "radar_sites": ["KEMX;KEMX"],
        "tracklet_ids": ["KEMX_T001;KEMX_T001"],
        "n_radars": [1],
        "time_overlap_min": [10.0],
        "n_overlap_samples": [2],
        "median_altitude_difference_m": [100.0],
        "median_horizontal_difference_km": [2.5],
        "mean_telemetry_consistency_score": [0.8],
        "cross_radar_score": [0.85],
        "association_label": ["strong_cross_radar_candidate"],
        "notes": ["mock association"],
    })
    assoc_df.to_csv(out_dir / "cross_radar_tracklet_associations.csv", index=False)
    
    from scripts.make_plausible_tracklet_dashboard import main as make_dashboard_main
    
    test_args = ["make_plausible_tracklet_dashboard.py", str(config_file), "--primary-sites", "--overwrite"]
    
    with patch("shutil.copy"):
        with patch.object(sys, "argv", test_args):
            make_dashboard_main()
            
    assert (out_dir / "plausible_tracklet_dashboard.html").exists()
