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
