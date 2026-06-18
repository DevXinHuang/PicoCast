"""Tests for candidate path fitting logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_candidate_path import (  # noqa: E402
    MAX_PLAUSIBLE_SPEED_KMH,
    filter_plausible_candidates,
    generate_path_report,
    path_to_geojson,
    segment_bearing_deg,
    segment_speed_kmh,
    select_path,
    smoothed_path_geojson,
    smoothness_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 5) -> pd.DataFrame:
    """Create a minimal plausible candidate DataFrame."""
    base_time = pd.Timestamp("2026-03-22T20:00:00Z")
    rows = []
    for i in range(n):
        dt = base_time + pd.Timedelta(minutes=7 * i)
        scan = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({
            "case_id": "test",
            "radar_site": "KEMX",
            "scan_time_utc": scan,
            "scan_filename": f"f{i}",
            "search_window": "tight",
            "cluster_id": f"c{i}",
            "expected_lat_deg": 32.33 + i * 0.005,
            "expected_lon_deg": -110.60 + i * 0.01,
            "expected_alt_m": 5000.0 + i * 500,
            "candidate_lat_deg": 32.33 + i * 0.005 + 0.002,
            "candidate_lon_deg": -110.60 + i * 0.01 + 0.002,
            "candidate_alt_m": 5100.0 + i * 500,
            "horizontal_distance_km": 1.5 + i * 0.5,
            "vertical_distance_m": 100.0 + i * 20,
            "n_gates": 3,
            "max_reflectivity_dbz": 5.0,
            "mean_reflectivity_dbz": 3.0,
            "p95_reflectivity_dbz": 4.0,
            "velocity_mean_ms": 0.0,
            "spectrum_width_mean_ms": 0.0,
            "rhohv_mean": 0.5,
            "distance_score": 0.8,
            "altitude_score": 0.9,
            "reflectivity_score": 0.3,
            "compactness_score": 0.7,
            "isolation_score": 0.3,
            "temporal_continuity_score": 0.0,
            "candidate_score": 0.65,
            "candidate_rank": i + 1,
            "candidate_label": "strong_candidate",
            "notes": "",
            "signed_vertical_m": 100.0 + i * 20,
            "abs_vertical_distance_m": 100.0 + i * 20,
            "interpolated_expected_alt_m": 5000.0 + i * 500,
            "signed_vertical_interp_m": 100.0 + i * 20,
            "abs_vertical_interp_m": 100.0 + i * 20,
            "altitude_consistency_score": 1.0,
            "altitude_consistency_label": "excellent_altitude_match",
            "original_candidate_rank": i + 1,
            "altitude_priority_rank": i + 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test filtering
# ---------------------------------------------------------------------------


class TestFilterPlausible:
    """Tests for candidate filtering."""

    def test_filters_by_altitude_score(self):
        df = _make_candidates(3)
        df.loc[2, "altitude_consistency_score"] = 0.1  # too low
        result = filter_plausible_candidates(df)
        assert len(result) == 2

    def test_filters_by_candidate_score(self):
        df = _make_candidates(3)
        df.loc[1, "candidate_score"] = 0.05  # too low
        result = filter_plausible_candidates(df)
        assert len(result) == 2

    def test_time_window_filtering(self):
        df = _make_candidates(5)
        result = filter_plausible_candidates(
            df,
            start_time="2026-03-22T20:00:00Z",
            end_time="2026-03-22T20:15:00Z",
        )
        # Only first 3 scan times (0, 7, 14 min) fit in 0-15 min window
        assert len(result) <= 3

    def test_empty_input(self):
        df = _make_candidates(0)
        result = filter_plausible_candidates(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Test segment scoring
# ---------------------------------------------------------------------------


class TestSegmentScoring:
    """Tests for segment speed and bearing calculations."""

    def test_speed_zero_same_point(self):
        speed = segment_speed_kmh(
            32.0, -110.0, "2026-03-22T20:00:00Z",
            32.0, -110.0, "2026-03-22T20:07:00Z",
        )
        assert speed == pytest.approx(0.0)

    def test_speed_plausible(self):
        # ~1 km apart, 7 min = ~8.5 km/h
        speed = segment_speed_kmh(
            32.0, -110.0, "2026-03-22T20:00:00Z",
            32.01, -110.0, "2026-03-22T20:07:00Z",
        )
        assert 0 < speed < MAX_PLAUSIBLE_SPEED_KMH

    def test_bearing_north(self):
        bearing = segment_bearing_deg(32.0, -110.0, 33.0, -110.0)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_bearing_east(self):
        bearing = segment_bearing_deg(32.0, -110.0, 32.0, -109.0)
        assert bearing == pytest.approx(90.0, abs=2.0)

    def test_smoothness_impossible_speed(self):
        score = smoothness_score(300.0, None, 45.0)
        assert score == 0.0

    def test_smoothness_zero_speed(self):
        score = smoothness_score(0.0, None, 45.0)
        assert score > 0.5

    def test_smoothness_bearing_reversal_penalized(self):
        s1 = smoothness_score(50.0, 0.0, 10.0)    # small change
        s2 = smoothness_score(50.0, 0.0, 170.0)    # near reversal
        assert s1 > s2


# ---------------------------------------------------------------------------
# Test path selection
# ---------------------------------------------------------------------------


class TestPathSelection:
    """Tests for greedy path selection."""

    def test_selects_one_per_scan_time(self):
        df = _make_candidates(3)
        # Add duplicate for first scan time
        dup = df.iloc[0:1].copy()
        dup["cluster_id"] = "dup"
        dup["candidate_alt_m"] = 5050.0
        df = pd.concat([df, dup], ignore_index=True)
        path = select_path(df)
        # Should have at most 3 path points (one per unique scan time)
        assert len(path) == 3

    def test_path_sorted_by_time(self):
        df = _make_candidates(4)
        path = select_path(df)
        times = pd.to_datetime(path["scan_time_utc"])
        assert (times.diff().dropna() > pd.Timedelta(0)).all()

    def test_rejects_impossible_jump(self):
        df = _make_candidates(3)
        # Move second candidate very far away (impossible jump)
        df.loc[1, "candidate_lat_deg"] = 45.0  # ~1400 km north
        df.loc[1, "candidate_lon_deg"] = -110.0
        path = select_path(df)
        # Should skip the impossible point or select an alternative
        if len(path) >= 2:
            speeds = path["segment_speed_kmh"]
            assert (speeds <= MAX_PLAUSIBLE_SPEED_KMH).all()

    def test_empty_input(self):
        df = _make_candidates(0)
        path = select_path(df)
        assert len(path) == 0

    def test_single_point(self):
        df = _make_candidates(1)
        path = select_path(df)
        assert len(path) == 1

    def test_path_has_required_columns(self):
        df = _make_candidates(3)
        path = select_path(df)
        required = [
            "case_id", "radar_site", "scan_time_utc",
            "original_candidate_rank", "altitude_priority_rank",
            "candidate_lat_deg", "candidate_lon_deg", "candidate_alt_m",
            "signed_vertical_m", "abs_vertical_distance_m",
            "horizontal_distance_km", "candidate_score",
            "altitude_consistency_score", "altitude_consistency_label",
            "segment_speed_kmh", "segment_bearing_deg",
            "path_step_score", "path_selected_reason",
        ]
        for col in required:
            assert col in path.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test GeoJSON output
# ---------------------------------------------------------------------------


class TestGeoJSON:
    """Tests for GeoJSON generation."""

    def test_path_geojson_has_features(self):
        df = _make_candidates(3)
        path = select_path(df)
        gj = path_to_geojson(path)
        assert gj["type"] == "FeatureCollection"
        # 3 points + 1 line = 4 features
        assert len(gj["features"]) == 4

    def test_smoothed_geojson_has_features(self):
        df = _make_candidates(4)
        path = select_path(df)
        gj = smoothed_path_geojson(path)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) > 0

    def test_geojson_serializable(self):
        df = _make_candidates(3)
        path = select_path(df)
        gj = path_to_geojson(path)
        # Should not raise
        json.dumps(gj, default=str)


# ---------------------------------------------------------------------------
# Test report generation
# ---------------------------------------------------------------------------


class TestReport:
    """Tests for path fit report."""

    def test_report_creates_file(self, tmp_path):
        df = _make_candidates(3)
        path = select_path(df)
        plausible = df.copy()
        out = tmp_path / "report.md"
        generate_path_report(
            path, plausible, "test", "KEMX",
            "2026-03-22T20:00:00Z", "2026-03-22T20:30:00Z", out,
        )
        assert out.exists()
        text = out.read_text()
        assert "# Candidate Trajectory Report" in text

    def test_report_contains_caveats(self, tmp_path):
        df = _make_candidates(3)
        path = select_path(df)
        out = tmp_path / "report.md"
        generate_path_report(
            path, df, "test", "KEMX",
            "2026-03-22T20:00:00Z", "2026-03-22T20:30:00Z", out,
        )
        text = out.read_text()
        assert "Maidenhead" in text
        assert "beam-center" in text
        assert "not a confirmed" in text or "not confirmed" in text

    def test_report_no_detection_claim(self, tmp_path):
        df = _make_candidates(3)
        path = select_path(df)
        out = tmp_path / "report.md"
        generate_path_report(
            path, df, "test", "KEMX",
            "2026-03-22T20:00:00Z", "2026-03-22T20:30:00Z", out,
        )
        text = out.read_text()
        # Report uses cautious language: "not a confirmed detection"
        assert "not a confirmed" in text.lower() or "not confirmed" in text.lower()

    def test_report_empty_path(self, tmp_path):
        df = _make_candidates(0)
        path = select_path(df)
        out = tmp_path / "report.md"
        generate_path_report(
            path, df, "test", "KEMX",
            "2026-03-22T20:00:00Z", "2026-03-22T20:30:00Z", out,
        )
        text = out.read_text()
        assert out.exists()
        assert "0" in text  # should mention 0 path points
