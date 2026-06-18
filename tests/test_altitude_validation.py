"""Tests for altitude-first validation logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_altitude_consistency import (  # noqa: E402
    ALTITUDE_THRESHOLDS,
    DEFAULT_LABEL,
    DEFAULT_SCORE,
    altitude_consistency_label,
    altitude_consistency_score,
    build_altitude_prioritized,
    generate_report,
    interpolate_expected_altitude,
)

# ---------------------------------------------------------------------------
# Test altitude_consistency_score
# ---------------------------------------------------------------------------


class TestAltitudeConsistencyScore:
    """Tests for the altitude_consistency_score function."""

    def test_zero_mismatch(self):
        assert altitude_consistency_score(0.0) == 1.0

    def test_excellent_boundary(self):
        assert altitude_consistency_score(250.0) == 1.0

    def test_strong_boundary(self):
        assert altitude_consistency_score(500.0) == 0.8

    def test_moderate_boundary(self):
        assert altitude_consistency_score(1000.0) == 0.6

    def test_weak_boundary(self):
        assert altitude_consistency_score(2000.0) == 0.3

    def test_poor_above_threshold(self):
        assert altitude_consistency_score(2001.0) == DEFAULT_SCORE

    def test_very_large_mismatch(self):
        assert altitude_consistency_score(50000.0) == DEFAULT_SCORE

    def test_nan_returns_default(self):
        assert altitude_consistency_score(float("nan")) == DEFAULT_SCORE

    def test_just_below_strong(self):
        assert altitude_consistency_score(499.0) == 0.8

    def test_between_excellent_and_strong(self):
        assert altitude_consistency_score(300.0) == 0.8

    def test_between_moderate_and_weak(self):
        assert altitude_consistency_score(1500.0) == 0.3


# ---------------------------------------------------------------------------
# Test altitude_consistency_label
# ---------------------------------------------------------------------------


class TestAltitudeConsistencyLabel:
    """Tests for the altitude_consistency_label function."""

    def test_excellent(self):
        assert altitude_consistency_label(100.0) == "excellent_altitude_match"

    def test_strong(self):
        assert altitude_consistency_label(400.0) == "strong_altitude_match"

    def test_moderate(self):
        assert altitude_consistency_label(800.0) == "moderate_altitude_match"

    def test_weak(self):
        assert altitude_consistency_label(1500.0) == "weak_altitude_match"

    def test_poor(self):
        assert altitude_consistency_label(3000.0) == DEFAULT_LABEL

    def test_nan(self):
        assert altitude_consistency_label(float("nan")) == DEFAULT_LABEL

    def test_boundary_250(self):
        assert altitude_consistency_label(250.0) == "excellent_altitude_match"

    def test_boundary_251(self):
        assert altitude_consistency_label(251.0) == "strong_altitude_match"

    def test_all_labels_are_unique(self):
        labels = {label for _, _, label in ALTITUDE_THRESHOLDS}
        labels.add(DEFAULT_LABEL)
        assert len(labels) == len(ALTITUDE_THRESHOLDS) + 1


# ---------------------------------------------------------------------------
# Test interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Tests for altitude interpolation from telemetry."""

    @pytest.fixture()
    def sample_track(self):
        return pd.DataFrame({
            "time_utc": [
                "2026-03-22T19:00:00Z",
                "2026-03-22T20:00:00Z",
                "2026-03-22T21:00:00Z",
            ],
            "alt_m": [2000.0, 5000.0, 8000.0],
        })

    def test_exact_match_first(self, sample_track):
        alt = interpolate_expected_altitude("2026-03-22T19:00:00Z", sample_track)
        assert alt == pytest.approx(2000.0)

    def test_exact_match_last(self, sample_track):
        alt = interpolate_expected_altitude("2026-03-22T21:00:00Z", sample_track)
        assert alt == pytest.approx(8000.0)

    def test_midpoint_interpolation(self, sample_track):
        alt = interpolate_expected_altitude("2026-03-22T19:30:00Z", sample_track)
        assert alt == pytest.approx(3500.0, abs=1.0)

    def test_before_first_clamps(self, sample_track):
        alt = interpolate_expected_altitude("2026-03-22T18:00:00Z", sample_track)
        assert alt == pytest.approx(2000.0)

    def test_after_last_clamps(self, sample_track):
        alt = interpolate_expected_altitude("2026-03-22T22:00:00Z", sample_track)
        assert alt == pytest.approx(8000.0)


# ---------------------------------------------------------------------------
# Test altitude-prioritized sorting
# ---------------------------------------------------------------------------


class TestAltitudePrioritizedSorting:
    """Tests for altitude-first candidate re-ranking."""

    @pytest.fixture()
    def sample_candidates(self):
        """Candidates with varying vertical distances."""
        return pd.DataFrame({
            "case_id": ["c1"] * 4,
            "radar_site": ["KEMX"] * 4,
            "scan_time_utc": ["2026-03-22T20:00:00Z"] * 4,
            "scan_filename": ["f1"] * 4,
            "search_window": ["tight"] * 4,
            "cluster_id": ["a", "b", "c", "d"],
            "expected_lat_deg": [32.0] * 4,
            "expected_lon_deg": [-110.0] * 4,
            "expected_alt_m": [5000.0] * 4,
            "candidate_lat_deg": [32.01] * 4,
            "candidate_lon_deg": [-110.01] * 4,
            "candidate_alt_m": [5100.0, 5600.0, 7000.0, 4900.0],
            "horizontal_distance_km": [2.0, 5.0, 8.0, 1.5],
            "vertical_distance_m": [100.0, 600.0, 2000.0, 100.0],
            "n_gates": [3, 10, 50, 2],
            "max_reflectivity_dbz": [5.0, 15.0, 25.0, 3.0],
            "mean_reflectivity_dbz": [3.0, 12.0, 20.0, 2.0],
            "p95_reflectivity_dbz": [4.0, 14.0, 23.0, 2.5],
            "velocity_mean_ms": [0.0] * 4,
            "spectrum_width_mean_ms": [0.0] * 4,
            "rhohv_mean": [0.5] * 4,
            "distance_score": [0.9, 0.5, 0.2, 0.95],
            "altitude_score": [0.9, 0.4, 0.0, 0.9],
            "reflectivity_score": [0.3, 0.6, 0.8, 0.2],
            "compactness_score": [0.8, 0.6, 0.3, 0.9],
            "isolation_score": [0.3, 0.5, 0.7, 0.2],
            "temporal_continuity_score": [0.0] * 4,
            "candidate_score": [0.7, 0.6, 0.5, 0.65],
            "candidate_rank": [1, 2, 3, 4],
            "candidate_label": ["strong", "moderate", "weak", "strong"],
            "notes": [""] * 4,
        })

    @pytest.fixture()
    def sample_track(self):
        return pd.DataFrame({
            "time_utc": [
                "2026-03-22T19:00:00Z",
                "2026-03-22T21:00:00Z",
            ],
            "alt_m": [3000.0, 7000.0],
        })

    def test_preserves_original_rank(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        assert "original_candidate_rank" in result.columns
        assert set(result["original_candidate_rank"]) == {1, 2, 3, 4}

    def test_adds_altitude_priority_rank(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        assert "altitude_priority_rank" in result.columns
        assert list(result["altitude_priority_rank"]) == [1, 2, 3, 4]

    def test_altitude_consistent_ranked_first(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        # Candidates a and d have small vertical mismatch (excellent)
        # c has 2000m mismatch (weak/poor)
        top = result.iloc[0]
        assert top["altitude_consistency_label"] in (
            "excellent_altitude_match",
            "strong_altitude_match",
        )

    def test_poor_candidate_ranked_last(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        last = result.iloc[-1]
        assert last["altitude_consistency_label"] in (
            "weak_altitude_match",
            "poor_altitude_match",
        )

    def test_signed_vertical_both_signs(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        signs = result["signed_vertical_m"].values
        # candidate d (4900m) - expected (5000m) = -100  (negative)
        # candidate c (7000m) - expected (5000m) = +2000 (positive)
        assert (signs < 0).any()
        assert (signs > 0).any()

    def test_abs_vertical_always_positive(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        assert (result["abs_vertical_distance_m"] >= 0).all()

    def test_has_interpolated_columns(self, sample_candidates, sample_track):
        result = build_altitude_prioritized(sample_candidates, sample_track)
        assert "interpolated_expected_alt_m" in result.columns
        assert "signed_vertical_interp_m" in result.columns
        assert "abs_vertical_interp_m" in result.columns


# ---------------------------------------------------------------------------
# Test report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Tests for the altitude validation report."""

    def test_report_is_markdown(self, tmp_path):
        track = pd.DataFrame({
            "time_utc": ["2026-03-22T19:00:00Z", "2026-03-22T21:00:00Z"],
            "alt_m": [3000.0, 7000.0],
        })
        candidates = pd.DataFrame({
            "scan_time_utc": ["2026-03-22T20:00:00Z"],
            "candidate_alt_m": [5100.0],
            "expected_alt_m": [5000.0],
            "signed_vertical_m": [100.0],
            "abs_vertical_distance_m": [100.0],
            "interpolated_expected_alt_m": [5000.0],
            "signed_vertical_interp_m": [100.0],
            "abs_vertical_interp_m": [100.0],
            "altitude_consistency_score": [1.0],
            "altitude_consistency_label": ["excellent_altitude_match"],
            "original_candidate_rank": [1],
            "altitude_priority_rank": [1],
            "candidate_score": [0.7],
            "candidate_label": ["strong_candidate"],
            "horizontal_distance_km": [2.0],
            "candidate_rank": [1],
        })
        out = tmp_path / "report.md"
        generate_report(candidates, track, "test_case", "KEMX", out)
        assert out.exists()
        text = out.read_text()
        assert "# Altitude Validation Report" in text

    def test_report_contains_caveats(self, tmp_path):
        track = pd.DataFrame({
            "time_utc": ["2026-03-22T19:00:00Z", "2026-03-22T21:00:00Z"],
            "alt_m": [3000.0, 7000.0],
        })
        candidates = pd.DataFrame({
            "scan_time_utc": ["2026-03-22T20:00:00Z"],
            "candidate_alt_m": [5100.0],
            "expected_alt_m": [5000.0],
            "signed_vertical_m": [100.0],
            "abs_vertical_distance_m": [100.0],
            "interpolated_expected_alt_m": [5000.0],
            "signed_vertical_interp_m": [100.0],
            "abs_vertical_interp_m": [100.0],
            "altitude_consistency_score": [1.0],
            "altitude_consistency_label": ["excellent_altitude_match"],
            "original_candidate_rank": [1],
            "altitude_priority_rank": [1],
            "candidate_score": [0.7],
            "candidate_label": ["strong_candidate"],
            "horizontal_distance_km": [2.0],
            "candidate_rank": [1],
        })
        out = tmp_path / "report.md"
        generate_report(candidates, track, "test_case", "KEMX", out)
        text = out.read_text()
        assert "Maidenhead grid" in text
        assert "beam-center" in text
        assert "linearly interpolated" in text
        assert "not sufficient" in text
        assert "several kilometers" in text

    def test_report_contains_no_detection_claim(self, tmp_path):
        track = pd.DataFrame({
            "time_utc": ["2026-03-22T19:00:00Z"],
            "alt_m": [5000.0],
        })
        candidates = pd.DataFrame({
            "scan_time_utc": ["2026-03-22T19:00:00Z"],
            "candidate_alt_m": [5100.0],
            "expected_alt_m": [5000.0],
            "signed_vertical_m": [100.0],
            "abs_vertical_distance_m": [100.0],
            "interpolated_expected_alt_m": [5000.0],
            "signed_vertical_interp_m": [100.0],
            "abs_vertical_interp_m": [100.0],
            "altitude_consistency_score": [1.0],
            "altitude_consistency_label": ["excellent_altitude_match"],
            "original_candidate_rank": [1],
            "altitude_priority_rank": [1],
            "candidate_score": [0.7],
            "candidate_label": ["strong_candidate"],
            "horizontal_distance_km": [2.0],
            "candidate_rank": [1],
        })
        out = tmp_path / "report.md"
        generate_report(candidates, track, "test_case", "KEMX", out)
        text = out.read_text().lower()
        assert "detection" not in text or "not" in text.split("detection")[0][-30:]
