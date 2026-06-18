import pandas as pd

from scripts.candidate_utils import (
    SEARCH_WINDOWS,
    candidate_label,
    horizontal_distance_km,
)
from scripts.cluster_near_track_gates import cluster_group
from scripts.score_near_track_candidates import score_candidates
from scripts.write_candidate_report import build_report


def test_horizontal_distance_calculation():
    distance = float(horizontal_distance_km(32.0, -111.0, 32.0, -110.9))

    assert 9.4 < distance < 9.6


def test_search_window_vertical_behavior():
    expected_alt_m = 10_000.0
    gate_alt_m = 12_500.0
    vertical_distance_km = abs(gate_alt_m - expected_alt_m) / 1000.0

    assert vertical_distance_km > SEARCH_WINDOWS["tight"]["vertical_radius_km"]
    assert vertical_distance_km <= SEARCH_WINDOWS["normal"]["vertical_radius_km"]
    assert vertical_distance_km <= SEARCH_WINDOWS["loose"]["vertical_radius_km"]


def test_cluster_group_finds_compact_cluster():
    gates = pd.DataFrame(
        [
            _gate_row(32.0, -111.0, 10_000, 8.0),
            _gate_row(32.0005, -111.0005, 10_020, 7.0),
            _gate_row(32.08, -110.92, 10_000, 6.0),
        ]
    )

    clusters = cluster_group(gates)

    assert not clusters.empty
    assert clusters["n_gates"].max() >= 2


def test_candidate_labels():
    assert candidate_label(0.10) == "no_candidate"
    assert candidate_label(0.30) == "weak_candidate"
    assert candidate_label(0.50) == "moderate_candidate"
    assert candidate_label(0.80) == "strong_candidate"


def test_candidate_scoring_and_top_sorting():
    clusters = pd.DataFrame(
        [
            _cluster_row("good", 0.5, 100.0, 2, 12.0, 0.1),
            _cluster_row("weak", 9.0, 2_800.0, 50, -8.0, 4.0),
        ]
    )
    summaries = pd.DataFrame(
        [
            {
                "scan_time_utc": "2026-03-22T20:00:00Z",
                "search_window": "normal",
                "expected_lat_deg": 32.0,
                "expected_lon_deg": -111.0,
                "expected_alt_m": 10_000.0,
            }
        ]
    )

    scored = score_candidates(clusters, summaries)

    assert scored.iloc[0]["cluster_id"] == "good"
    assert scored.iloc[0]["candidate_score"] > scored.iloc[1]["candidate_score"]
    assert scored.iloc[0]["candidate_rank"] == 1


def test_report_generation_uses_cautious_language(tmp_path):
    top = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "scan_time_utc": "2026-03-22T20:00:00Z",
                "search_window": "normal",
                "candidate_score": 0.72,
                "candidate_label": "strong_candidate",
                "horizontal_distance_km": 0.8,
                "vertical_distance_m": 150.0,
                "max_reflectivity_dbz": 12.0,
                "n_gates": 3,
            }
        ]
    )
    scores = top.assign(cluster_id="cluster_1")

    report = build_report(
        case_id="k7uaz_20260322",
        radar_site="KEMX",
        scans_analyzed=36,
        scans_inside=27,
        scans_with_possible_returns=12,
        scores=scores,
        top=top,
        output_dir=tmp_path,
    )

    assert "candidate radar return" in report
    assert "near-track radar feature" in report
    assert "requires visual inspection and multi-radar confirmation" in report
    assert "detected the balloon" not in report
    assert "confirmed detection" not in report


def _gate_row(lat, lon, alt_m, reflectivity):
    return {
        "case_id": "case",
        "radar_site": "KEMX",
        "scan_time_utc": "2026-03-22T20:00:00Z",
        "scan_filename": "KEMX20260322_200000_V06",
        "search_window": "normal",
        "expected_lat_deg": 32.0,
        "expected_lon_deg": -111.0,
        "expected_alt_m": 10_000.0,
        "gate_lat_deg": lat,
        "gate_lon_deg": lon,
        "gate_alt_m": alt_m,
        "horizontal_distance_km": float(horizontal_distance_km(lat, lon, 32.0, -111.0)),
        "vertical_distance_m": abs(alt_m - 10_000.0),
        "reflectivity_dbz": reflectivity,
        "velocity_ms": 4.0,
        "spectrum_width_ms": 1.0,
        "cross_correlation_ratio": 0.8,
    }


def _cluster_row(cluster_id, distance_km, vertical_m, n_gates, reflectivity, compactness):
    return {
        "case_id": "case",
        "radar_site": "KEMX",
        "scan_time_utc": "2026-03-22T20:00:00Z",
        "scan_filename": "KEMX20260322_200000_V06",
        "search_window": "normal",
        "cluster_id": cluster_id,
        "n_gates": n_gates,
        "cluster_center_lat_deg": 32.0,
        "cluster_center_lon_deg": -111.0 + distance_km / 94.5,
        "cluster_center_alt_m": 10_000.0 + vertical_m,
        "cluster_min_distance_to_expected_km": distance_km,
        "cluster_min_vertical_distance_m": vertical_m,
        "cluster_max_reflectivity_dbz": reflectivity,
        "cluster_mean_reflectivity_dbz": reflectivity - 1.0,
        "cluster_p95_reflectivity_dbz": reflectivity,
        "cluster_velocity_mean_ms": 4.0,
        "cluster_spectrum_width_mean_ms": 1.0,
        "cluster_rhohv_mean": 0.8,
        "compactness_km": compactness,
        "notes": "test",
    }
