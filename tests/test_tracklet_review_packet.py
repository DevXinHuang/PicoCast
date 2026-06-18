"""Tests for the regional tracklet review packet builder."""

from __future__ import annotations

import pandas as pd

from scripts.build_tracklet_review_packet import (
    build_combined_review_queue,
    build_cross_radar_queue,
    build_tracklet_families,
    write_review_report,
)


def make_tracklets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tracklet_id": "KEMX_T001",
                "radar_site": "KEMX",
                "n_points": 3,
                "duration_min": 20.0,
                "median_abs_vertical_mismatch_m": 80.0,
                "mean_distance_to_corridor_km": 5.0,
                "spaghetti_score": 90.0,
                "tracklet_score": 1.1,
                "quality_label": "excellent_plausible_tracklet",
                "stability_label": "stable_candidate",
                "detection_fraction": 0.8,
                "start_time_utc": "2026-03-22T20:00:00Z",
                "end_time_utc": "2026-03-22T20:20:00Z",
            },
            {
                "tracklet_id": "KEMX_T002",
                "radar_site": "KEMX",
                "n_points": 3,
                "duration_min": 20.0,
                "median_abs_vertical_mismatch_m": 50.0,
                "mean_distance_to_corridor_km": 4.0,
                "spaghetti_score": 40.0,
                "tracklet_score": 1.0,
                "quality_label": "excellent_plausible_tracklet",
                "stability_label": "stable_candidate",
                "detection_fraction": 0.8,
                "start_time_utc": "2026-03-22T20:00:00Z",
                "end_time_utc": "2026-03-22T20:20:00Z",
            },
            {
                "tracklet_id": "KIWA_T001",
                "radar_site": "KIWA",
                "n_points": 3,
                "duration_min": 20.0,
                "median_abs_vertical_mismatch_m": 100.0,
                "mean_distance_to_corridor_km": 6.0,
                "spaghetti_score": 60.0,
                "tracklet_score": 1.2,
                "quality_label": "good_plausible_tracklet",
                "stability_label": "moderately_stable_candidate",
                "detection_fraction": 0.5,
                "start_time_utc": "2026-03-22T20:00:00Z",
                "end_time_utc": "2026-03-22T20:20:00Z",
            },
        ]
    )


def make_points() -> pd.DataFrame:
    rows = []
    for tid, clusters, lat_offset in [
        ("KEMX_T001", ["c1", "c2", "c3"], 0.0),
        ("KEMX_T002", ["c1", "c2", "c3"], 0.001),
        ("KIWA_T001", ["k1", "k2", "k3"], 0.02),
    ]:
        for idx, cluster_id in enumerate(clusters):
            rows.append(
                {
                    "tracklet_id": tid,
                    "radar_site": tid.split("_")[0],
                    "scan_time_utc": f"2026-03-22T20:{idx * 10:02d}:00Z",
                    "cluster_id": cluster_id,
                    "cluster_lat_deg": 32.0 + lat_offset + idx * 0.01,
                    "cluster_lon_deg": -110.0 + idx * 0.01,
                    "cluster_alt_m": 5000.0 + idx * 100.0,
                    "expected_alt_m": 5000.0 + idx * 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_duplicate_tracklets_are_grouped_and_representative_is_deterministic():
    family_df, mapping = build_tracklet_families(make_tracklets(), make_points())

    assert len(family_df) == 2
    kemx_family = family_df[family_df["radar_site"] == "KEMX"].iloc[0]
    assert kemx_family["representative_tracklet_id"] == "KEMX_T002"
    assert kemx_family["n_members"] == 2
    assert mapping["KEMX_T001"] == mapping["KEMX_T002"]


def test_cross_radar_items_rank_before_single_radar_families():
    family_df, mapping = build_tracklet_families(make_tracklets(), make_points())
    associations = pd.DataFrame(
        [
            {
                "association_id": "A001",
                "radar_sites": "KEMX;KIWA",
                "tracklet_ids": "KEMX_T002;KIWA_T001",
                "time_overlap_min": 8.8,
                "median_horizontal_difference_km": 9.8,
                "median_altitude_difference_m": 144.0,
                "cross_radar_score": 0.86,
                "association_label": "strong_cross_radar_candidate",
            }
        ]
    )

    cross_queue = build_cross_radar_queue(associations, mapping)
    review_queue = build_combined_review_queue(family_df, cross_queue)

    assert cross_queue.iloc[0]["association_id"] == "A001"
    assert review_queue.iloc[0]["item_type"] == "cross_radar_association"
    assert review_queue.iloc[0]["item_id"] == "A001"
    assert "KEMX_F" in review_queue.iloc[0]["family_ids"]


def test_review_report_uses_cautious_language(tmp_path):
    family_df, mapping = build_tracklet_families(make_tracklets(), make_points())
    associations = pd.DataFrame(
        [
            {
                "association_id": "A001",
                "radar_sites": "KEMX;KIWA",
                "tracklet_ids": "KEMX_T002;KIWA_T001",
                "time_overlap_min": 8.8,
                "median_horizontal_difference_km": 9.8,
                "median_altitude_difference_m": 144.0,
                "cross_radar_score": 0.86,
                "association_label": "strong_cross_radar_candidate",
            }
        ]
    )
    cross_queue = build_cross_radar_queue(associations, mapping)
    review_queue = build_combined_review_queue(family_df, cross_queue)
    report_path = tmp_path / "report.md"

    text = write_review_report(
        {"case_id": "test_case"},
        report_path,
        family_df,
        cross_queue,
        review_queue,
    )

    assert "telemetry-consistent near-track radar features" in text
    assert "visual inspection and multi-radar confirmation" in text
    assert "not a detection claim" in text
    assert "confirmed detection" not in text
    assert "detected the balloon" not in text
