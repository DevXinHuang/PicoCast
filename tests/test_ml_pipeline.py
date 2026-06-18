#!/usr/bin/env python
# ruff: noqa: E501
"""Unit tests for the machine learning feature store, example selection, and training pipeline."""

from __future__ import annotations

import sys
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
import yaml

from scripts.build_ml_feature_table import main as build_feature_table_main
from scripts.select_examples_for_labeling import main as select_examples_main
from scripts.train_candidate_classifier import main as train_classifier_main


@pytest.fixture
def mock_case_env(tmp_path):
    """Set up a mock PicoCAST case environment with all intermediate pipeline outputs."""
    case_dir = tmp_path / "cases" / "k7uaz_20260322"
    case_dir.mkdir(parents=True)

    # config.yaml
    config_data = {
        "case_id": "k7uaz_20260322",
        "discovery": {
            "radar_sites_primary": ["KEMX", "KIWA"],
            "radar_sites_secondary": [],
        },
        "radar_sites": {
            "KEMX": {"lat": 32.0, "lon": -110.0, "alt_m": 1000.0},
            "KIWA": {"lat": 33.0, "lon": -111.0, "alt_m": 1000.0},
        }
    }
    config_file = case_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    # nexrad/regional_radar_geometry.csv
    nexrad_dir = case_dir / "nexrad"
    nexrad_dir.mkdir()
    geom_df = pd.DataFrame({
        "radar_site": ["KEMX", "KIWA"],
        "geometry_status": ["include", "include"],
    })
    geom_df.to_csv(nexrad_dir / "regional_radar_geometry.csv", index=False)

    out_dir = case_dir / "outputs" / "discovery"
    out_dir.mkdir(parents=True)

    # regional_discovered_clusters.parquet
    clusters_df = pd.DataFrame({
        "case_id": ["k7uaz_20260322"] * 3,
        "radar_site": ["KEMX", "KEMX", "KIWA"],
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z", "2026-03-22T20:00:00Z"],
        "cluster_id": [1, 2, 1],
        "n_gates": [10, 8, 12],
        "max_reflectivity_dbz": [18.5, 12.0, 25.0],
        "mean_reflectivity_dbz": [15.0, 9.5, 20.0],
        "range_km": [120.0, 80.0, 110.0],
        "inside_or_near_grid_corridor": [True, True, True],
        "balloon_like_cluster_score": [0.85, 0.45, 0.90],
    })
    clusters_df.to_parquet(out_dir / "regional_discovered_clusters.parquet", index=False)

    # KEMX candidate files
    kemx_dir = out_dir / "KEMX"
    kemx_dir.mkdir()
    kemx_tracklets = pd.DataFrame({
        "case_id": ["k7uaz_20260322"],
        "radar_site": ["KEMX"],
        "tracklet_id": ["KEMX_T001"],
        "n_points": [2],
        "start_time_utc": ["2026-03-22T20:00:00Z"],
        "end_time_utc": ["2026-03-22T20:10:00Z"],
        "duration_min": [10.0],
        "median_abs_vertical_mismatch_m": [50.0],
        "max_abs_vertical_mismatch_m": [100.0],
        "median_segment_speed_kmh": [40.0],
        "max_segment_speed_kmh": [50.0],
        "mean_balloon_like_score": [0.65],
        "path_smoothness_score": [0.85],
        "altitude_consistency_score": [0.90],
        "telemetry_match_score": [0.85],
        "tracklet_score": [0.75],
    })
    kemx_tracklets.to_csv(kemx_dir / "candidate_tracklets.csv", index=False)

    kemx_points = pd.DataFrame({
        "tracklet_id": ["KEMX_T001", "KEMX_T001"],
        "radar_site": ["KEMX", "KEMX"],
        "scan_time_utc": ["2026-03-22T20:00:00Z", "2026-03-22T20:10:00Z"],
        "cluster_id": [1, 2],
        "n_gates": [10, 8],
        "max_reflectivity_dbz": [18.5, 12.0],
        "mean_reflectivity_dbz": [15.0, 9.5],
        "range_km": [120.0, 80.0],
        "inside_or_near_grid_corridor": [True, True],
        "balloon_like_cluster_score": [0.85, 0.45],
    })
    kemx_points.to_csv(kemx_dir / "tracklet_points.csv", index=False)

    # KIWA candidate files
    kiwa_dir = out_dir / "KIWA"
    kiwa_dir.mkdir()
    kiwa_tracklets = pd.DataFrame({
        "case_id": ["k7uaz_20260322"],
        "radar_site": ["KIWA"],
        "tracklet_id": ["KIWA_T001"],
        "n_points": [1],
        "start_time_utc": ["2026-03-22T20:00:00Z"],
        "end_time_utc": ["2026-03-22T20:00:00Z"],
        "duration_min": [0.0],
        "median_abs_vertical_mismatch_m": [150.0],
        "max_abs_vertical_mismatch_m": [150.0],
        "median_segment_speed_kmh": [0.0],
        "max_segment_speed_kmh": [0.0],
        "mean_balloon_like_score": [0.90],
        "path_smoothness_score": [1.0],
        "altitude_consistency_score": [0.80],
        "telemetry_match_score": [0.90],
        "tracklet_score": [0.88],
    })
    kiwa_tracklets.to_csv(kiwa_dir / "candidate_tracklets.csv", index=False)

    kiwa_points = pd.DataFrame({
        "tracklet_id": ["KIWA_T001"],
        "radar_site": ["KIWA"],
        "scan_time_utc": ["2026-03-22T20:00:00Z"],
        "cluster_id": [1],
        "n_gates": [12],
        "max_reflectivity_dbz": [25.0],
        "mean_reflectivity_dbz": [20.0],
        "range_km": [110.0],
        "inside_or_near_grid_corridor": [True],
        "balloon_like_cluster_score": [0.90],
    })
    kiwa_points.to_csv(kiwa_dir / "tracklet_points.csv", index=False)

    # tracklet_quality_diagnostics.csv
    diag_df = pd.DataFrame({
        "tracklet_id": ["KEMX_T001", "KIWA_T001"],
        "radar_site": ["KEMX", "KIWA"],
        "status": ["plausible", "plausible"],
        "quality_label": ["excellent_plausible_tracklet", "excellent_plausible_tracklet"],
        "spaghetti_score": [12.5, 8.5],
        "reject_reason": ["", ""],
    })
    diag_df.to_csv(out_dir / "tracklet_quality_diagnostics.csv", index=False)

    # cross_radar_tracklet_associations.csv
    assoc_df = pd.DataFrame({
        "association_id": ["A001"],
        "radar_sites": ["KEMX;KIWA"],
        "tracklet_ids": ["KEMX_T001;KIWA_T001"],
        "n_radars": [2],
        "time_overlap_min": [0.0],
        "n_overlap_samples": [1],
        "median_altitude_difference_m": [100.0],
        "median_horizontal_difference_km": [2.5],
        "mean_telemetry_consistency_score": [0.8],
        "cross_radar_score": [0.85],
        "association_label": ["strong_cross_radar_candidate"],
        "notes": ["mock assoc"],
    })
    assoc_df.to_csv(out_dir / "cross_radar_tracklet_associations.csv", index=False)

    return config_file, case_dir / "outputs" / "ml"


def test_build_ml_feature_table(mock_case_env):
    config_file, ml_dir = mock_case_env

    test_args = ["build_ml_feature_table.py", str(config_file), "--primary-sites"]
    with patch.object(sys, "argv", test_args):
        build_feature_table_main()

    assert (ml_dir / "cluster_features.parquet").exists()
    assert (ml_dir / "tracklet_features.parquet").exists()
    assert (ml_dir / "manual_labels_template.csv").exists()

    # Verify content of cluster_features
    c_feat = pd.read_parquet(ml_dir / "cluster_features.parquet")
    assert "is_in_tracklet" in c_feat.columns
    assert "tracklet_id" in c_feat.columns
    assert "is_plausible" in c_feat.columns
    assert c_feat.loc[c_feat["cluster_id"] == 1, "is_in_tracklet"].any()

    # Verify content of tracklet_features
    t_feat = pd.read_parquet(ml_dir / "tracklet_features.parquet")
    assert "spaghetti_score" in t_feat.columns
    assert "is_associated" in t_feat.columns
    assert "n_associations" in t_feat.columns
    assert "mean_n_gates" in t_feat.columns
    assert t_feat.loc[t_feat["tracklet_id"] == "KEMX_T001", "is_associated"].iloc[0]

    # Verify labeling template contents
    template = pd.read_csv(ml_dir / "manual_labels_template.csv")
    assert "object_id" in template.columns
    assert "object_type" in template.columns
    assert "manual_label" in template.columns
    # Should have KEMX_T001, KIWA_T001 and high scoring clusters (balloon_like_cluster_score >= 0.5)
    assert "KEMX_T001" in template["object_id"].values


def test_select_examples_for_labeling(mock_case_env):
    config_file, ml_dir = mock_case_env

    # Run build script to generate inputs first
    test_args_build = ["build_ml_feature_table.py", str(config_file), "--primary-sites"]
    with patch.object(sys, "argv", test_args_build):
        build_feature_table_main()

    # Run selection script
    test_args_sel = ["select_examples_for_labeling.py", str(config_file)]
    with patch.object(sys, "argv", test_args_sel):
        select_examples_main()

    assert (ml_dir / "labeling_queue.csv").exists()
    queue = pd.read_csv(ml_dir / "labeling_queue.csv")
    assert len(queue) > 0
    assert "object_id" in queue.columns


def test_training_pipeline_graceful_exit_under_sparse_labels(mock_case_env):
    config_file, ml_dir = mock_case_env

    # Run build script first
    test_args_build = ["build_ml_feature_table.py", str(config_file), "--primary-sites"]
    with patch.object(sys, "argv", test_args_build):
        build_feature_table_main()

    # Case 1: manual_labels.csv is completely missing
    test_args_train = ["train_candidate_classifier.py", str(config_file)]
    with patch.object(sys, "argv", test_args_train):
        train_classifier_main()

    assert (ml_dir / "model_report.md").exists()
    with open(ml_dir / "model_report.md") as f:
        report_content = f.read()
    assert "Not enough labeled examples" in report_content

    # Case 2: manual_labels.csv has < 5 labeled tracklets
    labels_df = pd.DataFrame({
        "object_id": ["KEMX_T001"],
        "object_type": ["tracklet"],
        "manual_label": ["balloon_like"],
    })
    labels_df.to_csv(ml_dir / "manual_labels.csv", index=False)

    with patch.object(sys, "argv", test_args_train):
        train_classifier_main()

    with open(ml_dir / "model_report.md") as f:
        report_content = f.read()
    assert "Not enough labeled examples" in report_content


def test_successful_model_training(mock_case_env):
    config_file, ml_dir = mock_case_env

    # Set up mock tracklet features with 10 rows to support CV splits
    features_df = pd.DataFrame({
        "tracklet_id": [f"T_{i:03d}" for i in range(10)],
        "radar_site": ["KEMX"] * 10,
        "n_points": [5, 6, 7, 5, 6, 7, 5, 6, 7, 5],
        "duration_min": [20.0, 25.0, 30.0, 22.0, 28.0, 32.0, 21.0, 26.0, 29.0, 24.0],
        "median_abs_vertical_mismatch_m": [100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 110.0, 160.0, 190.0, 140.0],
        "max_abs_vertical_mismatch_m": [200.0, 250.0, 300.0, 220.0, 280.0, 320.0, 210.0, 260.0, 290.0, 240.0],
        "median_segment_speed_kmh": [35.0, 45.0, 55.0, 38.0, 48.0, 58.0, 36.0, 46.0, 56.0, 40.0],
        "max_segment_speed_kmh": [50.0, 60.0, 70.0, 52.0, 62.0, 72.0, 51.0, 61.0, 71.0, 55.0],
        "mean_balloon_like_score": [0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.82, 0.72, 0.62, 0.78],
        "path_smoothness_score": [0.8, 0.85, 0.9, 0.81, 0.86, 0.91, 0.79, 0.84, 0.89, 0.83],
        "altitude_consistency_score": [0.8, 0.85, 0.9, 0.81, 0.86, 0.91, 0.79, 0.84, 0.89, 0.83],
        "telemetry_match_score": [0.8, 0.85, 0.9, 0.81, 0.86, 0.91, 0.79, 0.84, 0.89, 0.83],
        "tracklet_score": [0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.82, 0.72, 0.62, 0.78],
        "spaghetti_score": [15.0, 20.0, 25.0, 16.0, 21.0, 26.0, 14.0, 19.0, 24.0, 18.0],
        "n_associations": [1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
        "mean_n_gates": [10.0] * 10,
        "max_n_gates": [12.0] * 10,
        "mean_max_reflectivity_dbz": [20.0] * 10,
        "max_max_reflectivity_dbz": [22.0] * 10,
        "mean_mean_reflectivity_dbz": [18.0] * 10,
        "mean_range_km": [100.0] * 10,
    })
    ml_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(ml_dir / "tracklet_features.parquet", index=False)

    # 10 labeled tracklets (5 balloon_like, 5 terrain_clutter_like)
    labels_df = pd.DataFrame({
        "object_id": [f"T_{i:03d}" for i in range(10)],
        "object_type": ["tracklet"] * 10,
        "manual_label": ["balloon_like"] * 5 + ["terrain_clutter_like"] * 5,
    })
    labels_df.to_csv(ml_dir / "manual_labels.csv", index=False)

    test_args_train = ["train_candidate_classifier.py", str(config_file)]
    with patch.object(sys, "argv", test_args_train):
        train_classifier_main()

    assert (ml_dir / "picocast_candidate_classifier.joblib").exists()
    assert (ml_dir / "feature_importance.csv").exists()
    assert (ml_dir / "candidate_ml_scores.csv").exists()
    assert (ml_dir / "model_report.md").exists()

    # Load and check model scores
    scores = pd.read_csv(ml_dir / "candidate_ml_scores.csv")
    assert len(scores) == 10
    assert "balloon_like_probability" in scores.columns
    assert "clutter_probability" in scores.columns
    assert "artifact_probability" in scores.columns
    assert "ml_best_class" in scores.columns
    assert "ml_confidence" in scores.columns

    # Load and check model package contents
    model_pkg = joblib.load(ml_dir / "picocast_candidate_classifier.joblib")
    assert "model" in model_pkg
    assert "features" in model_pkg
    assert "imputer" in model_pkg
    assert "scaler" in model_pkg
    assert "classes" in model_pkg
