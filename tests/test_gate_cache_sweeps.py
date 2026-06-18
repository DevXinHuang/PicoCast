#!/usr/bin/env python
# ruff: noqa: E501
"""Unit tests for gate cache builder and parameter sweeps."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MockRadar:
    def __init__(self):
        nrays = 3
        ngates = 5
        self.nrays = nrays
        self.ngates = ngates
        self.gate_latitude = {"data": np.zeros((nrays, ngates)) + 32.27}
        self.gate_longitude = {"data": np.zeros((nrays, ngates)) - 110.96}
        self.gate_altitude = {"data": np.zeros((nrays, ngates)) + 2340.0}
        self.azimuth = {"data": np.array([0.0, 90.0, 180.0])}
        self.elevation = {"data": np.array([0.5, 0.5, 0.5])}
        self.range = {"data": np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])}
        
        self.fields = {
            "reflectivity": {"data": np.zeros((nrays, ngates)) + 10.0},
            "velocity": {"data": np.zeros((nrays, ngates)) + 5.0},
            "spectrum_width": {"data": np.zeros((nrays, ngates)) + 1.0},
            "cross_correlation_ratio": {"data": np.zeros((nrays, ngates)) + 0.98},
        }


@pytest.fixture
def mock_case_env(tmp_path):
    case_dir = tmp_path / "cases" / "test_case"
    case_dir.mkdir(parents=True)
    
    # 1. config.yaml
    config = {
        "case_id": "test_case",
        "gate_cache": {
            "enabled": True,
            "output_format": "parquet",
            "compression": "zstd",
            "float_precision": "float32",
            "cache_all_valid_gates": True,
            "cache_only_discovery_region": False,
            "overwrite": True
        },
        "discovery": {
            "radar_sites_primary": ["KEMX"],
            "radar_sites_secondary": [],
            "altitude_padding_m": 1500,
            "horizontal_corridor_padding_km": 40,
            "reflectivity_max_dbz": 20,
            "max_gates_per_scan_to_cluster": 200000,
            "min_gates_per_cluster": 1,
            "max_gates_per_cluster": 25,
        },
        "performance": {
            "max_workers": 1,
            "process_one_file_at_a_time": True,
            "do_not_keep_all_gates_in_memory": True
        },
        "tracklet_linking": {
            "max_tracklets_per_radar": 5,
            "max_active_paths_per_step": 50,
            "max_segment_speed_kmh": 200,
            "max_altitude_jump_m": 2500,
            "max_missing_scans": 2,
            "duplicate_overlap_threshold": 0.7
        }
    }
    
    config_path = case_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # 2. expected_track.csv
    track_df = pd.DataFrame({
        "lat_deg": [32.27, 32.28, 32.29],
        "lon_deg": [-110.96, -110.84, -110.72],
        "alt_m": [2340.0, 3200.0, 4000.0],
        "maidenhead_grid": ["DM42mg", "DM42ng", "DM42og"]
    })
    track_df.to_csv(case_dir / "expected_track.csv", index=False)
    
    # 3. regional_radar_geometry.csv
    geom_df = pd.DataFrame({
        "radar_site": ["KEMX"],
        "geometry_status": ["include"]
    })
    geom_dir = case_dir / "nexrad"
    geom_dir.mkdir(parents=True)
    geom_df.to_csv(geom_dir / "regional_radar_geometry.csv", index=False)
    
    # 4. scan_track_matches.csv
    match_df = pd.DataFrame({
        "case_id": ["test_case", "test_case", "test_case"],
        "radar_site": ["KEMX", "KEMX", "KEMX"],
        "scan_time_utc": ["2026-03-22T19:00:00Z", "2026-03-22T19:10:00Z", "2026-03-22T19:20:00Z"],
        "scan_filename": ["KEMX20260322_190000_V06", "KEMX20260322_191000_V06", "KEMX20260322_192000_V06"],
        "scan_local_path": ["some_path", "some_path", "some_path"],
        "expected_lat_deg": [32.27, 32.28, 32.29],
        "expected_lon_deg": [-110.96, -110.84, -110.72],
        "expected_alt_m": [2340.0, 3200.0, 4000.0],
        "is_geometrically_visible": [True, True, True]
    })
    index_dir = geom_dir / "KEMX" / "index"
    index_dir.mkdir(parents=True)
    match_df.to_csv(index_dir / "scan_track_matches.csv", index=False)
    
    # 5. create empty raw files so local_path.exists() is True
    raw_dir = geom_dir / "KEMX" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "KEMX20260322_190000_V06").touch()
    (raw_dir / "KEMX20260322_191000_V06").touch()
    (raw_dir / "KEMX20260322_192000_V06").touch()
    
    return case_dir, config_path


def test_gate_cache_and_sweeps(mock_case_env):
    case_dir, config_path = mock_case_env
    
    # Mock pyart reading
    with patch("pyart.io.read_nexrad_archive", return_value=MockRadar()):
        from scripts.build_gate_feature_cache import build_radar_gate_cache
        
        # Test cache building
        records = build_radar_gate_cache("KEMX", config_path, overwrite=True)
        assert len(records) == 3
        assert records[0]["n_cached_gates"] > 0
        
        # Verify parquet existence
        cache_parquet_dir = case_dir / "cache" / "gates" / "KEMX"
        assert cache_parquet_dir.exists()
        parquets = list(cache_parquet_dir.glob("*.parquet"))
        assert len(parquets) == 3
        
        # Test inventory creation
        from scripts.build_gate_feature_cache import main as cache_main
        with patch("sys.argv", ["build_gate_feature_cache.py", str(config_path), "--primary-sites"]):
            cache_main()
            
        inventory_path = case_dir / "cache" / "gates" / "gate_cache_inventory.csv"
        assert inventory_path.exists()
        inv_df = pd.read_csv(inventory_path)
        assert len(inv_df) == 3
        
        # Test discovery script using cache
        from scripts.discover_regional_balloon_like_clusters import main as discovery_main
        with patch("sys.argv", ["discover_regional_balloon_like_clusters.py", str(config_path), "--primary-sites", "--use-gate-cache", "--overwrite"]):
            discovery_main()
            
        # Verify cluster output
        cluster_parquet = case_dir / "outputs" / "discovery" / "KEMX" / "discovered_clusters.parquet"
        assert cluster_parquet.exists()
        cluster_df = pd.read_parquet(cluster_parquet)
        assert len(cluster_df) > 0
        
        # Test parameter sweep script using the cached gates
        from scripts.run_discovery_parameter_sweep import main as sweep_main
        with patch("sys.argv", ["run_discovery_parameter_sweep.py", str(config_path), "--primary-sites"]):
            sweep_main()
            
        # Verify sweep outputs
        sweep_csv = case_dir / "outputs" / "sweeps" / "discovery_parameter_sweep.csv"
        stability_csv = case_dir / "outputs" / "sweeps" / "tracklet_stability.csv"
        
        assert sweep_csv.exists()
        assert stability_csv.exists()
        
        sweep_df = pd.read_csv(sweep_csv)
        stability_df = pd.read_csv(stability_csv)
        
        assert len(sweep_df) > 0
        assert "sweep_id" in sweep_df.columns
        assert "n_clusters" in sweep_df.columns
        
        assert len(stability_df) > 0
        assert "stability_label" in stability_df.columns
