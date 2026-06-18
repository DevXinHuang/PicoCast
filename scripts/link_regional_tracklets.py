#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 4: Branching forward tracklet linking for regional candidates."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    horizontal_distance_km,
)

haversine_distance_km = horizontal_distance_km

TRACKLET_SUMMARY_COLUMNS = [
    "case_id",
    "radar_site",
    "tracklet_id",
    "tracklet_rank",
    "n_points",
    "start_time_utc",
    "end_time_utc",
    "duration_min",
    "median_abs_vertical_mismatch_m",
    "max_abs_vertical_mismatch_m",
    "median_segment_speed_kmh",
    "max_segment_speed_kmh",
    "mean_balloon_like_score",
    "path_smoothness_score",
    "altitude_consistency_score",
    "telemetry_match_score",
    "tracklet_score",
    "tracklet_label",
    "notes",
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def segment_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return float((np.rad2deg(bearing) + 360.0) % 360.0)


def build_tracklets_for_radar(
    site: str,
    config_path: Path,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    case_dir = config_path.parent
    case_id = config["case_id"]
    
    out_dir = case_dir / "outputs" / "discovery" / site
    points_path = out_dir / "tracklet_points.csv"
    summary_path = out_dir / "candidate_tracklets.csv"
    
    if points_path.exists() and summary_path.exists() and not overwrite:
        print(f"Reloading existing tracklets for {site}")
        return
        
    clusters_path = out_dir / "discovered_clusters.parquet"
    if not clusters_path.exists():
        print(f"Skipping {site}: no discovered clusters found.")
        return
        
    df = pd.read_parquet(clusters_path)
    if df.empty:
        print(f"No candidate clusters found for {site}.")
        pd.DataFrame(columns=TRACKLET_SUMMARY_COLUMNS).to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(points_path, index=False)
        return
        
    # Parameters from config
    tl_cfg = config.get("tracklet_linking", {})
    max_tracklets = tl_cfg.get("max_tracklets_per_radar", 10)
    max_active = tl_cfg.get("max_active_paths_per_step", 200)
    max_speed = tl_cfg.get("max_segment_speed_kmh", 200.0)
    max_alt_jump = tl_cfg.get("max_altitude_jump_m", 2500.0)
    max_missing = tl_cfg.get("max_missing_scans", 2)
    dup_threshold = tl_cfg.get("duplicate_overlap_threshold", 0.7)
    
    # Sort and group chronologically
    df["time_dt"] = df["scan_time_utc"].apply(parse_utc)
    df = df.sort_values("time_dt").reset_index(drop=True)
    scan_times = df["time_dt"].unique()
    scan_time_to_idx = {t: idx for idx, t in enumerate(scan_times)}
    
    groups = {t: grp.to_dict("records") for t, grp in df.groupby("time_dt")}
    
    # Active paths queue
    # Each path is a dict: { "nodes": [...], "score": float, "last_bearing": float | None }
    active_paths: list[dict] = []
    
    # Branching forward search
    for t_idx, t in enumerate(scan_times):
        candidates = groups[t]
        
        # 1. Initialize new paths starting at this scan time
        for cand in candidates:
            # S_initial = 0.5 * cluster_score + 0.5 * altitude_score
            alt_score = 1.0 - cand["abs_vertical_distance_m"] / 1500.0
            initial_score = 0.5 * cand["balloon_like_cluster_score"] + 0.5 * alt_score
            active_paths.append({
                "nodes": [cand],
                "score": initial_score,
                "last_bearing": None,
                "missing_scans": 0,
            })
            
        # 2. Try to propagate existing active paths to these candidates
        new_active_paths = []
        for path in active_paths:
            last_node = path["nodes"][-1]
            last_t = last_node["time_dt"]
            last_t_idx = scan_time_to_idx[last_t]
            
            # Check missing scans limit
            scans_skipped = t_idx - last_t_idx - 1
            if scans_skipped < 0:
                # Same scan time, keep path as is
                new_active_paths.append(path)
                continue
            elif scans_skipped > max_missing:
                # Exceeded missing scans, do not propagate further (path is dead/finalized)
                new_active_paths.append(path)
                continue
                
            # Try to extend path with each candidate at time t
            dt_s = (t - last_t).total_seconds()
            for cand in candidates:
                # Physical constraints check
                # Ground speed
                dist_km = haversine_distance_km(
                    last_node["cluster_lat_deg"], last_node["cluster_lon_deg"],
                    cand["cluster_lat_deg"], cand["cluster_lon_deg"]
                )
                speed = dist_km / (dt_s / 3600.0) if dt_s > 0 else 0.0
                if speed > max_speed:
                    continue
                    
                # Altitude jump
                alt_jump = abs(last_node["cluster_alt_m"] - cand["cluster_alt_m"])
                if alt_jump > max_alt_jump:
                    continue
                    
                # Calculate scores
                alt_score = 1.0 - cand["abs_vertical_distance_m"] / 1500.0
                speed_score = 1.0 - speed / max_speed
                
                # Smoothness score
                bearing = segment_bearing_deg(
                    last_node["cluster_lat_deg"], last_node["cluster_lon_deg"],
                    cand["cluster_lat_deg"], cand["cluster_lon_deg"]
                )
                if path["last_bearing"] is not None:
                    db = abs(bearing - path["last_bearing"])
                    if db > 180.0:
                        db = 360.0 - db
                    smooth_score = 1.0 - db / 180.0
                else:
                    smooth_score = 1.0
                    
                segment_score = (
                    0.40 * cand["balloon_like_cluster_score"]
                    + 0.30 * alt_score
                    + 0.15 * speed_score
                    + 0.15 * smooth_score
                )
                
                # Compute cumulative average path score
                k = len(path["nodes"])
                new_score = (path["score"] * k + segment_score) / (k + 1)
                
                new_active_paths.append({
                    "nodes": path["nodes"] + [cand],
                    "score": new_score,
                    "last_bearing": bearing,
                    "missing_scans": 0,
                })
                
            # Also keep the path as is (allowing it to miss this scan time)
            path_copy = path.copy()
            path_copy["missing_scans"] = scans_skipped + 1
            new_active_paths.append(path_copy)
            
        # Prune active paths at this step to prevent combinatorial explosion
        # Sort by length * score, keeping at most max_active
        new_active_paths = sorted(
            new_active_paths,
            key=lambda x: x["score"] * (1.0 + 0.1 * min(len(x["nodes"]) - 1, 5)),
            reverse=True
        )
        active_paths = new_active_paths[:max_active]
        
    # Post-process: finalize all paths, filter by length >= 3
    final_paths = []
    for p in active_paths:
        if len(p["nodes"]) >= 3:
            final_paths.append(p)
            
    # Apply duplicate overlap pruning
    final_paths = sorted(
        final_paths,
        key=lambda x: x["score"] * (1.0 + 0.1 * min(len(x["nodes"]) - 1, 5)),
        reverse=True
    )
    
    pruned_paths: list[dict] = []
    for path in final_paths:
        # Check overlap with already selected paths
        is_duplicate = False
        path_node_ids = {n["cluster_id"] for n in path["nodes"]}
        for p_sel in pruned_paths:
            sel_node_ids = {n["cluster_id"] for n in p_sel["nodes"]}
            common = path_node_ids & sel_node_ids
            overlap = len(common) / min(len(path_node_ids), len(sel_node_ids))
            if overlap >= dup_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            pruned_paths.append(path)
            
    # Take top N
    top_paths = pruned_paths[:max_tracklets]
    print(f"Found {len(top_paths)} candidate tracklets for {site}")
    
    summary_rows = []
    point_rows = []
    
    for rank, path in enumerate(top_paths, start=1):
        nodes = path["nodes"]
        tracklet_id = f"{site}_T{rank:03d}"
        
        # Calculate speeds between nodes
        speeds = []
        for i in range(len(nodes) - 1):
            n0, n1 = nodes[i], nodes[i+1]
            dt = (n1["time_dt"] - n0["time_dt"]).total_seconds()
            dist = haversine_distance_km(n0["cluster_lat_deg"], n0["cluster_lon_deg"], n1["cluster_lat_deg"], n1["cluster_lon_deg"])
            speeds.append(dist / (dt / 3600.0) if dt > 0 else 0.0)
            
        # Calculate bearings for smoothness
        bearings = []
        for i in range(len(nodes) - 1):
            n0, n1 = nodes[i], nodes[i+1]
            bearings.append(segment_bearing_deg(n0["cluster_lat_deg"], n0["cluster_lon_deg"], n1["cluster_lat_deg"], n1["cluster_lon_deg"]))
            
        smoothness_scores = []
        for i in range(len(bearings) - 1):
            db = abs(bearings[i+1] - bearings[i])
            if db > 180.0:
                db = 360.0 - db
            smoothness_scores.append(1.0 - db / 180.0)
            
        smoothness_val = float(np.mean(smoothness_scores)) if smoothness_scores else 1.0
        
        # Altitude mismatch metrics
        abs_v_mismatches = [n["abs_vertical_distance_m"] for n in nodes]
        med_v_mismatch = float(np.median(abs_v_mismatches))
        max_v_mismatch = float(np.max(abs_v_mismatches))
        
        # Speeds
        med_speed = float(np.median(speeds)) if speeds else 0.0
        max_speed_val = float(np.max(speeds)) if speeds else 0.0
        
        # Corridor proximity
        inside_corridor = all(n["inside_or_near_grid_corridor"] for n in nodes)
        
        # Assign label
        if med_v_mismatch <= 500.0 and med_speed <= 100.0 and inside_corridor:
            label = "telemetry_consistent_candidate_tracklet"
        elif med_v_mismatch <= 1000.0 and med_speed <= 150.0:
            label = "altitude_consistent_candidate_tracklet"
        else:
            label = "plausible_radar_assisted_candidate_tracklet"
            
        # Build summary row
        duration = (nodes[-1]["time_dt"] - nodes[0]["time_dt"]).total_seconds() / 60.0
        mean_balloon_like = float(np.mean([n["balloon_like_cluster_score"] for n in nodes]))
        mean_alt_consistency = float(np.mean([1.0 - n["abs_vertical_distance_m"] / 1500.0 for n in nodes]))
        
        # Telemetry match score
        distance_scores = [1.0 - n["distance_to_nearest_grid_center_km"] / 40.0 for n in nodes]
        mean_dist_score = float(np.mean(distance_scores))
        telemetry_match = 0.5 * mean_alt_consistency + 0.5 * mean_dist_score
        
        # Length weighted final score
        final_score = path["score"] * (1.0 + 0.1 * min(len(nodes) - 1, 5))
        
        summary_rows.append({
            "case_id": case_id,
            "radar_site": site,
            "tracklet_id": tracklet_id,
            "tracklet_rank": rank,
            "n_points": len(nodes),
            "start_time_utc": nodes[0]["scan_time_utc"],
            "end_time_utc": nodes[-1]["scan_time_utc"],
            "duration_min": round(duration, 1),
            "median_abs_vertical_mismatch_m": round(med_v_mismatch, 1),
            "max_abs_vertical_mismatch_m": round(max_v_mismatch, 1),
            "median_segment_speed_kmh": round(med_speed, 1),
            "max_segment_speed_kmh": round(max_speed_val, 1),
            "mean_balloon_like_score": round(mean_balloon_like, 4),
            "path_smoothness_score": round(smoothness_val, 4),
            "altitude_consistency_score": round(mean_alt_consistency, 4),
            "telemetry_match_score": round(telemetry_match, 4),
            "tracklet_score": round(final_score, 4),
            "tracklet_label": label,
            "notes": f"Tracklet linking: {len(nodes)} points, score={final_score:.3f}",
        })
        
        # Add points to list
        for node in nodes:
            node_copy = node.copy()
            # Remove timestamp objects to keep serialization clean
            node_copy.pop("time_dt", None)
            node_copy["tracklet_id"] = tracklet_id
            node_copy["tracklet_rank"] = rank
            point_rows.append(node_copy)
            
    summary_df = pd.DataFrame(summary_rows, columns=TRACKLET_SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary table: {summary_path}")
    
    if point_rows:
        points_df = pd.DataFrame(point_rows)
        # Reorder to put tracklet_id and tracklet_rank first
        cols = ["tracklet_id", "tracklet_rank"] + [c for c in points_df.columns if c not in ["tracklet_id", "tracklet_rank"]]
        points_df = points_df[cols]
        points_df.to_csv(points_path, index=False)
        print(f"Wrote tracklet points table: {points_path}")
    else:
        pd.DataFrame().to_csv(points_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Link regional candidates into tracklets.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already linked tracklets")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    
    # Read geometry report
    geometry_csv_path = case_dir / "nexrad" / "regional_radar_geometry.csv"
    if not geometry_csv_path.exists():
        raise FileNotFoundError(f"Geometry report {geometry_csv_path} not found.")
        
    geom_df = pd.read_csv(geometry_csv_path)
    included_radars = geom_df[geom_df["geometry_status"] == "include"]["radar_site"].tolist()
    
    discovery = config.get("discovery", {})
    primary = discovery.get("radar_sites_primary", [])
    secondary = discovery.get("radar_sites_secondary", [])
    
    if args.radar_site:
        target_sites = [args.radar_site]
    elif args.all_sites:
        target_sites = primary + secondary
    elif args.primary_sites:
        target_sites = primary
    else:
        target_sites = primary
        
    active_sites = [r for r in target_sites if r in included_radars]
    print(f"Linking tracklets for active sites: {', '.join(active_sites)}")
    
    for site in active_sites:
        build_tracklets_for_radar(site, args.config, args.overwrite)


if __name__ == "__main__":
    main()
