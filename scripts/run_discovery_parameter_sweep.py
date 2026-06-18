#!/usr/bin/env python
# ruff: noqa: E501
"""Run discovery and tracklet linking parameter sweeps using cached gates."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Suppress Py-ART and deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    bounded_score,
    horizontal_distance_km,
    local_xy_km,
    reflectivity_score,
)
from scripts.discover_regional_balloon_like_clusters import (  # noqa: E402
    distance_to_polyline_km,
    get_track_polyline_segments,
    run_dbscan,
)
from scripts.link_regional_tracklets import (  # noqa: E402
    parse_utc,
    segment_bearing_deg,
)

haversine_distance_km = horizontal_distance_km


class TrackletCandidateGroup:
    def __init__(self, first_tracklet, group_id: int):
        self.group_id = group_id
        self.radar_site = first_tracklet["radar_site"]
        self.scan_times = set([n["scan_time_utc"] for n in first_tracklet["nodes"]])
        self.sweep_ids = set()
        if "sweep_id" in first_tracklet:
            self.sweep_ids.add(first_tracklet["sweep_id"])
        self.members = [first_tracklet]
        
    def matches(self, other_tracklet) -> bool:
        if self.radar_site != other_tracklet["radar_site"]:
            return False
        other_times = set([n["scan_time_utc"] for n in other_tracklet["nodes"]])
        intersection = self.scan_times & other_times
        union = self.scan_times | other_times
        if not union:
            return False
        return (len(intersection) / len(union)) >= 0.5
        
    def add(self, other_tracklet):
        self.members.append(other_tracklet)
        self.scan_times.update([n["scan_time_utc"] for n in other_tracklet["nodes"]])
        if "sweep_id" in other_tracklet:
            self.sweep_ids.add(other_tracklet["sweep_id"])


def stability_label(detection_fraction: float) -> str:
    if detection_fraction >= 0.75:
        return "stable_candidate"
    if detection_fraction >= 0.35:
        return "moderately_stable_candidate"
    return "unstable_candidate"


def compute_stability_rows(
    candidate_groups: list[TrackletCandidateGroup],
    n_possible_sweeps: int,
) -> list[dict]:
    stability_rows = []

    for g in candidate_groups:
        scan_times_sorted = sorted(g.scan_times)
        start_time = scan_times_sorted[0] if scan_times_sorted else ""
        end_time = scan_times_sorted[-1] if scan_times_sorted else ""

        n_points_median = float(np.median([len(m["nodes"]) for m in g.members]))
        duration_median = float(np.median([m["duration_min"] for m in g.members]))

        member_tracklet_count = len(g.members)
        detection_count = len(g.sweep_ids) if g.sweep_ids else member_tracklet_count
        detection_fraction = detection_count / n_possible_sweeps if n_possible_sweeps else 0.0
        detection_fraction = float(np.clip(detection_fraction, 0.0, 1.0))
        label = stability_label(detection_fraction)

        tracklet_sig = f"{g.radar_site}_{start_time}_{end_time}"

        stability_rows.append({
            "tracklet_signature": tracklet_sig,
            "radar_site": g.radar_site,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "n_points_median": round(n_points_median, 1),
            "duration_min_median": round(duration_median, 1),
            "detection_count": detection_count,
            "member_tracklet_count": member_tracklet_count,
            "n_possible_sweeps": n_possible_sweeps,
            "detection_fraction": round(detection_fraction, 4),
            "stability_label": label,
        })

    return stability_rows


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_candidates_from_cache(
    cache_df: pd.DataFrame,
    alt_padding: float,
    horiz_padding: float,
    refl_max: float,
    min_gates: int,
    max_gates: int,
    track_lats: np.ndarray,
    track_lons: np.ndarray,
    track_grids: list[str],
    max_gates_to_cluster: int = 200000,
) -> list[dict]:
    # Apply thresholds
    alt_mask = cache_df["abs_vertical_distance_m"] <= alt_padding
    corridor_mask = cache_df["distance_to_track_corridor_km"] <= horiz_padding
    
    refl_val = cache_df["reflectivity_dbz"].to_numpy()
    valid_refl_mask = (~np.isnan(refl_val)) & (refl_val <= refl_max)
    
    combined_mask = alt_mask & corridor_mask & valid_refl_mask
    filtered_df = cache_df[combined_mask].copy()
    if filtered_df.empty:
        return []
        
    lats = filtered_df["gate_lat_deg"].to_numpy()
    if len(lats) > max_gates_to_cluster:
        return []
    lons = filtered_df["gate_lon_deg"].to_numpy()
    alts = filtered_df["gate_alt_m"].to_numpy()
    refl = filtered_df["reflectivity_dbz"].to_numpy()
    vel = filtered_df["velocity_ms"].to_numpy()
    sw = filtered_df["spectrum_width_ms"].to_numpy()
    rhohv = filtered_df["rhohv"].to_numpy()
    
    azimuths = filtered_df["azimuth_deg"].to_numpy()
    elevations = filtered_df["elevation_deg"].to_numpy()
    ranges = filtered_df["range_km"].to_numpy()
    
    expected_alt_m = float(filtered_df["expected_alt_m"].iloc[0])
    expected_lat = float(filtered_df["expected_lat_deg"].iloc[0])
    expected_lon = float(filtered_df["expected_lon_deg"].iloc[0])
    scan_time_str = str(filtered_df["scan_time_utc"].iloc[0])
    filename = str(filtered_df["scan_filename"].iloc[0])
    case_id = str(filtered_df["case_id"].iloc[0])
    site = str(filtered_df["radar_site"].iloc[0])
    
    origin_lat = expected_lat
    origin_lon = expected_lon
    gx, gy = local_xy_km(lats, lons, origin_lat, origin_lon)
    gz = (alts - expected_alt_m) / 1000.0
    coords_km = np.column_stack([gx, gy, gz])
    
    labels = run_dbscan(coords_km, eps_km=1.0)
    unique_labels = sorted(set(labels))
    
    stamp = scan_time_str.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    
    candidates = []
    for label in unique_labels:
        if label < 0:
            continue
        label_mask = labels == label
        n_pts = int(np.sum(label_mask))
        
        if not (min_gates <= n_pts <= max_gates):
            continue
            
        cluster_id = f"{site}_{stamp}_{label:03d}"
        
        c_lats = lats[label_mask]
        c_lons = lons[label_mask]
        c_alts = alts[label_mask]
        c_refl = refl[label_mask]
        c_vel = vel[label_mask]
        c_sw = sw[label_mask]
        c_rho = rhohv[label_mask]
        
        center_lat = float(np.mean(c_lats))
        center_lon = float(np.mean(c_lons))
        center_alt = float(np.mean(c_alts))
        
        cx, cy = local_xy_km(c_lats, c_lons, center_lat, center_lon)
        cz = (c_alts - center_alt) / 1000.0
        compactness = float(np.sqrt(cx**2 + cy**2 + cz**2).max()) if len(cx) > 0 else 0.0
        
        dist_to_grid_center = float(haversine_distance_km(center_lat, center_lon, expected_lat, expected_lon))
        dist_to_corridor = float(distance_to_polyline_km(np.array([center_lat]), np.array([center_lon]), track_lats, track_lons)[0])
        
        dists_to_track_points = [haversine_distance_km(center_lat, center_lon, p_lat, p_lon)
                                 for p_lat, p_lon in zip(track_lats, track_lons, strict=True)]
        nearest_idx = int(np.argmin(dists_to_track_points))
        nearest_grid = track_grids[nearest_idx]
        
        inside_corridor = dist_to_corridor <= horiz_padding
        
        distance_score = bounded_score(dist_to_grid_center, 0.0, 40.0)
        v_mismatch = abs(center_alt - expected_alt_m)
        altitude_score = bounded_score(v_mismatch, 0.0, 1500.0)
        refl_score_val = reflectivity_score(float(np.max(c_refl)), float(np.percentile(c_refl, 95)) if len(c_refl) > 1 else float(c_refl[0]))
        compact_score = bounded_score(compactness, 0.0, 2.0)
        isolation_score = bounded_score(float(n_pts), 1.0, 25.0)
        
        composite_score = (
            0.20 * distance_score
            + 0.30 * altitude_score
            + 0.20 * refl_score_val
            + 0.15 * compact_score
            + 0.15 * isolation_score
        )
        
        candidates.append({
            "case_id": case_id,
            "radar_site": site,
            "scan_time_utc": scan_time_str,
            "scan_filename": filename,
            "cluster_id": cluster_id,
            "cluster_lat_deg": round(center_lat, 6),
            "cluster_lon_deg": round(center_lon, 6),
            "cluster_alt_m": round(center_alt, 1),
            "n_gates": n_pts,
            "max_reflectivity_dbz": round(float(np.max(c_refl)), 1),
            "mean_reflectivity_dbz": round(float(np.mean(c_refl)), 2),
            "p95_reflectivity_dbz": round(float(np.percentile(c_refl, 95)) if len(c_refl) > 1 else float(c_refl[0]), 1),
            "velocity_mean_ms": round(float(np.nanmean(c_vel)), 2) if not np.all(np.isnan(c_vel)) else np.nan,
            "spectrum_width_mean_ms": round(float(np.nanmean(c_sw)), 2) if not np.all(np.isnan(c_sw)) else np.nan,
            "rhohv_mean": round(float(np.nanmean(c_rho)), 3) if not np.all(np.isnan(c_rho)) else np.nan,
            "compactness_km": round(compactness, 3),
            "range_km": round(float(np.mean(ranges[label_mask])), 2),
            "azimuth_deg": round(float(np.mean(azimuths[label_mask])), 1),
            "elevation_deg": round(float(np.mean(elevations[label_mask])), 2),
            "expected_alt_m": round(expected_alt_m, 1),
            "signed_vertical_m": round(center_alt - expected_alt_m, 1),
            "abs_vertical_distance_m": round(v_mismatch, 1),
            "nearest_grid": nearest_grid,
            "distance_to_nearest_grid_center_km": round(dist_to_grid_center, 2),
            "distance_to_track_corridor_km": round(dist_to_corridor, 2),
            "inside_or_near_grid_corridor": inside_corridor,
            "balloon_like_cluster_score": round(composite_score, 4),
            "notes": "DBSCAN cluster discovery candidate",
        })
    return candidates


def link_candidates_into_tracklets(candidates: list[dict], tl_cfg: dict) -> list[dict]:
    if not candidates:
        return []
        
    df = pd.DataFrame(candidates)
    
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
    
    active_paths: list[dict] = []
    
    for t_idx, t in enumerate(scan_times):
        cands = groups[t]
        
        # 1. Initialize
        for cand in cands:
            alt_score = 1.0 - cand["abs_vertical_distance_m"] / 1500.0
            initial_score = 0.5 * cand["balloon_like_cluster_score"] + 0.5 * alt_score
            active_paths.append({
                "nodes": [cand],
                "score": initial_score,
                "last_bearing": None,
                "missing_scans": 0,
            })
            
        # 2. Propagate
        new_active_paths = []
        for path in active_paths:
            last_node = path["nodes"][-1]
            last_t = last_node["time_dt"]
            last_t_idx = scan_time_to_idx[last_t]
            
            scans_skipped = t_idx - last_t_idx - 1
            if scans_skipped < 0:
                new_active_paths.append(path)
                continue
            elif scans_skipped > max_missing:
                new_active_paths.append(path)
                continue
                
            dt_s = (t - last_t).total_seconds()
            for cand in cands:
                dist_km = haversine_distance_km(
                    last_node["cluster_lat_deg"], last_node["cluster_lon_deg"],
                    cand["cluster_lat_deg"], cand["cluster_lon_deg"]
                )
                speed = dist_km / (dt_s / 3600.0) if dt_s > 0 else 0.0
                if speed > max_speed:
                    continue
                    
                alt_jump = abs(last_node["cluster_alt_m"] - cand["cluster_alt_m"])
                if alt_jump > max_alt_jump:
                    continue
                    
                alt_score = 1.0 - cand["abs_vertical_distance_m"] / 1500.0
                speed_score = 1.0 - speed / max_speed
                
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
                
                k = len(path["nodes"])
                new_score = (path["score"] * k + segment_score) / (k + 1)
                
                new_active_paths.append({
                    "nodes": path["nodes"] + [cand],
                    "score": new_score,
                    "last_bearing": bearing,
                    "missing_scans": 0,
                })
                
            path_copy = path.copy()
            path_copy["missing_scans"] = scans_skipped + 1
            new_active_paths.append(path_copy)
            
        new_active_paths = sorted(
            new_active_paths,
            key=lambda x: x["score"] * (1.0 + 0.1 * min(len(x["nodes"]) - 1, 5)),
            reverse=True
        )
        active_paths = new_active_paths[:max_active]
        
    final_paths = []
    for p in active_paths:
        if len(p["nodes"]) >= 3:
            final_paths.append(p)
            
    final_paths = sorted(
        final_paths,
        key=lambda x: x["score"] * (1.0 + 0.1 * min(len(x["nodes"]) - 1, 5)),
        reverse=True
    )
    
    pruned_paths: list[dict] = []
    for path in final_paths:
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
            
    top_paths = pruned_paths[:max_tracklets]
    
    linked_tracklets = []
    for rank, path in enumerate(top_paths, start=1):
        nodes = path["nodes"]
        site = nodes[0]["radar_site"]
        tracklet_id = f"{site}_T{rank:03d}"
        
        speeds = []
        for i in range(len(nodes) - 1):
            n0, n1 = nodes[i], nodes[i+1]
            dt = (n1["time_dt"] - n0["time_dt"]).total_seconds()
            dist = haversine_distance_km(n0["cluster_lat_deg"], n0["cluster_lon_deg"], n1["cluster_lat_deg"], n1["cluster_lon_deg"])
            speeds.append(dist / (dt / 3600.0) if dt > 0 else 0.0)
            
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
        abs_v_mismatches = [n["abs_vertical_distance_m"] for n in nodes]
        med_v_mismatch = float(np.median(abs_v_mismatches))
        max_v_mismatch = float(np.max(abs_v_mismatches))
        
        med_speed = float(np.median(speeds)) if speeds else 0.0
        max_speed_val = float(np.max(speeds)) if speeds else 0.0
        
        inside_corridor = all(n["inside_or_near_grid_corridor"] for n in nodes)
        
        if med_v_mismatch <= 500.0 and med_speed <= 100.0 and inside_corridor:
            label = "telemetry_consistent_candidate_tracklet"
        elif med_v_mismatch <= 1000.0 and med_speed <= 150.0:
            label = "altitude_consistent_candidate_tracklet"
        else:
            label = "plausible_radar_assisted_candidate_tracklet"
            
        duration = (nodes[-1]["time_dt"] - nodes[0]["time_dt"]).total_seconds() / 60.0
        mean_balloon_like = float(np.mean([n["balloon_like_cluster_score"] for n in nodes]))
        mean_alt_consistency = float(np.mean([1.0 - n["abs_vertical_distance_m"] / 1500.0 for n in nodes]))
        
        distance_scores = [1.0 - n["distance_to_nearest_grid_center_km"] / 40.0 for n in nodes]
        mean_dist_score = float(np.mean(distance_scores))
        telemetry_match = 0.5 * mean_alt_consistency + 0.5 * mean_dist_score
        
        final_score = path["score"] * (1.0 + 0.1 * min(len(nodes) - 1, 5))
        
        linked_tracklets.append({
            "radar_site": site,
            "tracklet_id": tracklet_id,
            "tracklet_rank": rank,
            "n_points": len(nodes),
            "duration_min": duration,
            "median_abs_vertical_mismatch_m": med_v_mismatch,
            "max_abs_vertical_mismatch_m": max_v_mismatch,
            "median_segment_speed_kmh": med_speed,
            "max_segment_speed_kmh": max_speed_val,
            "mean_balloon_like_score": mean_balloon_like,
            "path_smoothness_score": smoothness_val,
            "altitude_consistency_score": mean_alt_consistency,
            "telemetry_match_score": telemetry_match,
            "tracklet_score": final_score,
            "tracklet_label": label,
            "nodes": nodes,
        })
        
    return linked_tracklets


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweeps on cached gates.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    
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
    print(f"Sweeping parameters for active sites: {', '.join(active_sites)}")
    
    # Sweep grid arrays
    alt_paddings = [500, 1000, 1500, 2000]
    horiz_paddings = [10, 20, 40, 60]
    refl_maxes = [5.0, 10.0, 20.0]
    max_gates_options = [5, 10, 25]
    
    # Tracklet linking config
    tl_cfg = config.get("tracklet_linking", {})
    max_gates_to_cluster = discovery.get("max_gates_per_scan_to_cluster", 200000)
    
    # Load track polylines
    track_path = case_dir / "expected_track.csv"
    track_lats, track_lons, track_grids = get_track_polyline_segments(track_path)
    
    sweep_results = []
    all_sweep_tracklets = []
    
    # Total combinations = 4 * 4 * 3 * 3 = 144
    total_combinations = len(alt_paddings) * len(horiz_paddings) * len(refl_maxes) * len(max_gates_options)
    
    # To keep memory footprint low, process one radar site at a time
    for site in active_sites:
        # Load all cached parquet files for this site
        site_parquet_dir = case_dir / "cache" / "gates" / site
        if not site_parquet_dir.exists():
            print(f"Skipping {site}: no gate cache directory found.")
            continue
            
        parquet_files = sorted(site_parquet_dir.glob(f"{site}_*.parquet"))
        if not parquet_files:
            print(f"Skipping {site}: no cached parquets found.")
            continue
            
        print(f"Loading {len(parquet_files)} cached scans for {site}...")
        scans_df_list = []
        for pf in parquet_files:
            try:
                scans_df_list.append(pd.read_parquet(pf))
            except Exception as e:
                print(f"Error loading {pf.name}: {e}")
                
        if not scans_df_list:
            continue
            
        # Run grid search
        sweep_counter = 0
        pbar = tqdm(total=total_combinations, desc=f"Sweeping {site}")
        for alt_pad in alt_paddings:
            for horiz_pad in horiz_paddings:
                for refl_max in refl_maxes:
                    for max_gates in max_gates_options:
                        sweep_counter += 1
                        sweep_id = f"SWEEP_{sweep_counter:03d}"
                        
                        # Candidate cluster discovery on each scan
                        candidates = []
                        for scan_df in scans_df_list:
                            cands = discover_candidates_from_cache(
                                scan_df,
                                alt_padding=alt_pad,
                                horiz_padding=horiz_pad,
                                refl_max=refl_max,
                                min_gates=1,
                                max_gates=max_gates,
                                track_lats=track_lats,
                                track_lons=track_lons,
                                track_grids=track_grids,
                                max_gates_to_cluster=max_gates_to_cluster,
                            )
                            candidates.extend(cands)
                            
                        # Link candidates into tracklets
                        tracklets = link_candidates_into_tracklets(candidates, tl_cfg)
                        
                        # Compute metrics
                        n_clusters = len(candidates)
                        n_tracklets = len(tracklets)
                        n_consistent = sum(
                            1 for t in tracklets
                            if t["tracklet_label"] in ["telemetry_consistent_candidate_tracklet", "altitude_consistent_candidate_tracklet"]
                        )
                        mean_tracklet_score = float(np.mean([t["tracklet_score"] for t in tracklets])) if tracklets else 0.0
                        mean_balloon_score = float(np.mean([t["mean_balloon_like_score"] for t in tracklets])) if tracklets else 0.0
                        
                        sweep_results.append({
                            "sweep_id": sweep_id,
                            "radar_site": site,
                            "altitude_padding_m": alt_pad,
                            "horizontal_corridor_padding_km": horiz_pad,
                            "reflectivity_max_dbz": refl_max,
                            "max_gates_per_cluster": max_gates,
                            "n_clusters": n_clusters,
                            "n_tracklets": n_tracklets,
                            "n_consistent_tracklets": n_consistent,
                            "mean_tracklet_score": round(mean_tracklet_score, 4),
                            "mean_balloon_like_score": round(mean_balloon_score, 4),
                        })
                        
                        # Collect tracklets for stability analysis
                        for t in tracklets:
                            t_copy = t.copy()
                            t_copy["sweep_id"] = sweep_id
                            all_sweep_tracklets.append(t_copy)
                            
                        pbar.update(1)
        pbar.close()
        
    # Write sweep results
    out_dir = case_dir / "outputs" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    SWEEP_COLUMNS = [
        "sweep_id",
        "radar_site",
        "altitude_padding_m",
        "horizontal_corridor_padding_km",
        "reflectivity_max_dbz",
        "max_gates_per_cluster",
        "n_clusters",
        "n_tracklets",
        "n_consistent_tracklets",
        "mean_tracklet_score",
        "mean_balloon_like_score",
    ]
    sweep_csv_path = out_dir / "discovery_parameter_sweep.csv"
    sweep_df = pd.DataFrame(sweep_results, columns=SWEEP_COLUMNS)
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"Wrote parameter sweep results to {sweep_csv_path}")
    
    # Run tracklet stability analysis
    print("Running tracklet stability analysis...")
    candidate_groups: list[TrackletCandidateGroup] = []
    
    for t in all_sweep_tracklets:
        matched = False
        for g in candidate_groups:
            if g.matches(t):
                g.add(t)
                matched = True
                break
        if not matched:
            group_id = len(candidate_groups) + 1
            candidate_groups.append(TrackletCandidateGroup(t, group_id))
            
    # Calculate stability metrics. Count unique sweep configurations instead
    # of raw member tracklets so the detection fraction is bounded by 1.0.
    stability_rows = compute_stability_rows(candidate_groups, total_combinations)
        
    STABILITY_COLUMNS = [
        "tracklet_signature",
        "radar_site",
        "start_time_utc",
        "end_time_utc",
        "n_points_median",
        "duration_min_median",
        "detection_count",
        "member_tracklet_count",
        "n_possible_sweeps",
        "detection_fraction",
        "stability_label",
    ]
    stability_csv_path = out_dir / "tracklet_stability.csv"
    stability_df = pd.DataFrame(stability_rows, columns=STABILITY_COLUMNS)
    stability_df.to_csv(stability_csv_path, index=False)
    print(f"Wrote tracklet stability results to {stability_csv_path}")


if __name__ == "__main__":
    main()
