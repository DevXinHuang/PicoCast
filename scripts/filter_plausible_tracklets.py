#!/usr/bin/env python
# ruff: noqa: E501, E402
"""Filter plausible tracklets based on strict physical and corridor constraints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (
    horizontal_distance_km,
    load_config,
)

haversine_distance_km = horizontal_distance_km


def segment_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return float((np.rad2deg(bearing) + 360.0) % 360.0)


def compute_spaghetti_score(summary_row: pd.Series, t_pts: pd.DataFrame, all_pts_df: pd.DataFrame) -> float:
    # 1. Smoothness penalty
    smooth_score = summary_row.get("path_smoothness_score", 1.0)
    smooth_penalty = 100.0 * (1.0 - smooth_score)
    
    # 2. Speed jump penalty
    med_speed = summary_row.get("median_segment_speed_kmh", 0.0)
    max_speed = summary_row.get("max_segment_speed_kmh", 0.0)
    speed_jump_penalty = 10.0 * (max_speed / (med_speed + 1.0))
    
    # 3. Bearing reversals
    reversals = 0
    if len(t_pts) >= 3:
        lats = t_pts["cluster_lat_deg"].values
        lons = t_pts["cluster_lon_deg"].values
        bearings = []
        for i in range(len(lats) - 1):
            bearings.append(segment_bearing_deg(lats[i], lons[i], lats[i+1], lons[i+1]))
        for i in range(len(bearings) - 1):
            db = abs(bearings[i+1] - bearings[i])
            if db > 180.0:
                db = 360.0 - db
            if db >= 135.0:  # reversal
                reversals += 1
    reversal_penalty = reversals * 50.0
    
    # 4. Overlap count (with other tracklets)
    overlap_count = 0
    t_coords = set(zip(t_pts["scan_time_utc"], t_pts["cluster_lat_deg"], t_pts["cluster_lon_deg"], strict=True))
    
    other_pts = all_pts_df[all_pts_df["tracklet_id"] != summary_row["tracklet_id"]]
    if not other_pts.empty:
        other_coords = set(zip(other_pts["scan_time_utc"], other_pts["cluster_lat_deg"], other_pts["cluster_lon_deg"], strict=True))
        overlap_count = len(t_coords.intersection(other_coords))
    overlap_penalty = overlap_count * 20.0
    
    # 5. Far from corridor penalty
    mean_dist_corridor = t_pts["distance_to_track_corridor_km"].mean() if "distance_to_track_corridor_km" in t_pts.columns else 0.0
    corridor_penalty = 2.0 * mean_dist_corridor
    
    score = smooth_penalty + speed_jump_penalty + reversal_penalty + overlap_penalty + corridor_penalty
    return float(score)


def evaluate_tracklet_quality(row: pd.Series, t_pts: pd.DataFrame, all_points_df: pd.DataFrame) -> dict:
    tid = row["tracklet_id"]
    t_pts = t_pts.sort_values("scan_time_utc")
    
    # Calculate spaghetti score
    spag_score = compute_spaghetti_score(row, t_pts, all_points_df)
    
    # Calculate corridor fraction
    corridor_col = "inside_or_near_grid_corridor"
    if corridor_col not in t_pts.columns and "inside_or_near_grid_corridor" in all_points_df.columns:
        corridor_col = "inside_or_near_grid_corridor"
    elif "inside_or_near_grid_corridor" not in t_pts.columns:
        t_pts["inside_or_near_grid_corridor"] = t_pts["distance_to_track_corridor_km"] <= 40.0
        corridor_col = "inside_or_near_grid_corridor"
        
    corridor_fraction = t_pts[corridor_col].mean() if len(t_pts) > 0 else 0.0
    
    # Calculate segment speeds for single-segment dominance check
    speeds = []
    if len(t_pts) >= 2:
        lats = t_pts["cluster_lat_deg"].values
        lons = t_pts["cluster_lon_deg"].values
        times = pd.to_datetime(t_pts["scan_time_utc"])
        for i in range(len(lats) - 1):
            dt = (times.iloc[i+1] - times.iloc[i]).total_seconds()
            dist = haversine_distance_km(lats[i], lons[i], lats[i+1], lons[i+1])
            speeds.append(dist / (dt / 3600.0) if dt > 0 else 0.0)
            
    max_speed = max(speeds) if speeds else 0.0
    med_speed = np.median(speeds) if speeds else 0.0
    
    speed_dominates = len(speeds) >= 3 and max_speed > 5.0 * med_speed and max_speed > 30.0
    
    reject_reason = ""
    quality_label = ""
    
    if row["n_points"] < 4 or row["duration_min"] < 15.0:
        quality_label = "rejected_too_short"
        reject_reason = f"Points ({row['n_points']}) < 4 or duration ({row['duration_min']:.1f} min) < 15 min"
    elif row["median_abs_vertical_mismatch_m"] > 750.0 or row["max_abs_vertical_mismatch_m"] > 2000.0:
        quality_label = "rejected_altitude_mismatch"
        reject_reason = f"Alt mismatch (median {row['median_abs_vertical_mismatch_m']:.0f}m, max {row['max_abs_vertical_mismatch_m']:.0f}m) exceeds limits (750m, 2000m)"
    elif row["median_segment_speed_kmh"] < 5.0 or row["median_segment_speed_kmh"] > 100.0:
        quality_label = "rejected_speed_jump"
        reject_reason = f"Median speed ({row['median_segment_speed_kmh']:.1f} km/h) is outside [5, 100] km/h"
    elif row["max_segment_speed_kmh"] > 150.0 or speed_dominates:
        quality_label = "rejected_speed_jump"
        if row["max_segment_speed_kmh"] > 150.0:
            reject_reason = f"Max speed ({row['max_segment_speed_kmh']:.1f} km/h) exceeds 150 km/h"
        else:
            reject_reason = f"Single segment speed jump ({max_speed:.1f} km/h) dominates median ({med_speed:.1f} km/h)"
    elif corridor_fraction < 0.5:
        quality_label = "rejected_not_near_telemetry_corridor"
        reject_reason = f"Fraction of points in corridor ({corridor_fraction * 100:.1f}%) < 50%"
    elif row["path_smoothness_score"] < 0.4:
        quality_label = "rejected_spaghetti_tracklet"
        reject_reason = f"Smoothness ({row['path_smoothness_score']:.2f}) < 0.4"
    else:
        if row["median_abs_vertical_mismatch_m"] <= 150.0 and 10.0 <= row["median_segment_speed_kmh"] <= 80.0 and row["path_smoothness_score"] >= 0.7 and corridor_fraction >= 0.8:
            quality_label = "excellent_plausible_tracklet"
        elif row["median_abs_vertical_mismatch_m"] <= 400.0 and 5.0 <= row["median_segment_speed_kmh"] <= 90.0 and row["path_smoothness_score"] >= 0.6:
            quality_label = "good_plausible_tracklet"
        else:
            quality_label = "weak_plausible_tracklet"
            
    return {
        "tracklet_id": tid,
        "radar_site": row["radar_site"],
        "n_points": int(row["n_points"]),
        "duration_min": row["duration_min"],
        "median_abs_vertical_mismatch_m": row["median_abs_vertical_mismatch_m"],
        "max_abs_vertical_mismatch_m": row["max_abs_vertical_mismatch_m"],
        "median_segment_speed_kmh": row["median_segment_speed_kmh"],
        "max_segment_speed_kmh": row["max_segment_speed_kmh"],
        "path_smoothness_score": row["path_smoothness_score"],
        "corridor_fraction": corridor_fraction,
        "spaghetti_score": round(spag_score, 2),
        "quality_label": quality_label,
        "status": "plausible" if "plausible" in quality_label else "rejected",
        "reject_reason": reject_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter plausible, telemetry-consistent balloon-like tracklets.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    
    out_dir = case_dir / "outputs" / "discovery"
    
    geom_csv = case_dir / "nexrad" / "regional_radar_geometry.csv"
    if not geom_csv.exists():
        raise FileNotFoundError(f"Geometry report {geom_csv} not found. Run download first.")
        
    geom_df = pd.read_csv(geom_csv)
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
    
    raw_summaries = []
    raw_points = []
    
    for site in active_sites:
        t_csv = out_dir / site / "candidate_tracklets.csv"
        p_csv = out_dir / site / "tracklet_points.csv"
        
        if t_csv.exists() and p_csv.exists():
            tdf = pd.read_csv(t_csv)
            pdf = pd.read_csv(p_csv)
            if not tdf.empty:
                raw_summaries.append(tdf)
                raw_points.append(pdf)
                
    if not raw_summaries:
        print("No raw tracklets found to process.")
        pd.DataFrame().to_csv(out_dir / "plausible_tracklets.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "plausible_tracklet_points.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "plausible_cross_radar_associations.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tracklet_quality_diagnostics.csv", index=False)
        return
        
    all_summaries_df = pd.concat(raw_summaries, ignore_index=True)
    all_points_df = pd.concat(raw_points, ignore_index=True)
    
    diagnostics = []
    plausible_summaries = []
    
    for _, row in all_summaries_df.iterrows():
        tid = row["tracklet_id"]
        t_pts = all_points_df[all_points_df["tracklet_id"] == tid].copy()
        
        diag_row = evaluate_tracklet_quality(row, t_pts, all_points_df)
        diagnostics.append(diag_row)
        
        if diag_row["status"] == "plausible":
            row_copy = row.copy()
            row_copy["spaghetti_score"] = diag_row["spaghetti_score"]
            row_copy["quality_label"] = diag_row["quality_label"]
            plausible_summaries.append(row_copy)
            
    # Create diagnostics dataframe
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(out_dir / "tracklet_quality_diagnostics.csv", index=False)
    print(f"Wrote tracklet diagnostics: {out_dir / 'tracklet_quality_diagnostics.csv'}")
    
    if plausible_summaries:
        pl_sum_df = pd.DataFrame(plausible_summaries)
        pl_sum_df.to_csv(out_dir / "plausible_tracklets.csv", index=False)
        print(f"Wrote plausible tracklets: {out_dir / 'plausible_tracklets.csv'}")
        
        plausible_ids = pl_sum_df["tracklet_id"].tolist()
        pl_pts_df = all_points_df[all_points_df["tracklet_id"].isin(plausible_ids)]
        pl_pts_df.to_csv(out_dir / "plausible_tracklet_points.csv", index=False)
        print(f"Wrote plausible tracklet points: {out_dir / 'plausible_tracklet_points.csv'}")
    else:
        pd.DataFrame().to_csv(out_dir / "plausible_tracklets.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "plausible_tracklet_points.csv", index=False)
        print("No plausible tracklets found. Wrote empty tables.")
        
    # Process cross-radar associations
    assoc_csv = out_dir / "cross_radar_tracklet_associations.csv"
    pl_assoc_csv = out_dir / "plausible_cross_radar_associations.csv"
    
    if assoc_csv.exists():
        assoc_df = pd.read_csv(assoc_csv)
        if not assoc_df.empty and plausible_summaries:
            pl_ids = set(pl_sum_df["tracklet_id"])
            plausible_assocs = []
            for _, r in assoc_df.iterrows():
                tids = r["tracklet_ids"].split(";")
                # If all associated tracklets are in the plausible set
                if all(tid in pl_ids for tid in tids):
                    plausible_assocs.append(r)
            if plausible_assocs:
                pd.DataFrame(plausible_assocs).to_csv(pl_assoc_csv, index=False)
                print(f"Wrote {len(plausible_assocs)} plausible cross-radar associations: {pl_assoc_csv}")
            else:
                pd.DataFrame().to_csv(pl_assoc_csv, index=False)
                print("Wrote empty plausible cross-radar associations.")
        else:
            pd.DataFrame().to_csv(pl_assoc_csv, index=False)
            print("Wrote empty plausible cross-radar associations.")
    else:
        pd.DataFrame().to_csv(pl_assoc_csv, index=False)
        print("Wrote empty plausible cross-radar associations.")

    # Mirror all generated CSV tables to docs/discovery
    import shutil
    docs_discovery_dir = ROOT / "docs" / "discovery"
    docs_discovery_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in ["tracklet_quality_diagnostics.csv", "plausible_tracklets.csv", "plausible_tracklet_points.csv", "plausible_cross_radar_associations.csv"]:
        src = out_dir / filename
        if src.exists():
            shutil.copy(src, docs_discovery_dir / filename)
    print("Copied all plausible tables to docs/discovery/")


if __name__ == "__main__":
    main()
