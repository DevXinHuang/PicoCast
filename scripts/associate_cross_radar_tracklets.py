#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 5: Cross-radar candidate tracklet association."""

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
    bounded_score,
    horizontal_distance_km,
)

haversine_distance_km = horizontal_distance_km

ASSOCIATION_COLUMNS = [
    "association_id",
    "radar_sites",
    "tracklet_ids",
    "n_radars",
    "time_overlap_min",
    "n_overlap_samples",
    "median_altitude_difference_m",
    "median_horizontal_difference_km",
    "mean_telemetry_consistency_score",
    "cross_radar_score",
    "association_label",
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


def interpolate_tracklet_at_time(points_df: pd.DataFrame, target_time: datetime) -> dict | None:
    """Linearly interpolate latitude, longitude, and altitude from tracklet points at target_time."""
    # Convert timestamps to unix seconds
    times = points_df["time_dt"].tolist()
    if target_time < times[0] or target_time > times[-1]:
        return None
        
    times_sec = [t.timestamp() for t in times]
    target_sec = target_time.timestamp()
    
    lats = points_df["cluster_lat_deg"].to_numpy(dtype=float)
    lons = points_df["cluster_lon_deg"].to_numpy(dtype=float)
    alts = points_df["cluster_alt_m"].to_numpy(dtype=float)
    scores = points_df["balloon_like_cluster_score"].to_numpy(dtype=float)
    
    lat = float(np.interp(target_sec, times_sec, lats))
    lon = float(np.interp(target_sec, times_sec, lons))
    alt = float(np.interp(target_sec, times_sec, alts))
    score = float(np.interp(target_sec, times_sec, scores))
    
    return {"lat": lat, "lon": lon, "alt": alt, "score": score}


def evaluate_association(
    t1_summary: pd.Series, t1_points: pd.DataFrame,
    t2_summary: pd.Series, t2_points: pd.DataFrame
) -> dict | None:
    # Check time overlap
    t1_start = t1_points["time_dt"].min()
    t1_end = t1_points["time_dt"].max()
    t2_start = t2_points["time_dt"].min()
    t2_end = t2_points["time_dt"].max()
    
    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    
    overlap_sec = (overlap_end - overlap_start).total_seconds()
    if overlap_sec < 0:
        return None  # No time overlap
        
    overlap_min = overlap_sec / 60.0
    
    # Collect all actual scan times of both tracklets that fall in the overlap window
    all_times = sorted(list(set(
        [t for t in t1_points["time_dt"] if overlap_start <= t <= overlap_end] +
        [t for t in t2_points["time_dt"] if overlap_start <= t <= overlap_end]
    )))
    
    if not all_times:
        return None
        
    h_diffs = []
    v_diffs = []
    
    for t in all_times:
        p1 = interpolate_tracklet_at_time(t1_points, t)
        p2 = interpolate_tracklet_at_time(t2_points, t)
        if p1 and p2:
            h_dist = haversine_distance_km(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
            v_dist = abs(p1["alt"] - p2["alt"])
            h_diffs.append(h_dist)
            v_diffs.append(v_dist)
            
    if not h_diffs:
        return None
        
    med_h = float(np.median(h_diffs))
    med_v = float(np.median(v_diffs))
    
    # Mean telemetry match consistency
    mean_tel = 0.5 * (float(t1_summary["telemetry_match_score"]) + float(t2_summary["telemetry_match_score"]))
    
    # Composite cross-radar score
    h_score = bounded_score(med_h, 0.0, 50.0)
    v_score = bounded_score(med_v, 0.0, 2000.0)
    composite_score = 0.4 * h_score + 0.4 * v_score + 0.2 * mean_tel
    
    # Labeling
    t1_ok = t1_summary["tracklet_label"] in ["telemetry_consistent_candidate_tracklet", "altitude_consistent_candidate_tracklet"]
    t2_ok = t2_summary["tracklet_label"] in ["telemetry_consistent_candidate_tracklet", "altitude_consistent_candidate_tracklet"]
    
    if med_h <= 10.0 and med_v <= 500.0 and t1_ok and t2_ok:
        label = "strong_cross_radar_candidate"
    elif med_h <= 25.0 and med_v <= 1000.0 and (t1_ok or t2_ok):
        label = "moderate_cross_radar_candidate"
    elif med_h <= 50.0 and med_v <= 2000.0:
        label = "weak_cross_radar_candidate"
    else:
        label = "inconclusive"
        
    return {
        "radar_sites": f"{t1_summary['radar_site']};{t2_summary['radar_site']}",
        "tracklet_ids": f"{t1_summary['tracklet_id']};{t2_summary['tracklet_id']}",
        "n_radars": 2,
        "time_overlap_min": round(overlap_min, 1),
        "n_overlap_samples": len(h_diffs),
        "median_altitude_difference_m": round(med_v, 1),
        "median_horizontal_difference_km": round(med_h, 2),
        "mean_telemetry_consistency_score": round(mean_tel, 4),
        "cross_radar_score": round(composite_score, 4),
        "association_label": label,
        "notes": f"Associated: overlap={overlap_min:.1f}m, med_h={med_h:.1f}km, med_v={med_v:.0f}m",
    }


def main():
    parser = argparse.ArgumentParser(description="Associate regional candidate tracklets across radars.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Not used but allowed for interface consistency")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already computed associations")
    
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
    
    if args.all_sites:
        target_sites = primary + secondary
    else:
        target_sites = primary
        
    active_sites = [r for r in target_sites if r in included_radars]
    
    # Check if already processed
    out_csv = case_dir / "outputs" / "discovery" / "cross_radar_tracklet_associations.csv"
    if out_csv.exists() and not args.overwrite:
        print(f"Reloading existing associations: {out_csv}")
        return
        
    # Load all summaries and points
    summaries = []
    points = {}
    
    for site in active_sites:
        s_path = case_dir / "outputs" / "discovery" / site / "candidate_tracklets.csv"
        p_path = case_dir / "outputs" / "discovery" / site / "tracklet_points.csv"
        
        if s_path.exists() and p_path.exists():
            sdf = pd.read_csv(s_path)
            if not sdf.empty:
                summaries.append(sdf)
                pdf = pd.read_csv(p_path)
                pdf["time_dt"] = pdf["scan_time_utc"].apply(parse_utc)
                pdf = pdf.sort_values("time_dt")
                for tid, grp in pdf.groupby("tracklet_id"):
                    points[tid] = grp
                    
    if not summaries:
        print("No tracklets found for association.")
        pd.DataFrame(columns=ASSOCIATION_COLUMNS).to_csv(out_csv, index=False)
        return
        
    all_tracklets_df = pd.concat(summaries, ignore_index=True)
    
    associations = []
    
    # Pairwise comparison
    for i in range(len(all_tracklets_df)):
        for j in range(i + 1, len(all_tracklets_df)):
            t1 = all_tracklets_df.iloc[i]
            t2 = all_tracklets_df.iloc[j]
            
            if t1["radar_site"] == t2["radar_site"]:
                continue
                
            t1_pts = points.get(t1["tracklet_id"])
            t2_pts = points.get(t2["tracklet_id"])
            
            if t1_pts is None or t2_pts is None:
                continue
                
            assoc = evaluate_association(t1, t1_pts, t2, t2_pts)
            if assoc:
                associations.append(assoc)
                
    # Sort and filter associations
    assoc_df = pd.DataFrame(associations)
    if not assoc_df.empty:
        # Sort by cross_radar_score descending
        assoc_df = assoc_df.sort_values("cross_radar_score", ascending=False).reset_index(drop=True)
        # Add association_id
        assoc_df.insert(0, "association_id", [f"A{idx:03d}" for idx in range(1, len(assoc_df) + 1)])
        # filter to keep valid labels (strong, moderate, weak)
        assoc_df = assoc_df[assoc_df["association_label"] != "inconclusive"].copy()
    else:
        assoc_df = pd.DataFrame(columns=ASSOCIATION_COLUMNS)
        
    assoc_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(assoc_df)} cross-radar associations: {out_csv}")


if __name__ == "__main__":
    main()
