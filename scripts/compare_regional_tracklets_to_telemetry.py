#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 6: Compare candidate tracklets against expected balloon telemetry."""

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

COMPARISON_COLUMNS = [
    "case_id",
    "radar_site",
    "tracklet_id",
    "n_points",
    "overlap_duration_min",
    "mean_distance_to_corridor_km",
    "max_distance_to_corridor_km",
    "median_altitude_difference_m",
    "max_altitude_difference_m",
    "speed_consistency_ratio",
    "continues_after_telemetry",
    "telemetry_match_label",
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


def main():
    parser = argparse.ArgumentParser(description="Compare regional candidate tracklets to balloon telemetry.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Not used but allowed for interface consistency")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already computed comparisons")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    case_id = config["case_id"]
    
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
    out_csv = case_dir / "outputs" / "discovery" / "regional_tracklet_telemetry_comparison.csv"
    if out_csv.exists() and not args.overwrite:
        print(f"Reloading existing telemetry comparison: {out_csv}")
        return
        
    # Load expected track
    track_path = case_dir / "expected_track.csv"
    if not track_path.exists():
        raise FileNotFoundError(f"Expected track {track_path} not found.")
    track_df = pd.read_csv(track_path)
    track_df["time_dt"] = track_df["time_utc"].apply(parse_utc)
    track_start = track_df["time_dt"].min()
    track_end = track_df["time_dt"].max()
    
    # Load summaries and points
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
                for tid, grp in pdf.groupby("tracklet_id"):
                    points[tid] = grp
                    
    if not summaries:
        print("No tracklets found for comparison.")
        pd.DataFrame(columns=COMPARISON_COLUMNS).to_csv(out_csv, index=False)
        return
        
    all_tracklets_df = pd.concat(summaries, ignore_index=True)
    comparison_rows = []
    
    for _, t in all_tracklets_df.iterrows():
        tid = t["tracklet_id"]
        t_pts = points.get(tid)
        if t_pts is None:
            continue
            
        t_start = t_pts["time_dt"].min()
        t_end = t_pts["time_dt"].max()
        
        # Calculate overlap in time
        overlap_start = max(t_start, track_start)
        overlap_end = min(t_end, track_end)
        overlap_sec = (overlap_end - overlap_start).total_seconds()
        overlap_min = max(0.0, overlap_sec / 60.0)
        
        # Calculate distance and altitude mismatch
        dist_to_corridors = t_pts["distance_to_track_corridor_km"].tolist()
        v_mismatches = t_pts["abs_vertical_distance_m"].tolist()
        
        mean_dist_corridor = float(np.mean(dist_to_corridors))
        max_dist_corridor = float(np.max(dist_to_corridors))
        med_alt_diff = float(np.median(v_mismatches))
        max_alt_diff = float(np.max(v_mismatches))
        
        # Check if tracklet continues after telemetry ends
        continues_after = t_end > track_end
        
        # Speed consistency
        # telemetry median speed is around 41 km/h
        t_speed = float(t["median_segment_speed_kmh"])
        # simple ratio (tracklet speed vs expected speed ~41 km/h)
        speed_ratio = round(t_speed / 41.0, 2)
        
        # Labeling logic
        if overlap_min <= 0:
            label = "inconclusive"
        elif med_alt_diff <= 500.0 and mean_dist_corridor <= 15.0:
            label = "telemetry_consistent_candidate"
        elif med_alt_diff <= 500.0 and mean_dist_corridor > 15.0:
            label = "altitude_consistent_but_position_uncertain"
        elif med_alt_diff > 500.0 and mean_dist_corridor <= 15.0:
            label = "position_consistent_but_altitude_weak"
        else:
            label = "not_consistent_with_telemetry"
            
        comparison_rows.append({
            "case_id": case_id,
            "radar_site": t["radar_site"],
            "tracklet_id": tid,
            "n_points": int(t["n_points"]),
            "overlap_duration_min": round(overlap_min, 1),
            "mean_distance_to_corridor_km": round(mean_dist_corridor, 2),
            "max_distance_to_corridor_km": round(max_dist_corridor, 2),
            "median_altitude_difference_m": round(med_alt_diff, 1),
            "max_altitude_difference_m": round(max_alt_diff, 1),
            "speed_consistency_ratio": speed_ratio,
            "continues_after_telemetry": continues_after,
            "telemetry_match_label": label,
            "notes": f"Match: med_v={med_alt_diff:.0f}m, mean_corridor={mean_dist_corridor:.1f}km",
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values("median_altitude_difference_m").reset_index(drop=True)
    comp_df.to_csv(out_csv, index=False)
    print(f"Wrote telemetry comparison report: {out_csv}")


if __name__ == "__main__":
    main()
