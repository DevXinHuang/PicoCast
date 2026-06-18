#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 2: Regional scan-to-telemetry matching with Earth-curvature visibility checks."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EARTH_RADIUS_M = 6371000.0
REFRACTION_INDEX_CORRECTION = 4.0 / 3.0

REGIONAL_MATCH_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "scan_local_path",
    "expected_lat_deg",
    "expected_lon_deg",
    "expected_alt_km",
    "expected_alt_m",
    "expected_maidenhead_grid",
    "distance_to_expected_km",
    "expected_elevation_deg",
    "is_geometrically_visible",
    "time_offset_min",
    "inside_track_window",
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def haversine_distance_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return float(2.0 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(a)))


def calculate_beam_elevation_deg(radar_alt_m: float, target_alt_m: float, distance_m: float) -> float:
    if distance_m <= 0:
        return 90.0
    re_corrected = REFRACTION_INDEX_CORRECTION * EARTH_RADIUS_M
    numerator = target_alt_m - radar_alt_m - (distance_m**2) / (2.0 * re_corrected)
    elevation_rad = np.arctan2(numerator, distance_m)
    return float(np.rad2deg(elevation_rad))


def get_expected_telemetry_points(expected_track_path: Path) -> list[dict]:
    df = pd.read_csv(expected_track_path)
    points = []
    for _, row in df.iterrows():
        points.append({
            "time_utc": parse_utc(row["time_utc"]),
            "lat": float(row["lat_deg"]),
            "lon": float(row["lon_deg"]),
            "alt_m": float(row["alt_m"]),
            "maidenhead_grid": str(row["maidenhead_grid"]),
        })
    return sorted(points, key=lambda x: x["time_utc"])


def interpolate_balloon_telemetry(points: list[dict], target_time: datetime) -> dict | None:
    if not points:
        return None
    times = [p["time_utc"] for p in points]
    if target_time < times[0] or target_time > times[-1]:
        return None
    
    idx = 0
    while idx < len(times) - 1 and times[idx + 1] < target_time:
        idx += 1
        
    p0, p1 = points[idx], points[idx + 1]
    t0, t1 = p0["time_utc"], p1["time_utc"]
    dt_total = (t1 - t0).total_seconds()
    if dt_total <= 0:
        return p0
        
    fraction = (target_time - t0).total_seconds() / dt_total
    
    lat = p0["lat"] + fraction * (p1["lat"] - p0["lat"])
    lon = p0["lon"] + fraction * (p1["lon"] - p0["lon"])
    alt_m = p0["alt_m"] + fraction * (p1["alt_m"] - p0["alt_m"])
    
    # Grid: just take the nearest point's grid for simplicity
    nearest_p = p0 if fraction < 0.5 else p1
    grid = nearest_p["maidenhead_grid"]
    
    return {"lat": lat, "lon": lon, "alt_m": alt_m, "maidenhead_grid": grid}


def main():
    parser = argparse.ArgumentParser(description="Match regional NEXRAD scans to expected balloon telemetry.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already computed match indices")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    case_id = config["case_id"]
    
    # Read geometry report to know what to process
    geometry_csv_path = case_dir / "nexrad" / "regional_radar_geometry.csv"
    if not geometry_csv_path.exists():
        raise FileNotFoundError(
            f"Geometry report {geometry_csv_path} not found. "
            "Please run download_regional_nexrad.py with --dry-run or normal download first."
        )
        
    geom_df = pd.read_csv(geometry_csv_path)
    included_radars = geom_df[geom_df["geometry_status"] == "include"]["radar_site"].tolist()
    
    # Resolve target sites based on CLI args
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
        
    # Filter to only target radars that are actually included by geometry
    active_sites = [r for r in target_sites if r in included_radars]
    print(f"Matching telemetry for active sites: {', '.join(active_sites)}")
    
    # Load expected track
    expected_track_path = case_dir / "expected_track.csv"
    telemetry_points = get_expected_telemetry_points(expected_track_path)
    track_start = telemetry_points[0]["time_utc"]
    track_end = telemetry_points[-1]["time_utc"]
    
    radar_sites_cfg = config.get("radar_sites", {})
    
    regional_dfs = []
    
    for site in active_sites:
        site_match_path = case_dir / "nexrad" / site / "index" / "scan_track_matches.csv"
        
        # Check if already processed and not overwriting
        if site_match_path.exists() and not args.overwrite:
            print(f"Reloading existing match index for {site}: {site_match_path}")
            df = pd.read_csv(site_match_path)
            regional_dfs.append(df)
            continue
            
        index_path = case_dir / "nexrad" / site / "index" / "nexrad_files.csv"
        if not index_path.exists():
            print(f"Skipping {site}: S3 downloader has not indexed this site (run download_regional_nexrad.py first).")
            continue
            
        scans_df = pd.read_csv(index_path)
        # Filter download errors if any
        scans_df = scans_df[scans_df["downloaded"]].copy()
        if scans_df.empty:
            print(f"Warning: No downloaded scans found for {site}.")
            continue
            
        site_cfg = radar_sites_cfg[site]
        radar_lat, radar_lon, radar_alt_m = site_cfg["lat"], site_cfg["lon"], site_cfg["alt_m"]
        
        match_rows = []
        for _, row in scans_df.iterrows():
            scan_time = parse_utc(row["scan_time_utc"])
            balloon = interpolate_balloon_telemetry(telemetry_points, scan_time)
            
            # Determine if inside track window
            inside_track_window = track_start <= scan_time <= track_end
            
            # Calculate values
            if balloon:
                exp_lat, exp_lon, exp_alt_m = balloon["lat"], balloon["lon"], balloon["alt_m"]
                exp_grid = balloon["maidenhead_grid"]
                dist_km = haversine_distance_km(radar_lat, radar_lon, exp_lat, exp_lon)
                elev_deg = calculate_beam_elevation_deg(radar_alt_m, exp_alt_m, dist_km * 1000.0)
                is_visible = (dist_km <= 250.0) and (0.5 <= elev_deg <= 19.5)
            else:
                # Outside track window entirely
                # Find nearest telemetry point to compute range/alt/grid offset
                diffs = [abs((p["time_utc"] - scan_time).total_seconds()) for p in telemetry_points]
                nearest_idx = int(np.argmin(diffs))
                p_nearest = telemetry_points[nearest_idx]
                exp_lat, exp_lon, exp_alt_m = p_nearest["lat"], p_nearest["lon"], p_nearest["alt_m"]
                exp_grid = p_nearest["maidenhead_grid"]
                dist_km = haversine_distance_km(radar_lat, radar_lon, exp_lat, exp_lon)
                elev_deg = calculate_beam_elevation_deg(radar_alt_m, exp_alt_m, dist_km * 1000.0)
                is_visible = (dist_km <= 250.0) and (0.5 <= elev_deg <= 19.5)
                
            # Time offset to nearest telemetry point
            time_diffs_min = [(scan_time - p["time_utc"]).total_seconds() / 60.0 for p in telemetry_points]
            nearest_time_offset_min = time_diffs_min[int(np.argmin(np.abs(time_diffs_min)))]
            
            match_rows.append({
                "case_id": case_id,
                "radar_site": site,
                "scan_time_utc": row["scan_time_utc"],
                "scan_filename": row["filename"],
                "scan_local_path": row["local_path"],
                "expected_lat_deg": round(exp_lat, 6),
                "expected_lon_deg": round(exp_lon, 6),
                "expected_alt_km": round(exp_alt_m / 1000.0, 4),
                "expected_alt_m": round(exp_alt_m, 1),
                "expected_maidenhead_grid": exp_grid,
                "distance_to_expected_km": round(dist_km, 2),
                "expected_elevation_deg": round(elev_deg, 2),
                "is_geometrically_visible": is_visible,
                "time_offset_min": round(nearest_time_offset_min, 4),
                "inside_track_window": inside_track_window,
            })
            
        site_df = pd.DataFrame(match_rows)
        site_df = site_df[REGIONAL_MATCH_COLUMNS]
        site_df.to_csv(site_match_path, index=False)
        print(f"Wrote scan matches for {site}: {site_match_path}")
        regional_dfs.append(site_df)
        
    if regional_dfs:
        global_matches_path = case_dir / "nexrad" / "regional_scan_matches.csv"
        pd.concat(regional_dfs, ignore_index=True).to_csv(global_matches_path, index=False)
        print(f"Wrote merged regional scan matches: {global_matches_path}")
    else:
        print("Warning: No scan matches generated.")


if __name__ == "__main__":
    main()
