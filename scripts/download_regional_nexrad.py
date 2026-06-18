#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 0 & 1: Pre-flight radar visibility analysis and regional NEXRAD parallel downloader."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.config import Config

# Standard constants
EARTH_RADIUS_M = 6371000.0
REFRACTION_INDEX_CORRECTION = 4.0 / 3.0


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
    """Calculate the elevation angle of the radar beam to point at target altitude."""
    if distance_m <= 0:
        return 90.0
    # Refraction corrected earth radius
    re_corrected = REFRACTION_INDEX_CORRECTION * EARTH_RADIUS_M
    numerator = target_alt_m - radar_alt_m - (distance_m**2) / (2.0 * re_corrected)
    elevation_rad = np.arctan2(numerator, distance_m)
    return float(np.rad2deg(elevation_rad))


def get_expected_telemetry_points(expected_track_path: Path) -> list[dict]:
    if not expected_track_path.exists():
        return []
    df = pd.read_csv(expected_track_path)
    points = []
    for _, row in df.iterrows():
        points.append({
            "time_utc": parse_utc(row["time_utc"]),
            "lat": float(row["lat_deg"]),
            "lon": float(row["lon_deg"]),
            "alt_m": float(row["alt_m"]),
        })
    return sorted(points, key=lambda x: x["time_utc"])


def interpolate_balloon_telemetry(points: list[dict], target_time: datetime) -> dict | None:
    if not points:
        return None
    times = [p["time_utc"] for p in points]
    if target_time < times[0] or target_time > times[-1]:
        return None
    
    # Find surrounding points
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
    
    return {"lat": lat, "lon": lon, "alt_m": alt_m}


def parse_scan_time(filename: str, radar_site: str) -> datetime | None:
    pattern = re.compile(rf"{re.escape(radar_site)}(\d{{8}})_(\d{{6}})")
    match = pattern.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def get_radar_s3_files(s3, bucket: str, radar_site: str, date_str: str, start_time: datetime, end_time: datetime) -> list[dict]:
    # Prefix is YYYY/MM/DD/RADAR/
    prefix = f"{date_str.replace('-', '/')}/{radar_site}/"
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                filename = Path(key).name
                if "MDM" in filename.upper():  # ignore metadata
                    continue
                scan_time = parse_scan_time(filename, radar_site)
                if scan_time and start_time <= scan_time <= end_time:
                    files.append({
                        "key": key,
                        "filename": filename,
                        "scan_time": scan_time,
                        "size": int(item.get("Size", 0)),
                    })
    except Exception as e:
        print(f"Warning: S3 listing failed for {radar_site}: {e}")
    return sorted(files, key=lambda x: x["scan_time"])


def evaluate_radar_geometry(
    radar_site: str,
    radar_lat: float,
    radar_lon: float,
    radar_alt_m: float,
    telemetry_points: list[dict],
    s3_files: list[dict]
) -> dict:
    if not telemetry_points:
        return {
            "radar_site": radar_site,
            "min_range_km": np.nan,
            "max_range_km": np.nan,
            "n_visible_scans": 0,
            "n_total_scans": len(s3_files),
            "visibility_fraction": 0.0,
            "geometry_status": "skip",
            "skip_reason": "No telemetry points loaded for analysis",
        }
    
    n_visible = 0
    ranges = []
    
    for f in s3_files:
        scan_time = f["scan_time"]
        balloon = interpolate_balloon_telemetry(telemetry_points, scan_time)
        if not balloon:
            continue
            
        dist_km = haversine_distance_km(radar_lat, radar_lon, balloon["lat"], balloon["lon"])
        ranges.append(dist_km)
        
        # Check range <= 250 km
        if dist_km <= 250.0:
            elev_deg = calculate_beam_elevation_deg(radar_alt_m, balloon["alt_m"], dist_km * 1000.0)
            # Check elevation coverage [0.5, 19.5]
            if 0.5 <= elev_deg <= 19.5:
                n_visible += 1
                
    n_total = len(s3_files)
    vis_fraction = n_visible / n_total if n_total > 0 else 0.0
    
    geometry_status = "include" if n_visible > 0 else "skip"
    skip_reason = ""
    if n_total == 0:
        geometry_status = "skip"
        skip_reason = "No scans available in the S3 archive for this window"
    elif n_visible == 0:
        geometry_status = "skip"
        min_r = min(ranges) if ranges else 999.9
        if min_r > 250.0:
            skip_reason = f"Radar is too far (min distance {min_r:.1f} km > 250 km)"
        else:
            skip_reason = "Target altitude falls outside 0.5 to 19.5 deg beam elevation sweep"
            
    return {
        "radar_site": radar_site,
        "min_range_km": round(min(ranges), 2) if ranges else np.nan,
        "max_range_km": round(max(ranges), 2) if ranges else np.nan,
        "n_visible_scans": n_visible,
        "n_total_scans": n_total,
        "visibility_fraction": round(vis_fraction, 4),
        "geometry_status": geometry_status,
        "skip_reason": skip_reason,
    }


def download_single_file(s3, bucket: str, key: str, out_path: Path) -> tuple[str, bool]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(out_path))
        return key, True
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return key, False


def main():
    parser = argparse.ArgumentParser(description="Download and index regional NEXRAD Level II scans.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Download primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Download all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    parser.add_argument("--dry-run", action="store_true", help="Perform geometry analysis but do not download raw files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already downloaded files and indices")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    case_id = config["case_id"]
    
    # Resolve target radars
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
        # Default to primary sites
        target_sites = primary
        
    print(f"Targeting radar sites: {', '.join(target_sites)}")
    
    # Load telemetry
    expected_track_path = case_dir / "expected_track.csv"
    telemetry_points = get_expected_telemetry_points(expected_track_path)
    if not telemetry_points:
        print(f"Warning: Expected track file {expected_track_path} not found. Running build_case_from_csv first is recommended.")
        
    # S3 Client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = config["nexrad"]["bucket"]
    date_str = str(config["nexrad"]["date_utc"])
    
    time_window = discovery.get("time_window_utc", {})
    start_time = parse_utc(time_window.get("start", "2026-03-22T18:45:00Z"))
    end_time = parse_utc(time_window.get("end", "2026-03-22T23:00:00Z"))
    
    # Phase 0: Geometry and inventory dry run
    print("Evaluating radar geometries and listings...")
    radar_sites_cfg = config.get("radar_sites", {})
    
    geometry_rows = []
    radar_scans_inventory = {}
    
    for site in target_sites:
        if site not in radar_sites_cfg:
            print(f"Skipping {site}: coordinates not configured in config.yaml")
            continue
            
        site_cfg = radar_sites_cfg[site]
        lat, lon, alt_m = site_cfg["lat"], site_cfg["lon"], site_cfg["alt_m"]
        
        scans = get_radar_s3_files(s3, bucket, site, date_str, start_time, end_time)
        radar_scans_inventory[site] = scans
        
        geom = evaluate_radar_geometry(site, lat, lon, alt_m, telemetry_points, scans)
        geom["radar_lat"] = lat
        geom["radar_lon"] = lon
        geom["radar_alt_m"] = alt_m
        geometry_rows.append(geom)
        
    # Write geometry report
    geometry_csv_path = case_dir / "nexrad" / "regional_radar_geometry.csv"
    geometry_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort geom table: included first, then by min range
    geometry_df = pd.DataFrame(geometry_rows)
    # Reorder columns
    cols = [
        "radar_site", "radar_lat", "radar_lon", "radar_alt_m",
        "min_range_km", "max_range_km", "n_visible_scans", "n_total_scans",
        "visibility_fraction", "geometry_status", "skip_reason"
    ]
    geometry_df = geometry_df[cols]
    geometry_df.to_csv(geometry_csv_path, index=False)
    print(f"Wrote geometry report: {geometry_csv_path}")
    print(geometry_df.to_string(index=False))
    
    # Skip download for any skipped radars
    active_sites = [r["radar_site"] for r in geometry_rows if r["geometry_status"] == "include"]
    print(f"Active sites to download: {', '.join(active_sites)}")
    
    # Build inventory summary
    inventory_rows = []
    
    for geom in geometry_rows:
        site = geom["radar_site"]
        scans = radar_scans_inventory.get(site, [])
        
        # Check local files to count downloaded
        local_raw_dir = case_dir / "nexrad" / site / "raw"
        n_downloaded = 0
        if local_raw_dir.is_dir():
            for f in local_raw_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    n_downloaded += 1
                    
        total_size = sum(f["size"] for f in scans)
        time_min = scans[0]["scan_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if scans else ""
        time_max = scans[-1]["scan_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if scans else ""
        
        status = "complete" if n_downloaded >= len(scans) and len(scans) > 0 else "incomplete"
        if geom["geometry_status"] == "skip":
            status = "skipped"
            
        inventory_rows.append({
            "radar_site": site,
            "n_files_available": len(scans),
            "n_files_downloaded": n_downloaded,
            "time_min_utc": time_min,
            "time_max_utc": time_max,
            "total_size_bytes": total_size,
            "status": status,
            "notes": geom["skip_reason"] if geom["skip_reason"] else "Included in pipeline",
        })
        
    inventory_csv_path = case_dir / "nexrad" / "regional_nexrad_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_csv_path, index=False)
    print(f"Wrote inventory inventory: {inventory_csv_path}")
    
    if args.dry_run:
        print("Dry run completed. Skipping S3 download.")
        return
        
    # Phase 1: Download files
    max_workers = config.get("performance", {}).get("max_workers", 6)
    downloads_todo = []
    
    # We will build file indices per radar site
    for site in active_sites:
        scans = radar_scans_inventory[site]
        local_raw_dir = case_dir / "nexrad" / site / "raw"
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        
        radar_index_path = case_dir / "nexrad" / site / "index" / "nexrad_files.csv"
        radar_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if present and not overwriting
        existing_index = {}
        if radar_index_path.exists() and not args.overwrite:
            try:
                idf = pd.read_csv(radar_index_path)
                for _, row in idf.iterrows():
                    existing_index[row["filename"]] = dict(row)
            except Exception:
                pass
        
        for scan in scans:
            local_path = local_raw_dir / scan["filename"]
            exists = local_path.exists()
            
            # Check if we should download
            todo = False
            if not exists or args.overwrite:
                todo = True
            elif scan["filename"] in existing_index:
                # If it's already recorded as downloaded, don't download
                rec = existing_index[scan["filename"]]
                if not rec.get("downloaded", False):
                    todo = True
            else:
                todo = True
                
            if todo:
                downloads_todo.append((s3, bucket, scan["key"], local_path))
                
        # Fill placeholders for index rows (will update after download)
        
    if downloads_todo:
        print(f"Downloading {len(downloads_todo)} files in parallel (workers={max_workers})...")
        downloaded_keys = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_single_file, s3, bucket, key, path)
                for s3, bucket, key, path in downloads_todo
            ]
            for fut in as_completed(futures):
                key, success = fut.result()
                if success:
                    downloaded_keys.add(key)
        print("Downloads finished.")
        
    # Re-build file indices for active radars
    for site in active_sites:
        scans = radar_scans_inventory[site]
        local_raw_dir = case_dir / "nexrad" / site / "raw"
        radar_index_path = case_dir / "nexrad" / site / "index" / "nexrad_files.csv"
        
        index_rows = []
        for scan in scans:
            local_path = local_raw_dir / scan["filename"]
            exists = local_path.exists()
            size = local_path.stat().st_size if exists else 0
            
            index_rows.append({
                "case_id": case_id,
                "radar_site": site,
                "scan_time_utc": scan["scan_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "s3_bucket": bucket,
                "s3_key": scan["key"],
                "s3_uri": f"s3://{bucket}/{scan['key']}",
                "local_path": str(local_path.relative_to(case_dir.parent.parent)),
                "filename": scan["filename"],
                "downloaded": exists,
                "file_size_bytes": size,
                "download_error": "" if exists else "Missing or download failed",
            })
            
        pd.DataFrame(index_rows).to_csv(radar_index_path, index=False)
        print(f"Updated index for {site}: {radar_index_path}")
        
    # Re-write final inventory status
    inventory_rows_final = []
    for geom in geometry_rows:
        site = geom["radar_site"]
        scans = radar_scans_inventory.get(site, [])
        
        local_raw_dir = case_dir / "nexrad" / site / "raw"
        n_downloaded = 0
        if local_raw_dir.is_dir():
            for f in local_raw_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    n_downloaded += 1
                    
        total_size = sum(f["size"] for f in scans)
        time_min = scans[0]["scan_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if scans else ""
        time_max = scans[-1]["scan_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if scans else ""
        
        status = "complete" if n_downloaded >= len(scans) and len(scans) > 0 else "incomplete"
        if geom["geometry_status"] == "skip":
            status = "skipped"
            
        inventory_rows_final.append({
            "radar_site": site,
            "n_files_available": len(scans),
            "n_files_downloaded": n_downloaded,
            "time_min_utc": time_min,
            "time_max_utc": time_max,
            "total_size_bytes": total_size,
            "status": status,
            "notes": geom["skip_reason"] if geom["skip_reason"] else "Included in pipeline",
        })
        
    pd.DataFrame(inventory_rows_final).to_csv(inventory_csv_path, index=False)
    print(f"Updated final regional inventory inventory: {inventory_csv_path}")


if __name__ == "__main__":
    main()
