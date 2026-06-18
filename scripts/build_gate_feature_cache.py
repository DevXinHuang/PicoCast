#!/usr/bin/env python
# ruff: noqa: E501
"""Build gate-level feature cache for NEXRAD scans."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
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
    horizontal_distance_km,
    local_xy_km,
)

haversine_distance_km = horizontal_distance_km


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_track_polyline_segments(expected_track_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(expected_track_path)
    lats = df["lat_deg"].to_numpy(dtype=float)
    lons = df["lon_deg"].to_numpy(dtype=float)
    grids = df["maidenhead_grid"].tolist()
    return lats, lons, grids


def distance_to_polyline_km(
    gate_lats: np.ndarray,
    gate_lons: np.ndarray,
    track_lats: np.ndarray,
    track_lons: np.ndarray,
) -> np.ndarray:
    """Calculate the shortest distance from each gate to the track piecewise polyline in local km."""
    if len(track_lats) < 2:
        return np.asarray([haversine_distance_km(lat, lon, track_lats[0], track_lons[0])
                           for lat, lon in zip(gate_lats, gate_lons, strict=True)])
        
    origin_lat = track_lats[0]
    origin_lon = track_lons[0]
    
    track_x, track_y = local_xy_km(track_lats, track_lons, origin_lat, origin_lon)
    gate_x, gate_y = local_xy_km(gate_lats, gate_lons, origin_lat, origin_lon)
    
    min_dist2 = np.full(gate_x.shape, np.inf)
    
    for i in range(len(track_lats) - 1):
        ax, ay = track_x[i], track_y[i]
        bx, by = track_x[i+1], track_y[i+1]
        
        dx = bx - ax
        dy = by - ay
        len2 = dx**2 + dy**2
        if len2 <= 1e-6:
            d2 = (gate_x - ax)**2 + (gate_y - ay)**2
        else:
            t = ((gate_x - ax) * dx + (gate_y - ay) * dy) / len2
            t = np.clip(t, 0.0, 1.0)
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d2 = (gate_x - proj_x)**2 + (gate_y - proj_y)**2
            
        min_dist2 = np.minimum(min_dist2, d2)
        
    return np.sqrt(min_dist2)


def get_all_field_values(radar, field_name: str) -> np.ndarray:
    if field_name not in radar.fields:
        return np.full(radar.gate_latitude["data"].shape, np.nan, dtype=np.float32)
    values = radar.fields[field_name]["data"]
    if isinstance(values, ma.MaskedArray):
        return values.astype(np.float32).filled(np.nan)
    return np.asarray(values, dtype=np.float32)


def to_numpy(val, fill=np.nan) -> np.ndarray:
    if isinstance(val, ma.MaskedArray):
        return val.astype(np.float32).filled(fill)
    return np.asarray(val, dtype=np.float32)


def build_radar_gate_cache(
    site: str,
    config_path: Path,
    overwrite: bool,
) -> list[dict]:
    config = load_config(config_path)
    case_dir = config_path.parent
    case_id = config["case_id"]
    
    cache_dir = case_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    match_path = case_dir / "nexrad" / site / "index" / "scan_track_matches.csv"
    if not match_path.exists():
        print(f"Skipping {site}: no scan match index found.")
        return []
        
    matches = pd.read_csv(match_path)
    matches = matches[matches["is_geometrically_visible"]].copy()
    if matches.empty:
        print(f"No visible scans to process for {site}")
        return []
        
    track_path = case_dir / "expected_track.csv"
    track_lats, track_lons, track_grids = get_track_polyline_segments(track_path)
    
    import pyart
    
    records = []
    
    for _, match_row in matches.iterrows():
        scan_time_str = str(match_row["scan_time_utc"])
        filename = str(match_row["scan_filename"])
        local_path = case_dir / "nexrad" / site / "raw" / filename
        
        if not local_path.exists():
            print(f"Warning: file {local_path} not found locally.")
            continue
            
        stamp = scan_time_str.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
        radar_cache_dir = cache_dir / "gates" / site
        out_parquet_path = radar_cache_dir / f"{site}_{stamp}.parquet"
        
        # Calculate file size or check existence
        if out_parquet_path.exists() and not overwrite:
            # We already have this file, get metadata if possible
            try:
                cached_df = pd.read_parquet(out_parquet_path)
                n_cached = len(cached_df)
                records.append({
                    "case_id": case_id,
                    "radar_site": site,
                    "scan_time_utc": scan_time_str,
                    "scan_filename": filename,
                    "cache_parquet_path": str(out_parquet_path.relative_to(case_dir)),
                    "n_total_gates": -1,  # Unknown without reading raw, or fill from catalog
                    "n_cached_gates": n_cached,
                    "compression": config.get("gate_cache", {}).get("compression", "zstd"),
                    "precision": config.get("gate_cache", {}).get("float_precision", "float32"),
                    "file_size_bytes": out_parquet_path.stat().st_size,
                })
                continue
            except Exception:
                pass  # Fallback to recalculating if reading cached parquet fails
                
        try:
            radar = pyart.io.read_nexrad_archive(str(local_path))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        gate_lat = to_numpy(radar.gate_latitude["data"])
        gate_lon = to_numpy(radar.gate_longitude["data"])
        gate_alt = to_numpy(radar.gate_altitude["data"])
        
        nrays, ngates = gate_lat.shape
        n_total_gates = int(gate_lat.size)
        
        az_2d = to_numpy(np.tile(radar.azimuth["data"][:, np.newaxis], (1, ngates)))
        el_2d = to_numpy(np.tile(radar.elevation["data"][:, np.newaxis], (1, ngates)))
        rg_2d = to_numpy(np.tile(radar.range["data"][np.newaxis, :] / 1000.0, (nrays, 1)))
        
        expected_alt_m = float(match_row["expected_alt_m"])
        expected_lat = float(match_row["expected_lat_deg"])
        expected_lon = float(match_row["expected_lon_deg"])
        
        refl = get_all_field_values(radar, "reflectivity")
        valid_refl_mask = ~np.isnan(refl)
        
        if config.get("gate_cache", {}).get("cache_only_discovery_region", False):
            alt_diff = np.abs(gate_alt - expected_alt_m)
            alt_mask = alt_diff <= config.get("discovery", {}).get("altitude_padding_m", 1500)
            combined_mask = valid_refl_mask & alt_mask
            
            sub_lats = gate_lat[combined_mask]
            sub_lons = gate_lon[combined_mask]
            if len(sub_lats) > 0:
                sub_dists = distance_to_polyline_km(sub_lats, sub_lons, track_lats, track_lons)
                corridor_mask = sub_dists <= config.get("discovery", {}).get("horizontal_corridor_padding_km", 40)
                
                temp = np.zeros_like(combined_mask, dtype=bool)
                temp[combined_mask] = corridor_mask
                combined_mask = temp
            else:
                combined_mask = np.zeros_like(valid_refl_mask, dtype=bool)
                
            final_mask = combined_mask
        else:
            final_mask = valid_refl_mask
            
        f_lat = gate_lat[final_mask]
        f_lon = gate_lon[final_mask]
        f_alt = gate_alt[final_mask]
        
        f_refl = refl[final_mask]
        f_vel = get_all_field_values(radar, "velocity")[final_mask]
        f_sw = get_all_field_values(radar, "spectrum_width")[final_mask]
        f_rho = get_all_field_values(radar, "cross_correlation_ratio")[final_mask]
        
        f_az = az_2d[final_mask]
        f_el = el_2d[final_mask]
        f_rg = rg_2d[final_mask]
        
        n_cached_gates = int(len(f_lat))
        
        if n_cached_gates > 0:
            f_dist_corridor = distance_to_polyline_km(f_lat, f_lon, track_lats, track_lons)
            signed_vertical = f_alt - expected_alt_m
            abs_vertical = np.abs(signed_vertical)
            
            alt_pad = config.get("discovery", {}).get("altitude_padding_m", 1500)
            horiz_pad = config.get("discovery", {}).get("horizontal_corridor_padding_km", 40)
            
            inside_alt = abs_vertical <= alt_pad
            inside_horiz = f_dist_corridor <= horiz_pad
            inside_both = inside_alt & inside_horiz
            
            df = pd.DataFrame({
                "case_id": case_id,
                "radar_site": site,
                "scan_time_utc": scan_time_str,
                "scan_filename": filename,
                "azimuth_deg": f_az.astype(np.float32),
                "elevation_deg": f_el.astype(np.float32),
                "range_km": f_rg.astype(np.float32),
                "gate_lat_deg": f_lat.astype(np.float32),
                "gate_lon_deg": f_lon.astype(np.float32),
                "gate_alt_m": f_alt.astype(np.float32),
                "reflectivity_dbz": f_refl.astype(np.float32),
                "velocity_ms": f_vel.astype(np.float32),
                "spectrum_width_ms": f_sw.astype(np.float32),
                "rhohv": f_rho.astype(np.float32),
                "expected_alt_m": np.full(n_cached_gates, expected_alt_m, dtype=np.float32),
                "expected_lat_deg": np.full(n_cached_gates, expected_lat, dtype=np.float32),
                "expected_lon_deg": np.full(n_cached_gates, expected_lon, dtype=np.float32),
                "signed_vertical_m": signed_vertical.astype(np.float32),
                "abs_vertical_distance_m": abs_vertical.astype(np.float32),
                "distance_to_track_corridor_km": f_dist_corridor.astype(np.float32),
                "inside_altitude_corridor": inside_alt,
                "inside_horizontal_corridor": inside_horiz,
                "inside_both_corridors": inside_both,
                "available_fields": ",".join(sorted(radar.fields.keys())),
            })
            
            radar_cache_dir.mkdir(parents=True, exist_ok=True)
            compression = config.get("gate_cache", {}).get("compression", "zstd")
            df.to_parquet(out_parquet_path, index=False, compression=compression)
            
            records.append({
                "case_id": case_id,
                "radar_site": site,
                "scan_time_utc": scan_time_str,
                "scan_filename": filename,
                "cache_parquet_path": str(out_parquet_path.relative_to(case_dir)),
                "n_total_gates": n_total_gates,
                "n_cached_gates": n_cached_gates,
                "compression": compression,
                "precision": config.get("gate_cache", {}).get("float_precision", "float32"),
                "file_size_bytes": out_parquet_path.stat().st_size,
            })
            
    return records


def main():
    parser = argparse.ArgumentParser(description="Build gate-level feature cache.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already cached scans")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    
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
    print(f"Caching gates for active sites: {', '.join(active_sites)}")
    
    all_records = []
    
    for site in tqdm(active_sites, desc="Building cache per site"):
        records = build_radar_gate_cache(site, args.config, args.overwrite)
        all_records.extend(records)
        
    inventory_path = case_dir / "cache" / "gates" / "gate_cache_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    
    new_df = pd.DataFrame(all_records)
    
    if not new_df.empty:
        if inventory_path.exists() and not args.overwrite:
            old_df = pd.read_csv(inventory_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["scan_filename"], keep="last")
            combined.to_csv(inventory_path, index=False)
        else:
            new_df.to_csv(inventory_path, index=False)
        print(f"Updated gate cache inventory at {inventory_path} with {len(new_df)} new/updated records.")
    else:
        print("No new records added to the cache inventory.")


if __name__ == "__main__":
    main()
