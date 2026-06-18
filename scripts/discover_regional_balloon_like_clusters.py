#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 3: Broad regional balloon-like candidate cluster discovery using DBSCAN."""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
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
    bounded_score,
    horizontal_distance_km,
    local_xy_km,
    reflectivity_score,
)

haversine_distance_km = horizontal_distance_km

# Columns for output cluster files
CLUSTER_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "cluster_id",
    "cluster_lat_deg",
    "cluster_lon_deg",
    "cluster_alt_m",
    "n_gates",
    "max_reflectivity_dbz",
    "mean_reflectivity_dbz",
    "p95_reflectivity_dbz",
    "velocity_mean_ms",
    "spectrum_width_mean_ms",
    "rhohv_mean",
    "compactness_km",
    "range_km",
    "azimuth_deg",
    "elevation_deg",
    "expected_alt_m",
    "signed_vertical_m",
    "abs_vertical_distance_m",
    "nearest_grid",
    "distance_to_nearest_grid_center_km",
    "distance_to_track_corridor_km",
    "inside_or_near_grid_corridor",
    "balloon_like_cluster_score",
    "notes",
]


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
        # Fallback to distance to single point
        return np.asarray([haversine_distance_km(lat, lon, track_lats[0], track_lons[0])
                           for lat, lon in zip(gate_lats, gate_lons, strict=True)])
        
    # Use first track point as local origin
    origin_lat = track_lats[0]
    origin_lon = track_lons[0]
    
    # Project track and gates to local km
    track_x, track_y = local_xy_km(track_lats, track_lons, origin_lat, origin_lon)
    gate_x, gate_y = local_xy_km(gate_lats, gate_lons, origin_lat, origin_lon)
    
    min_dist2 = np.full(gate_x.shape, np.inf)
    
    # Iterate over the segments of the polyline
    for i in range(len(track_lats) - 1):
        ax, ay = track_x[i], track_y[i]
        bx, by = track_x[i+1], track_y[i+1]
        
        dx = bx - ax
        dy = by - ay
        len2 = dx**2 + dy**2
        if len2 <= 1e-6:
            # Segment is a point
            d2 = (gate_x - ax)**2 + (gate_y - ay)**2
        else:
            # Project gate onto segment AB
            t = ((gate_x - ax) * dx + (gate_y - ay) * dy) / len2
            t = np.clip(t, 0.0, 1.0)
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d2 = (gate_x - proj_x)**2 + (gate_y - proj_y)**2
            
        min_dist2 = np.minimum(min_dist2, d2)
        
    return np.sqrt(min_dist2)


def sweep_numbers_by_ray(radar) -> np.ndarray:
    sweep_numbers = np.zeros(radar.nrays, dtype=int)
    starts = radar.sweep_start_ray_index["data"]
    ends = radar.sweep_end_ray_index["data"]
    for sweep_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
        sweep_numbers[int(start) : int(end) + 1] = sweep_number
    return sweep_numbers


def get_field_values(radar, field_name: str, ray_idx: np.ndarray, gate_idx: np.ndarray) -> np.ndarray:
    if field_name not in radar.fields:
        return np.full(ray_idx.shape, np.nan, dtype=float)
    values = radar.fields[field_name]["data"][ray_idx, gate_idx]
    if isinstance(values, ma.MaskedArray):
        return values.astype(float).filled(np.nan)
    return np.asarray(values, dtype=float)


def run_dbscan(coords_km: np.ndarray, eps_km: float) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps_km, min_samples=1).fit_predict(coords_km)
    except Exception:
        # Fallback to connected components
        labels = np.full(len(coords_km), -1, dtype=int)
        label = 0
        for idx in range(len(coords_km)):
            if labels[idx] >= 0:
                continue
            dists = np.linalg.norm(coords_km - coords_km[idx], axis=1)
            members = np.where(dists <= eps_km)[0]
            labels[members] = label
            label += 1
        return labels


def discover_radar_clusters(
    site: str,
    config_path: Path,
    overwrite: bool,
    use_gate_cache: bool = False,
) -> None:
    config = load_config(config_path)
    case_dir = config_path.parent
    case_id = config["case_id"]
    
    discovery = config.get("discovery", {})
    alt_padding = discovery.get("altitude_padding_m", 1500)
    horiz_padding = discovery.get("horizontal_corridor_padding_km", 40)
    max_gates_to_cluster = discovery.get("max_gates_per_scan_to_cluster", 200000)
    min_gates = discovery.get("min_gates_per_cluster", 1)
    max_gates = discovery.get("max_gates_per_cluster", 25)
    
    out_dir = case_dir / "outputs" / "discovery" / site
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_parquet = out_dir / "discovered_clusters.parquet"
    out_csv = out_dir / "discovered_clusters.csv"
    
    if out_parquet.exists() and not overwrite:
        print(f"Reloading existing candidate clusters for {site}")
        return
        
    match_path = case_dir / "nexrad" / site / "index" / "scan_track_matches.csv"
    if not match_path.exists():
        print(f"Skipping {site}: no scan match index found.")
        return
        
    matches = pd.read_csv(match_path)
    # Only process geometrically visible scans in the window
    matches = matches[matches["is_geometrically_visible"]].copy()
    if matches.empty:
        print(f"No visible scans to process for {site}")
        pd.DataFrame(columns=CLUSTER_COLUMNS).to_csv(out_csv, index=False)
        pd.DataFrame(columns=CLUSTER_COLUMNS).to_parquet(out_parquet, index=False)
        return
        
    # Load track polylines
    track_path = case_dir / "expected_track.csv"
    track_lats, track_lons, track_grids = get_track_polyline_segments(track_path)
    
    # Import Py-ART locally to be thread-safe
    import pyart
    
    all_clusters = []
    
    for _, match_row in tqdm(matches.iterrows(), total=len(matches), desc=f"Clustering {site}"):
        scan_time_str = str(match_row["scan_time_utc"])
        filename = str(match_row["scan_filename"])
        local_path = case_dir / "nexrad" / site / "raw" / filename
        
        expected_alt_m = float(match_row["expected_alt_m"])
        expected_lat = float(match_row["expected_lat_deg"])
        expected_lon = float(match_row["expected_lon_deg"])
        
        loaded_from_cache = False
        if use_gate_cache:
            stamp = scan_time_str.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
            parquet_path = case_dir / "cache" / "gates" / site / f"{site}_{stamp}.parquet"
            if parquet_path.exists():
                try:
                    cache_df = pd.read_parquet(parquet_path)
                    
                    # Apply thresholds
                    # 1. Broad Altitude filter
                    alt_mask = cache_df["abs_vertical_distance_m"] <= alt_padding
                    
                    # 2. Broad corridor filter
                    corridor_mask = cache_df["distance_to_track_corridor_km"] <= horiz_padding
                    
                    # 3. Reflectivity filter
                    refl_val = cache_df["reflectivity_dbz"].to_numpy()
                    refl_max = discovery.get("reflectivity_max_dbz", 20.0)
                    if refl_max is None:
                        refl_max = 20.0
                    valid_refl_mask = (~np.isnan(refl_val)) & (refl_val <= refl_max)
                    
                    combined_mask = alt_mask & corridor_mask & valid_refl_mask
                    filtered_df = cache_df[combined_mask].copy()
                    
                    if filtered_df.empty:
                        continue
                        
                    # Extract fields needed for clustering and metadata
                    lats = filtered_df["gate_lat_deg"].to_numpy()
                    lons = filtered_df["gate_lon_deg"].to_numpy()
                    alts = filtered_df["gate_alt_m"].to_numpy()
                    refl = filtered_df["reflectivity_dbz"].to_numpy()
                    vel = filtered_df["velocity_ms"].to_numpy()
                    sw = filtered_df["spectrum_width_ms"].to_numpy()
                    rhohv = filtered_df["rhohv"].to_numpy()
                    
                    azimuths = filtered_df["azimuth_deg"].to_numpy()
                    elevations = filtered_df["elevation_deg"].to_numpy()
                    ranges = filtered_df["range_km"].to_numpy()
                    
                    loaded_from_cache = True
                except Exception as e:
                    print(f"Warning: Failed to load gate cache from {parquet_path}: {e}. Falling back to raw file.")
                    
        if not loaded_from_cache:
            if not local_path.exists():
                print(f"Warning: file {local_path} not found locally.")
                continue
                
            try:
                radar = pyart.io.read_nexrad_archive(str(local_path))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
            gate_lat = radar.gate_latitude["data"]
            gate_lon = radar.gate_longitude["data"]
            gate_alt = radar.gate_altitude["data"]
            
            # 1. Broad Altitude filter (±1500m)
            alt_diff = np.abs(gate_alt - expected_alt_m)
            alt_mask = alt_diff <= alt_padding
            
            ray_idx, gate_idx = np.where(alt_mask)
            if len(ray_idx) == 0:
                continue
                
            # Extract lats/lons for the altitude-matched gates
            subset_lats = gate_lat[ray_idx, gate_idx]
            subset_lons = gate_lon[ray_idx, gate_idx]
            
            # 2. Broad corridor filter (shortest distance to piecewise track polyline <= 40 km)
            subset_track_dist = distance_to_polyline_km(subset_lats, subset_lons, track_lats, track_lons)
            corridor_mask = subset_track_dist <= horiz_padding
            
            ray_idx = ray_idx[corridor_mask]
            gate_idx = gate_idx[corridor_mask]
            if len(ray_idx) == 0:
                continue
                
            # 3. Reflectivity filter (<= reflectivity_max_dbz, and not NaN)
            refl_max = discovery.get("reflectivity_max_dbz", 20.0)
            if refl_max is None:
                refl_max = 20.0
            refl = get_field_values(radar, "reflectivity", ray_idx, gate_idx)
            valid_refl_mask = (~np.isnan(refl)) & (refl <= refl_max)
            
            ray_idx = ray_idx[valid_refl_mask]
            gate_idx = gate_idx[valid_refl_mask]
            if len(ray_idx) == 0:
                continue
                
            # Extract metadata
            lats = gate_lat[ray_idx, gate_idx]
            lons = gate_lon[ray_idx, gate_idx]
            alts = gate_alt[ray_idx, gate_idx]
            refl = refl[valid_refl_mask]
            
            vel = get_field_values(radar, "velocity", ray_idx, gate_idx)
            sw = get_field_values(radar, "spectrum_width", ray_idx, gate_idx)
            rhohv = get_field_values(radar, "cross_correlation_ratio", ray_idx, gate_idx)
            
            # Extract polar coordinates
            azimuths = radar.azimuth["data"][ray_idx]
            elevations = radar.elevation["data"][ray_idx]
            ranges = radar.range["data"][gate_idx] / 1000.0
        
        # Guard against too many gates to prevent memory blowup during DBSCAN
        if len(lats) > max_gates_to_cluster:
            print(f"Warning: too many candidate gates ({len(lats)}) in {filename}, skipping clustering.")
            continue
            
        # 4. DBSCAN Clustering in local X/Y/Z km relative to expected position
        origin_lat = expected_lat
        origin_lon = expected_lon
        gx, gy = local_xy_km(lats, lons, origin_lat, origin_lon)
        gz = (alts - expected_alt_m) / 1000.0
        coords_km = np.column_stack([gx, gy, gz])
        
        labels = run_dbscan(coords_km, eps_km=1.0)
        unique_labels = sorted(set(labels))
        
        stamp = scan_time_str.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
        
        for label in unique_labels:
            if label < 0:
                continue
            label_mask = labels == label
            n_pts = int(np.sum(label_mask))
            
            # Check candidate cluster size limits
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
            
            # Center values
            center_lat = float(np.mean(c_lats))
            center_lon = float(np.mean(c_lons))
            center_alt = float(np.mean(c_alts))
            
            cx, cy = local_xy_km(c_lats, c_lons, center_lat, center_lon)
            cz = (c_alts - center_alt) / 1000.0
            compactness = float(np.sqrt(cx**2 + cy**2 + cz**2).max()) if len(cx) > 0 else 0.0
            
            # Compute distance metrics
            dist_to_grid_center = float(haversine_distance_km(center_lat, center_lon, expected_lat, expected_lon))
            dist_to_corridor = float(distance_to_polyline_km(np.array([center_lat]), np.array([center_lon]), track_lats, track_lons)[0])
            
            # Find nearest grid label
            dists_to_track_points = [haversine_distance_km(center_lat, center_lon, p_lat, p_lon)
                                     for p_lat, p_lon in zip(track_lats, track_lons, strict=True)]
            nearest_idx = int(np.argmin(dists_to_track_points))
            nearest_grid = track_grids[nearest_idx]
            
            # Visibility flags
            inside_corridor = dist_to_corridor <= horiz_padding
            
            # Scoring
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
            
            all_clusters.append({
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
            
    df = pd.DataFrame(all_clusters, columns=CLUSTER_COLUMNS)
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(df)} candidate clusters for {site}: {out_parquet}")


def main():
    parser = argparse.ArgumentParser(description="Discover regional balloon-like candidate clusters.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already computed clusters")
    parser.add_argument("--use-gate-cache", action="store_true", help="Load cached gates instead of reading raw scans")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    
    # Read geometry report to know what is active
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
    print(f"Discovering clusters for active sites: {', '.join(active_sites)}")
    
    # Parallel processing of radars (Phase 3)
    max_workers = config.get("performance", {}).get("max_workers", 6)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(discover_radar_clusters, site, args.config, args.overwrite, args.use_gate_cache): site
            for site in active_sites
        }
        for future in futures:
            site = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception while discovering clusters for {site}: {e}")
                
    # Phase 3: Regional merged Parquet file
    merged_parquet_path = case_dir / "outputs" / "discovery" / "regional_discovered_clusters.parquet"
    merged_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for site in active_sites:
        pq = case_dir / "outputs" / "discovery" / site / "discovered_clusters.parquet"
        if pq.exists():
            dfs.append(pd.read_parquet(pq))
            
    if dfs:
        pd.concat(dfs, ignore_index=True).to_parquet(merged_parquet_path, index=False)
        print(f"Wrote merged regional clusters: {merged_parquet_path}")
    else:
        print("Warning: No candidate clusters found across any radar sites.")


if __name__ == "__main__":
    main()
