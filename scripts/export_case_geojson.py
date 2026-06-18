#!/usr/bin/env python
"""Export GeoJSON files for PicoCAST geospatial mapping."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexrad_picoballoon.maidenhead_grid import (  # noqa: E402
    grid_polygon_coords,
    grid_precision_chars,
    grid_uncertainty_km,
)
from nexrad_picoballoon.radar_site import resolve_radar_location  # noqa: E402

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Read a PicoCAST case YAML config."""

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def radar_site_from_config(config: dict, radar_site_arg: str | None) -> str:
    """Resolve the requested radar site from CLI arg or config."""

    if radar_site_arg:
        return radar_site_arg
    return config.get(
        "primary_radar_site",
        config.get("nexrad", {}).get("primary_radar_site"),
    )


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def _feature(geometry: dict, properties: dict) -> dict:
    """Build a single GeoJSON Feature dict."""

    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _feature_collection(features: list[dict]) -> dict:
    """Build a GeoJSON FeatureCollection dict."""

    return {"type": "FeatureCollection", "features": features}


def _point(lon: float, lat: float) -> dict:
    return {"type": "Point", "coordinates": [lon, lat]}


def _polygon(rings: list[list[list[float]]]) -> dict:
    return {"type": "Polygon", "coordinates": rings}


def _linestring(coords: list[list[float]]) -> dict:
    return {"type": "LineString", "coordinates": coords}


def _write_geojson(path: Path, data: dict) -> None:
    """Write a GeoJSON dict to *path*, creating parent dirs if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, allow_nan=False, default=str)
    print(f"Wrote {path}")


def _safe_float(value, default=None) -> float | None:
    """Convert a value to float, returning *default* for NaN / None."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Range ring geometry
# ---------------------------------------------------------------------------

def make_range_ring(
    lat: float, lon: float, radius_km: float, n_points: int = 72,
) -> list[list[float]]:
    """Return a closed ring of [lon, lat] coords at *radius_km* from center."""

    coords: list[list[float]] = []
    for i in range(n_points + 1):
        bearing = 360.0 * i / n_points
        dest = geodesic(kilometers=radius_km).destination((lat, lon), bearing)
        coords.append([dest.longitude, dest.latitude])
    return coords


# ---------------------------------------------------------------------------
# Individual layer builders
# ---------------------------------------------------------------------------

def build_expected_track(track_df: pd.DataFrame) -> dict:
    """Build expected_track.geojson FeatureCollection."""

    features: list[dict] = []
    for _, row in track_df.iterrows():
        props = {
            "case_id": row["case_id"],
            "time_utc": row["time_utc"],
            "time_local": row["time_local"],
            "maidenhead_grid": row["maidenhead_grid"],
            "grid_precision_chars": grid_precision_chars(str(row["maidenhead_grid"])),
            "alt_m": _safe_float(row.get("alt_m")),
            "alt_km": _safe_float(row.get("alt_km")),
            "position_source": "maidenhead_grid_center",
        }
        features.append(
            _feature(_point(float(row["lon_deg"]), float(row["lat_deg"])), props)
        )
    return _feature_collection(features)


def build_grid_squares(track_df: pd.DataFrame) -> dict:
    """Build grid_squares.geojson FeatureCollection."""

    features: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for _, row in track_df.iterrows():
        grid = str(row["maidenhead_grid"])
        time_utc = row["time_utc"]
        key = (grid, str(time_utc))
        if key in seen:
            continue
        seen.add(key)
        ring = grid_polygon_coords(grid)
        props = {
            "case_id": row["case_id"],
            "time_utc": time_utc,
            "maidenhead_grid": grid,
            "grid_precision_chars": grid_precision_chars(grid),
            "grid_uncertainty_km": round(grid_uncertainty_km(grid), 3),
            "position_source": "maidenhead_grid_bounds",
        }
        features.append(_feature(_polygon([ring]), props))
    return _feature_collection(features)


def build_top_candidates(candidates_df: pd.DataFrame, radar_site: str) -> dict:
    """Build top_candidates.geojson FeatureCollection."""

    features: list[dict] = []
    for _, row in candidates_df.iterrows():
        props = {
            "candidate_rank": int(row["candidate_rank"]),
            "scan_time_utc": row["scan_time_utc"],
            "candidate_score": _safe_float(row["candidate_score"]),
            "candidate_label": row["candidate_label"],
            "search_window": row["search_window"],
            "horizontal_distance_km": _safe_float(row["horizontal_distance_km"]),
            "vertical_distance_m": _safe_float(row["vertical_distance_m"]),
            "max_reflectivity_dbz": _safe_float(row["max_reflectivity_dbz"]),
            "n_gates": int(row["n_gates"]),
            "radar_site": radar_site,
        }
        features.append(
            _feature(
                _point(float(row["candidate_lon_deg"]), float(row["candidate_lat_deg"])),
                props,
            )
        )
    return _feature_collection(features)


def build_candidate_clusters(clusters_df: pd.DataFrame, radar_site: str) -> dict:
    """Build candidate_clusters.geojson FeatureCollection."""

    features: list[dict] = []
    for _, row in clusters_df.iterrows():
        props = {
            "cluster_id": str(row["cluster_id"]),
            "scan_time_utc": row["scan_time_utc"],
            "n_gates": int(row["n_gates"]),
            "cluster_max_reflectivity_dbz": _safe_float(row["cluster_max_reflectivity_dbz"]),
            "compactness_km": _safe_float(row["compactness_km"]),
            "radar_site": radar_site,
        }
        features.append(
            _feature(
                _point(
                    float(row["cluster_center_lon_deg"]),
                    float(row["cluster_center_lat_deg"]),
                ),
                props,
            )
        )
    return _feature_collection(features)


def build_radar_site(site_info: dict) -> dict:
    """Build <site>_radar_site.geojson single Feature."""

    return _feature(
        _point(site_info["lon"], site_info["lat"]),
        {
            "site_id": site_info["site"],
            "lat": site_info["lat"],
            "lon": site_info["lon"],
            "alt_m": site_info["alt_m"],
        },
    )


def build_range_rings(
    site_info: dict, range_rings_km: list[int | float],
) -> dict:
    """Build <site>_range_rings.geojson FeatureCollection."""

    features: list[dict] = []
    for radius_km in range_rings_km:
        coords = make_range_ring(site_info["lat"], site_info["lon"], radius_km)
        props = {
            "site_id": site_info["site"],
            "range_km": radius_km,
        }
        features.append(_feature(_linestring(coords), props))
    return _feature_collection(features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_case_geojson(
    config_path: Path, radar_site_arg: str | None = None,
) -> None:
    """Export all GeoJSON layers for a PicoCAST case."""

    config = load_config(config_path)
    radar_site = radar_site_from_config(config, radar_site_arg)
    case_dir = config_path.parent
    maps_dir = case_dir / "outputs" / "maps"

    # --- Expected track ---
    track_csv = case_dir / "expected_track.csv"
    track_df = pd.read_csv(track_csv)
    _write_geojson(maps_dir / "expected_track.geojson", build_expected_track(track_df))

    # --- Grid squares ---
    _write_geojson(maps_dir / "grid_squares.geojson", build_grid_squares(track_df))

    # --- Top candidates ---
    top_csv = case_dir / "outputs" / "candidates" / radar_site / "top_candidates.csv"
    if top_csv.exists():
        candidates_df = pd.read_csv(top_csv)
        _write_geojson(
            maps_dir / "top_candidates.geojson",
            build_top_candidates(candidates_df, radar_site),
        )
    else:
        print(f"Skipped top_candidates.geojson — {top_csv} not found")

    # --- Candidate clusters ---
    clusters_csv = case_dir / "outputs" / "candidates" / radar_site / "gate_clusters.csv"
    if clusters_csv.exists():
        clusters_df = pd.read_csv(clusters_csv)
        _write_geojson(
            maps_dir / "candidate_clusters.geojson",
            build_candidate_clusters(clusters_df, radar_site),
        )
    else:
        print(f"Skipped candidate_clusters.geojson — {clusters_csv} not found")

    # --- Radar site point ---
    site_id_lower = radar_site.lower()
    site_info = resolve_radar_location(config_path, radar_site)
    _write_geojson(
        maps_dir / f"{site_id_lower}_radar_site.geojson",
        build_radar_site(site_info),
    )

    # --- Range rings ---
    mapping_cfg = config.get("mapping", {})
    range_rings_km: list[int | float] = mapping_cfg.get(
        "range_rings_km", [50, 100, 150, 200],
    )
    _write_geojson(
        maps_dir / f"{site_id_lower}_range_rings.geojson",
        build_range_rings(site_info, range_rings_km),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument(
        "--radar-site", help="Radar site to process, default from config",
    )
    args = parser.parse_args()
    export_case_geojson(args.config, radar_site_arg=args.radar_site)


if __name__ == "__main__":
    main()
