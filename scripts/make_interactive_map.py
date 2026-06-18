#!/usr/bin/env python
"""Create an interactive HTML map from pre-exported GeoJSON and candidate data."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import folium
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    case_id_from_config,
    load_config,
    radar_site_from_config,
)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def altitude_color(alt_m: float, min_alt: float, max_alt: float) -> str:
    """Map altitude to a green→yellow→red gradient."""
    if max_alt == min_alt:
        return "#3388ff"
    t = (alt_m - min_alt) / (max_alt - min_alt)
    hue = 0.33 * (1 - t)  # 0.33 = green, 0 = red
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def time_index_color(idx: int, total: int) -> str:
    """Map a 0-based time index to a blue→red gradient."""
    if total <= 1:
        return "#3388ff"
    t = idx / (total - 1)
    hue = 0.66 * (1 - t)  # 0.66 = blue, 0 = red
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def reflectivity_color(dbz: float) -> str:
    """Map reflectivity dBZ to a blue→green→yellow→red gradient."""
    low, high = -10.0, 50.0
    t = max(0.0, min(1.0, (dbz - low) / (high - low)))
    hue = 0.66 * (1 - t)  # 0.66 = blue, 0 = red
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.9)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------------------
# GeoJSON loaders
# ---------------------------------------------------------------------------

def load_geojson(path: Path) -> dict:
    """Read a GeoJSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def add_expected_track_points(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add expected balloon track points colored by altitude."""
    fg = folium.FeatureGroup(name="Expected Track Points", show=True)
    features = geojson.get("features", [])
    alts = [
        f["properties"].get("alt_m", 0) for f in features
        if f["geometry"]["type"] == "Point"
    ]
    min_alt = min(alts) if alts else 0
    max_alt = max(alts) if alts else 0

    for feat in features:
        if feat["geometry"]["type"] != "Point":
            continue
        coords = feat["geometry"]["coordinates"]  # [lon, lat]
        props = feat["properties"]
        alt = props.get("alt_m", 0)
        alt_km = props.get("alt_km", alt / 1000.0)
        color = altitude_color(alt, min_alt, max_alt)
        popup_html = (
            f"<b>Track Point</b><br>"
            f"Time: {props.get('time_utc', 'N/A')}<br>"
            f"Grid: {props.get('maidenhead_grid', 'N/A')}<br>"
            f"Alt: {alt:.0f} m ({alt_km:.1f} km)<br>"
            f"Source: {props.get('position_source', 'N/A')}<br>"
            f"<i style='color:#888;font-size:10px;'>"
            f"⚠️ Position estimated from Maidenhead grid-square center</i>"
        )
        folium.CircleMarker(
            location=[coords[1], coords[0]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_expected_track_line(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Connect expected track points with a dashed blue line."""
    fg = folium.FeatureGroup(name="Expected Track Line", show=True)
    features = geojson.get("features", [])
    points = []
    for feat in features:
        if feat["geometry"]["type"] != "Point":
            continue
        coords = feat["geometry"]["coordinates"]
        time_utc = feat["properties"].get("time_utc", "")
        points.append((time_utc, [coords[1], coords[0]]))

    # Sort by time
    points.sort(key=lambda x: x[0])
    if len(points) >= 2:
        folium.PolyLine(
            locations=[p[1] for p in points],
            color="blue",
            weight=2,
            dash_array="5 5",
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_grid_squares(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add Maidenhead grid-square polygons colored by time."""
    fg = folium.FeatureGroup(name="Grid Squares", show=True)
    features = [
        f for f in geojson.get("features", [])
        if f["geometry"]["type"] == "Polygon"
    ]
    total = len(features)

    for idx, feat in enumerate(features):
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        color = time_index_color(idx, total)
        # GeoJSON polygon rings are [[lon, lat], …] → flip to [lat, lon]
        locations = [[pt[1], pt[0]] for pt in coords[0]]
        popup_html = (
            f"<b>Grid Square</b><br>"
            f"Grid: {props.get('maidenhead_grid', 'N/A')}<br>"
            f"Time: {props.get('time_utc', 'N/A')}<br>"
            f"Precision: {props.get('grid_precision_chars', 'N/A')} chars<br>"
            f"Uncertainty: {props.get('grid_uncertainty_km', 'N/A')} km<br>"
            f"<i style='color:#888;font-size:10px;'>"
            f"⚠️ Position estimated from Maidenhead grid-square center</i>"
        )
        folium.Polygon(
            locations=locations,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.15,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_top_candidates(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add top candidate markers with rank labels."""
    fg = folium.FeatureGroup(name="Top Candidates", show=True)
    features = geojson.get("features", [])

    for feat in features:
        if feat["geometry"]["type"] != "Point":
            continue
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        rank = props.get("candidate_rank", "?")
        popup_html = (
            f"<b>Top Candidate #{rank}</b><br>"
            f"Scan: {props.get('scan_time_utc', 'N/A')}<br>"
            f"Score: {props.get('candidate_score', 'N/A')}<br>"
            f"Label: {props.get('candidate_label', 'N/A')}<br>"
            f"Window: {props.get('search_window', 'N/A')}<br>"
            f"H-dist: {props.get('horizontal_distance_km', 'N/A')} km<br>"
            f"V-dist: {props.get('vertical_distance_m', 'N/A')} m<br>"
            f"Max dBZ: {props.get('max_reflectivity_dbz', 'N/A')}<br>"
            f"Gates: {props.get('n_gates', 'N/A')}"
        )
        folium.CircleMarker(
            location=[coords[1], coords[0]],
            radius=8,
            color="#d63e2a",
            fill=True,
            fill_color="#f69730",
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)

        # Rank label via DivIcon
        folium.Marker(
            location=[coords[1], coords[0]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;font-weight:bold;color:#d63e2a;'
                    f'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,'
                    f'1px -1px 0 #fff,-1px 1px 0 #fff;">{rank}</div>'
                ),
                icon_size=(20, 20),
                icon_anchor=(-10, 10),
            ),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_all_candidates(
    m: folium.Map,
    scores_path: Path,
) -> folium.FeatureGroup:
    """Add a hidden layer with all candidate points from the scores CSV."""
    fg = folium.FeatureGroup(name="All Candidates", show=False)
    if not scores_path.exists():
        fg.add_to(m)
        return fg

    scores = pd.read_csv(scores_path)
    for _, row in scores.iterrows():
        lat = row.get("candidate_lat_deg")
        lon = row.get("candidate_lon_deg")
        if pd.isna(lat) or pd.isna(lon):
            continue
        popup_html = (
            f"<b>Candidate #{int(row.get('candidate_rank', 0))}</b><br>"
            f"Scan: {row.get('scan_time_utc', 'N/A')}<br>"
            f"Score: {row.get('candidate_score', 'N/A')}<br>"
            f"Label: {row.get('candidate_label', 'N/A')}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="gray",
            fill=True,
            fill_color="gray",
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_candidate_gates(
    m: folium.Map,
    gates_path: Path,
    top_geojson: dict,
    max_gates_per_candidate: int = 5000,
    top_n_for_gates: int = 5,
) -> folium.FeatureGroup:
    """Add gate-level points for the top N candidates (chunked read)."""
    fg = folium.FeatureGroup(name="Candidate Gates", show=False)
    if not gates_path.exists():
        fg.add_to(m)
        return fg

    # Collect (cluster_id, scan_time_utc, search_window) for top N candidates
    features = top_geojson.get("features", [])
    target_info: list[tuple[str, str, str]] = []
    for feat in features[:top_n_for_gates]:
        props = feat.get("properties", {})
        cid = str(props.get("cluster_id", ""))
        scan = str(props.get("scan_time_utc", ""))
        window = str(props.get("search_window", ""))
        if cid and scan and window:
            target_info.append((cid, scan, window))

    if not target_info:
        fg.add_to(m)
        return fg

    # Build fast-lookup sets for the two-column pre-filter
    target_scans: set[tuple[str, str]] = {
        (scan, window) for _, scan, window in target_info
    }

    # Read in chunks and filter
    filtered_rows: list[pd.DataFrame] = []

    for chunk in pd.read_csv(gates_path, chunksize=200_000):
        # Vectorized pre-filter on scan_time_utc + search_window
        mask = pd.Series(False, index=chunk.index)
        for scan_time, window in target_scans:
            mask = mask | (
                (chunk["scan_time_utc"] == scan_time)
                & (chunk["search_window"] == window)
            )
        matched = chunk[mask]
        if matched.empty:
            continue

        # Further filter by cluster_id (gates don't have cluster_id directly,
        # so we match by scan_time_utc + search_window and cap per candidate)
        filtered_rows.append(matched)

    if not filtered_rows:
        fg.add_to(m)
        return fg

    all_gates = pd.concat(filtered_rows, ignore_index=True)

    # Cap gates per candidate (scan_time_utc + search_window combo)
    for scan_time, window in target_scans:
        subset_mask = (
            (all_gates["scan_time_utc"] == scan_time)
            & (all_gates["search_window"] == window)
        )
        subset_idx = all_gates[subset_mask].index
        if len(subset_idx) > max_gates_per_candidate:
            drop_idx = subset_idx[max_gates_per_candidate:]
            all_gates = all_gates.drop(drop_idx)

    # Add markers
    for _, row in all_gates.iterrows():
        lat = row.get("gate_lat_deg")
        lon = row.get("gate_lon_deg")
        if pd.isna(lat) or pd.isna(lon):
            continue
        dbz = row.get("reflectivity_dbz")
        color = reflectivity_color(dbz) if pd.notna(dbz) else "#aaaaaa"
        popup_html = (
            f"dBZ: {dbz if pd.notna(dbz) else 'N/A'}<br>"
            f"Vel: {row.get('velocity_ms', 'N/A')} m/s<br>"
            f"Elev: {row.get('elevation_deg', 'N/A')}°<br>"
            f"Range: {row.get('range_km', 'N/A')} km"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_candidate_clusters(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add candidate cluster outlines (if present)."""
    fg = folium.FeatureGroup(name="Candidate Clusters", show=True)
    for feat in geojson.get("features", []):
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "Polygon":
            coords = geom["coordinates"]
            locations = [[pt[1], pt[0]] for pt in coords[0]]
            popup_html = (
                f"<b>Cluster {props.get('cluster_id', 'N/A')}</b><br>"
                f"Scan: {props.get('scan_time_utc', 'N/A')}<br>"
                f"Window: {props.get('search_window', 'N/A')}<br>"
                f"Gates: {props.get('n_gates', 'N/A')}"
            )
            folium.Polygon(
                locations=locations,
                color="#cc4400",
                weight=2,
                fill=True,
                fill_color="#ff8800",
                fill_opacity=0.1,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg)
    fg.add_to(m)
    return fg


def add_radar_site(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add the radar site marker."""
    fg = folium.FeatureGroup(name="Radar Site", show=True)
    for feat in geojson.get("features", []):
        if feat["geometry"]["type"] != "Point":
            continue
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        popup_html = (
            f"<b>Radar: {props.get('site_id', 'N/A')}</b><br>"
            f"Lat: {coords[1]:.4f}<br>"
            f"Lon: {coords[0]:.4f}<br>"
            f"Alt: {props.get('alt_m', 'N/A')} m"
        )
        folium.Marker(
            location=[coords[1], coords[0]],
            icon=folium.Icon(color="black", icon="tower-broadcast", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_range_rings(m: folium.Map, geojson: dict) -> folium.FeatureGroup:
    """Add radar range ring lines."""
    fg = folium.FeatureGroup(name="Range Rings", show=True)
    for feat in geojson.get("features", []):
        geom = feat["geometry"]
        props = feat.get("properties", {})
        ring_coords: list[list[float]] = []
        if geom["type"] == "LineString":
            ring_coords = geom["coordinates"]
        elif geom["type"] == "Polygon":
            ring_coords = geom["coordinates"][0]
        else:
            continue

        locations = [[pt[1], pt[0]] for pt in ring_coords]
        range_km = props.get("range_km", "?")
        folium.PolyLine(
            locations=locations,
            color="black",
            weight=1,
            dash_array="10 5",
            opacity=0.5,
        ).add_to(fg)
        # Add tooltip at the first point of the ring
        if locations:
            folium.CircleMarker(
                location=locations[0],
                radius=0,
                tooltip=f"{range_km} km",
            ).add_to(fg)
    fg.add_to(m)
    return fg


# ---------------------------------------------------------------------------
# Map assembly
# ---------------------------------------------------------------------------

def make_interactive_map(
    config_path: Path,
    radar_site: str | None = None,
) -> Path:
    """Build the interactive HTML map and return the output path."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_id = case_id_from_config(config)
    site_lower = site.lower()

    mapping_cfg = config.get("mapping", {})
    max_gates = int(mapping_cfg.get("max_gates_to_map", 5000))

    case_dir = config_path.parent
    maps_dir = case_dir / "outputs" / "maps"
    cand_dir = candidates_dir(config_path, site)

    # --- Load GeoJSON layers -----------------------------------------------
    track_geojson = load_geojson(maps_dir / "expected_track.geojson")
    grids_geojson = load_geojson(maps_dir / "grid_squares.geojson")
    top_geojson = load_geojson(maps_dir / "top_candidates.geojson")
    clusters_geojson = load_geojson(maps_dir / "candidate_clusters.geojson")
    radar_geojson = load_geojson(maps_dir / f"{site_lower}_radar_site.geojson")
    rings_geojson = load_geojson(maps_dir / f"{site_lower}_range_rings.geojson")

    # --- Compute map center from track points ------------------------------
    track_features = [
        f for f in track_geojson.get("features", [])
        if f["geometry"]["type"] == "Point"
    ]
    if track_features:
        lats = [f["geometry"]["coordinates"][1] for f in track_features]
        lons = [f["geometry"]["coordinates"][0] for f in track_features]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        center_lat, center_lon = 32.0, -110.0  # fallback

    # --- Create base map ---------------------------------------------------
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # Basemap tile layers
    folium.TileLayer("openstreetmap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="OpenTopoMap",
    ).add_to(m)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Esri WorldImagery",
    ).add_to(m)

    # --- Add feature layers ------------------------------------------------
    add_range_rings(m, rings_geojson)
    add_grid_squares(m, grids_geojson)
    add_expected_track_line(m, track_geojson)
    add_expected_track_points(m, track_geojson)
    add_candidate_clusters(m, clusters_geojson)
    add_top_candidates(m, top_geojson)
    add_radar_site(m, radar_geojson)

    # Optional CSV-based layers
    scores_path = cand_dir / "candidate_scores.csv"
    add_all_candidates(m, scores_path)

    gates_path = cand_dir / "near_track_gates.csv"
    add_candidate_gates(
        m,
        gates_path,
        top_geojson,
        max_gates_per_candidate=max_gates,
    )

    # --- Fit bounds --------------------------------------------------------
    all_lats: list[float] = []
    all_lons: list[float] = []
    for f in track_features:
        all_lats.append(f["geometry"]["coordinates"][1])
        all_lons.append(f["geometry"]["coordinates"][0])
    for f in top_geojson.get("features", []):
        if f["geometry"]["type"] == "Point":
            all_lats.append(f["geometry"]["coordinates"][1])
            all_lons.append(f["geometry"]["coordinates"][0])
    if all_lats and all_lons:
        m.fit_bounds(
            [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]],
            padding=[30, 30],
        )

    # --- Map title ---------------------------------------------------------
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                background: rgba(255,255,255,0.9); padding: 10px 15px;
                border-radius: 5px; border: 1px solid #ccc;
                font-family: Arial, sans-serif; font-size: 14px;">
        <b>PicoCAST &mdash; {case_id}</b><br>
        <span style="font-size: 11px; color: #666;">
            ⚠️ Balloon position from Maidenhead grid-square centers, not exact GPS.
            Candidate offsets are relative to the grid-center estimate.
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # --- Layer control -----------------------------------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Save --------------------------------------------------------------
    output_path = maps_dir / "interactive_candidate_map.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Wrote {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument(
        "--radar-site",
        help="Radar site to process (default: from config)",
    )
    args = parser.parse_args()
    make_interactive_map(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
