#!/usr/bin/env python
"""Generate a focused validation map for a single near-track radar candidate."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import folium
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    load_config,
    radar_site_from_config,
)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def reflectivity_color(dbz: float) -> str:
    """Map reflectivity dBZ to a hex color."""
    if dbz < -10:
        return "#cccccc"
    if dbz < 0:
        return "#88bbff"
    if dbz < 10:
        return "#44ff44"
    if dbz < 20:
        return "#ffff00"
    if dbz < 30:
        return "#ff8800"
    return "#ff0000"


def altitude_color(alt_m: float, min_alt: float, max_alt: float) -> str:
    """Map altitude to a green→red gradient via HLS."""
    if max_alt == min_alt:
        return "#3388ff"
    t = (alt_m - min_alt) / (max_alt - min_alt)
    hue = 0.33 * (1 - t)
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp to a timezone-aware datetime."""
    ts = ts.strip().rstrip("Z")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def find_closest_track_point(
    track_features: list[dict],
    target_time: datetime,
) -> dict | None:
    """Return the expected-track feature whose time_utc is nearest *target_time*."""
    best, best_delta = None, None
    for feat in track_features:
        props = feat.get("properties", {})
        time_str = props.get("time_utc")
        if not time_str:
            continue
        dt = parse_utc(time_str)
        delta = abs((dt - target_time).total_seconds())
        if best_delta is None or delta < best_delta:
            best, best_delta = feat, delta
    return best


def find_grid_square(
    grid_features: list[dict],
    maidenhead: str,
    time_utc: str,
) -> dict | None:
    """Return the grid-square polygon matching *maidenhead* and *time_utc*."""
    for feat in grid_features:
        props = feat.get("properties", {})
        if props.get("maidenhead_grid") == maidenhead and props.get("time_utc") == time_utc:
            return feat
    return None


# ---------------------------------------------------------------------------
# GeoJSON loaders
# ---------------------------------------------------------------------------

def load_geojson(path: Path) -> dict:
    """Load a GeoJSON file, returning an empty FeatureCollection if missing."""
    if not path.exists():
        print(f"Warning: {path} not found – skipping layer.")
        return {"type": "FeatureCollection", "features": []}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Main map builder
# ---------------------------------------------------------------------------

def make_candidate_validation_map(
    config_path: Path,
    radar_site: str | None = None,
    rank: int = 1,
) -> Path:
    """Build a folium map centred on a single ranked candidate."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    site_lower = site.lower()
    case_dir = config_path.parent
    maps_dir = case_dir / "outputs" / "maps"
    cand_dir = candidates_dir(config_path, site)
    mapping_cfg = config.get("mapping", {})
    max_gates = int(mapping_cfg.get("max_gates_to_map", 5000))

    # ---- load GeoJSON layers ------------------------------------------------
    expected_track = load_geojson(maps_dir / "expected_track.geojson")
    grid_squares = load_geojson(maps_dir / "grid_squares.geojson")
    top_candidates = load_geojson(maps_dir / "top_candidates.geojson")
    radar_site_gj = load_geojson(maps_dir / f"{site_lower}_radar_site.geojson")
    range_rings_gj = load_geojson(maps_dir / f"{site_lower}_range_rings.geojson")

    # ---- select the candidate by rank ---------------------------------------
    candidate_feat = None
    for feat in top_candidates.get("features", []):
        if feat.get("properties", {}).get("candidate_rank") == rank:
            candidate_feat = feat
            break
    if candidate_feat is None:
        raise SystemExit(f"No candidate with rank {rank} in top_candidates.geojson")
    cand_props = candidate_feat["properties"]
    cand_lon, cand_lat = candidate_feat["geometry"]["coordinates"][:2]
    scan_time_str = cand_props["scan_time_utc"]
    scan_time = parse_utc(scan_time_str)
    search_window = cand_props.get("search_window", "normal")

    # ---- load altitude validation data if available -------------------------
    alt_csv = cand_dir / "altitude_validation" / "altitude_prioritized_candidates.csv"
    alt_info: dict = {}
    if alt_csv.exists():
        alt_df = pd.read_csv(alt_csv)
        match = alt_df[alt_df["original_candidate_rank"] == rank]
        if not match.empty:
            row = match.iloc[0]
            alt_info = {
                "expected_alt_m": row.get("interpolated_expected_alt_m"),
                "candidate_alt_m": row.get("candidate_alt_m"),
                "signed_vertical_m": row.get("signed_vertical_interp_m"),
                "abs_vertical_m": row.get("abs_vertical_interp_m"),
                "altitude_label": row.get("altitude_consistency_label", ""),
                "altitude_priority_rank": row.get("altitude_priority_rank"),
            }

    # ---- find the closest expected-track point ------------------------------
    track_point = find_closest_track_point(expected_track.get("features", []), scan_time)
    if track_point is None:
        raise SystemExit("No expected-track point found for the candidate scan time.")
    tp_props = track_point["properties"]
    tp_lon, tp_lat = track_point["geometry"]["coordinates"][:2]
    tp_time_str = tp_props["time_utc"]

    # ---- find the grid-square polygon for that track point ------------------
    maidenhead = tp_props.get("maidenhead_grid", "")
    grid_feat = find_grid_square(grid_squares.get("features", []), maidenhead, tp_time_str)

    # ---- read nearby gates for this candidate (chunked) ---------------------
    gates_csv = cand_dir / "near_track_gates.csv"
    gate_rows: list[pd.DataFrame] = []
    if gates_csv.exists():
        for chunk in pd.read_csv(gates_csv, chunksize=50_000):
            matched = chunk[
                (chunk["scan_time_utc"] == scan_time_str)
                & (chunk["search_window"] == search_window)
            ]
            gate_rows.append(matched)
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    if len(gates) > max_gates:
        gates = gates.head(max_gates)

    # ---- build folium map ---------------------------------------------------
    center_lat = (cand_lat + tp_lat) / 2
    center_lon = (cand_lon + tp_lon) / 2
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # basemaps
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="OpenTopoMap",
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri WorldImagery",
    ).add_to(fmap)

    # ---- layer: grid square -------------------------------------------------
    if grid_feat is not None:
        grid_group = folium.FeatureGroup(name="Grid Square")
        folium.GeoJson(
            grid_feat,
            style_function=lambda _: {
                "fillColor": "#3388ff",
                "color": "#3388ff",
                "weight": 2,
                "fillOpacity": 0.15,
            },
            tooltip=f"Grid: {maidenhead}",
        ).add_to(grid_group)
        grid_group.add_to(fmap)

    # ---- layer: expected grid center ----------------------------------------
    track_group = folium.FeatureGroup(name="Expected Grid Center")
    folium.CircleMarker(
        location=[tp_lat, tp_lon],
        radius=7,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.7,
        tooltip=f"Expected center ({tp_time_str})",
    ).add_to(track_group)
    track_group.add_to(fmap)

    # ---- layer: candidate center --------------------------------------------
    cand_group = folium.FeatureGroup(name="Candidate Center")
    score = cand_props.get("candidate_score", 0)
    label = cand_props.get("candidate_label", "")
    h_dist = cand_props.get("horizontal_distance_km", 0)
    v_dist = cand_props.get("vertical_distance_m", 0)
    max_dbz = cand_props.get("max_reflectivity_dbz", None)

    # Build altitude info lines for popup
    alt_lines = ""
    if alt_info:
        exp_a = alt_info.get("expected_alt_m")
        cand_a = alt_info.get("candidate_alt_m")
        signed_v = alt_info.get("signed_vertical_m")
        alt_label = alt_info.get("altitude_label", "")
        alt_rank = alt_info.get("altitude_priority_rank")
        alt_lines = (
            f"<b>Altitude:</b><br>"
            f"&nbsp;Expected balloon alt: {exp_a:.0f} m (interpolated)<br>"
            f"&nbsp;Candidate gate alt: {cand_a:.0f} m (beam-center)<br>"
            f"&nbsp;Vertical mismatch: {signed_v:+.0f} m<br>"
            f"&nbsp;Altitude match: {alt_label.replace('_', ' ')}<br>"
            f"&nbsp;Altitude priority rank: {alt_rank}<br>"
        )

    popup_html = (
        f"<b>Rank {rank}</b> – {label}<br>"
        f"Score: {score:.3f}<br>"
        f"H-dist: {h_dist:.2f} km<br>"
        f"V-dist: {v_dist:.0f} m<br>"
        f"Max dBZ: {max_dbz}<br>"
        f"Scan: {scan_time_str}<br>"
        f"{alt_lines}"
        f"<i>Balloon position is estimated from Maidenhead grid-square "
        f"centers, not exact GPS. Radar gate altitude is beam-center "
        f"altitude (beam width increases with range).</i>"
    )

    folium.CircleMarker(
        location=[cand_lat, cand_lon],
        radius=8,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=320),
        tooltip=f"Rank {rank}: score {score:.3f}",
    ).add_to(cand_group)
    cand_group.add_to(fmap)

    # ---- layer: line from grid center to candidate --------------------------
    line_group = folium.FeatureGroup(name="Distance Line")
    folium.PolyLine(
        locations=[[tp_lat, tp_lon], [cand_lat, cand_lon]],
        color="red",
        weight=2,
        tooltip=f"H-distance: {h_dist:.2f} km",
    ).add_to(line_group)
    line_group.add_to(fmap)

    # ---- layer: nearby gates ------------------------------------------------
    if not gates.empty:
        gate_group = folium.FeatureGroup(name="Nearby Gates")
        for _, g in gates.iterrows():
            dbz = g.get("reflectivity_dbz")
            color = reflectivity_color(dbz) if pd.notna(dbz) else "#cccccc"
            tip = f"dBZ={dbz}, alt={g.get('gate_alt_m', '')} m"
            folium.CircleMarker(
                location=[g["gate_lat_deg"], g["gate_lon_deg"]],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=tip,
            ).add_to(gate_group)
        gate_group.add_to(fmap)

    # ---- layer: radar site marker -------------------------------------------
    radar_group = folium.FeatureGroup(name=f"{site} Radar Site")
    for feat in radar_site_gj.get("features", []):
        coords = feat["geometry"]["coordinates"]
        folium.Marker(
            location=[coords[1], coords[0]],
            icon=folium.Icon(color="darkblue", icon="tower-broadcast", prefix="fa"),
            tooltip=site,
        ).add_to(radar_group)
    radar_group.add_to(fmap)

    # ---- layer: range rings -------------------------------------------------
    ring_group = folium.FeatureGroup(name="Range Rings")
    for feat in range_rings_gj.get("features", []):
        folium.GeoJson(
            feat,
            style_function=lambda _: {
                "color": "#666666",
                "weight": 1,
                "fillOpacity": 0,
                "dashArray": "4 4",
            },
            tooltip=feat.get("properties", {}).get("label", ""),
        ).add_to(ring_group)
    ring_group.add_to(fmap)

    # ---- title --------------------------------------------------------------
    # Build altitude summary for title
    alt_title = ""
    if alt_info:
        signed_v = alt_info.get("signed_vertical_m")
        alt_label = alt_info.get("altitude_label", "").replace("_", " ")
        alt_title = (
            f'<br>Alt mismatch: {signed_v:+.0f} m · {alt_label}'
        )

    title_html = (
        f'<div style="position:fixed;top:10px;left:60px;z-index:1000;'
        f'background:white;padding:8px 14px;border-radius:6px;'
        f'box-shadow:0 2px 6px rgba(0,0,0,.3);font-size:14px;">'
        f'<b>Rank {rank}</b> – {label} – score {score:.3f}<br>'
        f'H-dist {h_dist:.2f} km · V-dist {v_dist:.0f} m · '
        f'max dBZ {max_dbz}{alt_title}<br>'
        f'<span style="font-size:11px;color:#666;">'
        f'Balloon position estimated from Maidenhead grid-square centers, '
        f'not exact GPS. Gate altitude is beam-center (beam width '
        f'increases with range).</span>'
        f'</div>'
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    # ---- fit bounds ---------------------------------------------------------
    bounds_lats = [cand_lat, tp_lat]
    bounds_lons = [cand_lon, tp_lon]
    if grid_feat is not None:
        for ring in grid_feat["geometry"].get("coordinates", []):
            coords_list = ring if grid_feat["geometry"]["type"] == "Polygon" else ring
            for coord in coords_list:
                bounds_lons.append(coord[0])
                bounds_lats.append(coord[1])
    fmap.fit_bounds(
        [[min(bounds_lats) - 0.02, min(bounds_lons) - 0.02],
         [max(bounds_lats) + 0.02, max(bounds_lons) + 0.02]]
    )

    folium.LayerControl(collapsed=False).add_to(fmap)

    # ---- save ---------------------------------------------------------------
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / f"rank_{rank:02d}_validation_map.html"
    fmap.save(str(output_path))
    print(f"Wrote {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    parser.add_argument("--rank", type=int, default=1, help="Candidate rank to map (default: 1)")
    args = parser.parse_args()
    make_candidate_validation_map(args.config, radar_site=args.radar_site, rank=args.rank)


if __name__ == "__main__":
    main()
