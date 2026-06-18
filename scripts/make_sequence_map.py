#!/usr/bin/env python
"""Generate a multi-candidate temporal sequence map for selected ranks."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import folium
from folium import DivIcon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
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

def make_sequence_map(
    config_path: Path,
    radar_site: str | None = None,
    ranks: list[int] | None = None,
) -> Path:
    """Build a folium map showing multiple candidates in temporal sequence."""

    if not ranks:
        raise SystemExit("At least one rank must be specified via --ranks.")

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    site_lower = site.lower()
    case_dir = config_path.parent
    maps_dir = case_dir / "outputs" / "maps"

    # ---- load GeoJSON layers ------------------------------------------------
    expected_track = load_geojson(maps_dir / "expected_track.geojson")
    grid_squares = load_geojson(maps_dir / "grid_squares.geojson")
    top_candidates = load_geojson(maps_dir / "top_candidates.geojson")
    radar_site_gj = load_geojson(maps_dir / f"{site_lower}_radar_site.geojson")
    range_rings_gj = load_geojson(maps_dir / f"{site_lower}_range_rings.geojson")

    # ---- select candidates by rank ------------------------------------------
    rank_set = set(ranks)
    selected: list[dict] = []
    for feat in top_candidates.get("features", []):
        if feat.get("properties", {}).get("candidate_rank") in rank_set:
            selected.append(feat)
    if not selected:
        raise SystemExit(f"No candidates found for ranks {ranks}")

    # sort by scan_time_utc
    selected.sort(key=lambda f: parse_utc(f["properties"]["scan_time_utc"]))

    # ---- determine time range -----------------------------------------------
    times = [parse_utc(f["properties"]["scan_time_utc"]) for f in selected]
    t_min, t_max = min(times), max(times)
    padding = timedelta(minutes=10)
    t_range_start = t_min - padding
    t_range_end = t_max + padding

    # output filename from hour:minute range
    start_hm = t_min.strftime("%H%M")
    end_hm = t_max.strftime("%H%M")

    # ---- filter expected track to time range --------------------------------
    track_features = expected_track.get("features", [])
    track_in_range: list[dict] = []
    for feat in track_features:
        time_str = feat.get("properties", {}).get("time_utc")
        if not time_str:
            continue
        dt = parse_utc(time_str)
        if t_range_start <= dt <= t_range_end:
            track_in_range.append(feat)
    track_in_range.sort(key=lambda f: parse_utc(f["properties"]["time_utc"]))

    # ---- build map ----------------------------------------------------------
    all_lats: list[float] = []
    all_lons: list[float] = []
    for feat in selected:
        lon, lat = feat["geometry"]["coordinates"][:2]
        all_lats.append(lat)
        all_lons.append(lon)
    for feat in track_in_range:
        lon, lat = feat["geometry"]["coordinates"][:2]
        all_lats.append(lat)
        all_lons.append(lon)

    center_lat = sum(all_lats) / len(all_lats) if all_lats else 32.0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else -110.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10)

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

    # ---- layer: expected track points + line --------------------------------
    track_group = folium.FeatureGroup(name="Expected Track")
    if track_in_range:
        # connecting line
        track_coords = [
            [feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0]]
            for feat in track_in_range
        ]
        folium.PolyLine(
            locations=track_coords,
            color="#3388ff",
            weight=2,
            opacity=0.7,
        ).add_to(track_group)

        # individual points
        alt_vals = [
            feat["geometry"]["coordinates"][2]
            for feat in track_in_range
            if len(feat["geometry"]["coordinates"]) > 2
        ]
        min_alt = min(alt_vals) if alt_vals else 0
        max_alt = max(alt_vals) if alt_vals else 0
        for feat in track_in_range:
            coords = feat["geometry"]["coordinates"]
            lon, lat = coords[:2]
            alt = coords[2] if len(coords) > 2 else 0
            color = altitude_color(alt, min_alt, max_alt)
            props = feat["properties"]
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Track {props.get('time_utc', '')} alt={alt:.0f} m",
            ).add_to(track_group)
    track_group.add_to(fmap)

    # ---- layer: grid squares for selected candidates ------------------------
    grid_group = folium.FeatureGroup(name="Grid Squares")
    for feat in selected:
        cand_time = parse_utc(feat["properties"]["scan_time_utc"])
        tp = find_closest_track_point(track_features, cand_time)
        if tp is None:
            continue
        tp_props = tp["properties"]
        maidenhead = tp_props.get("maidenhead_grid", "")
        tp_time = tp_props.get("time_utc", "")
        grid_feat = find_grid_square(grid_squares.get("features", []), maidenhead, tp_time)
        if grid_feat is not None:
            folium.GeoJson(
                grid_feat,
                style_function=lambda _: {
                    "fillColor": "#3388ff",
                    "color": "#3388ff",
                    "weight": 2,
                    "fillOpacity": 0.10,
                },
                tooltip=f"Grid: {maidenhead}",
            ).add_to(grid_group)
    grid_group.add_to(fmap)

    # ---- layer: candidate centers with rank labels --------------------------
    cand_group = folium.FeatureGroup(name="Candidates")
    for feat in selected:
        coords = feat["geometry"]["coordinates"]
        lon, lat = coords[:2]
        props = feat["properties"]
        rank = props.get("candidate_rank", "?")
        score = props.get("candidate_score", 0)
        scan_t = props.get("scan_time_utc", "")

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            tooltip=f"Rank {rank}: score {score:.3f} @ {scan_t}",
        ).add_to(cand_group)

        # DivIcon label
        folium.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(40, 20),
                icon_anchor=(0, 0),
                html=(
                    f'<div style="font-size:11px;font-weight:bold;'
                    f'color:#b00;text-shadow:1px 1px 0 #fff;">'
                    f'R{rank}</div>'
                ),
            ),
        ).add_to(cand_group)

        # time label below rank
        time_short = parse_utc(scan_t).strftime("%H:%M:%S")
        folium.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(60, 16),
                icon_anchor=(0, -14),
                html=(
                    f'<div style="font-size:9px;color:#555;'
                    f'text-shadow:1px 1px 0 #fff;">{time_short}</div>'
                ),
            ),
        ).add_to(cand_group)
    cand_group.add_to(fmap)

    # ---- layer: connecting line between candidates (time order) -------------
    seq_group = folium.FeatureGroup(name="Candidate Sequence")
    if len(selected) > 1:
        seq_coords = [
            [feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0]]
            for feat in selected
        ]
        folium.PolyLine(
            locations=seq_coords,
            color="orange",
            weight=2,
            dash_array="8 4",
            tooltip="Candidate sequence (time order)",
        ).add_to(seq_group)
    seq_group.add_to(fmap)

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
    rank_list = ", ".join(str(f["properties"]["candidate_rank"]) for f in selected)
    title_html = (
        f'<div style="position:fixed;top:10px;left:60px;z-index:1000;'
        f'background:white;padding:8px 14px;border-radius:6px;'
        f'box-shadow:0 2px 6px rgba(0,0,0,.3);font-size:14px;">'
        f'<b>Sequence Map</b> – {start_hm}–{end_hm} UTC<br>'
        f'Ranks: {rank_list}'
        f'</div>'
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    # ---- fit bounds ---------------------------------------------------------
    if all_lats and all_lons:
        fmap.fit_bounds(
            [[min(all_lats) - 0.05, min(all_lons) - 0.05],
             [max(all_lats) + 0.05, max(all_lons) + 0.05]]
        )

    folium.LayerControl(collapsed=False).add_to(fmap)

    # ---- save ---------------------------------------------------------------
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / f"sequence_{start_hm}_{end_hm}_map.html"
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
    parser.add_argument(
        "--ranks", type=int, nargs="+", required=True,
        help="Candidate ranks to include (e.g. --ranks 1 3 6 7 9)",
    )
    args = parser.parse_args()
    make_sequence_map(args.config, radar_site=args.radar_site, ranks=args.ranks)


if __name__ == "__main__":
    main()
