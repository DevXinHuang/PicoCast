#!/usr/bin/env python
"""Create a GIS overlay map for the deduped review-packet tracklets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import folium
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexrad_picoballoon.maidenhead_grid import grid_polygon_coords  # noqa: E402
from scripts.candidate_utils import load_config  # noqa: E402

SITE_COLORS = {
    "KEMX": "#d73027",
    "KIWA": "#1f78b4",
    "KFSX": "#fdae61",
    "KYUX": "#984ea3",
    "KEPZ": "#33a02c",
}


def parse_ids(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def site_color(site: str) -> str:
    return SITE_COLORS.get(site, "#666666")


def add_tile_layers(map_obj: folium.Map) -> None:
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/"
        "MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Topo",
    ).add_to(map_obj)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
    ).add_to(map_obj)


def add_expected_track(map_obj: folium.Map, track: pd.DataFrame) -> list[list[float]]:
    layer = folium.FeatureGroup(name="Expected telemetry track", show=True)
    track = track.sort_values("time_utc")
    coords = list(zip(track["lat_deg"], track["lon_deg"], strict=True))

    folium.PolyLine(
        coords,
        color="#222222",
        weight=4,
        opacity=0.65,
        tooltip="Expected K7UAZ track from telemetry grid centers",
    ).add_to(layer)

    for _, row in track.iterrows():
        folium.CircleMarker(
            location=[row["lat_deg"], row["lon_deg"]],
            radius=4,
            color="#111111",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.95,
            tooltip=(
                f"Expected track<br>UTC: {row['time_utc']}<br>"
                f"Grid: {row['maidenhead_grid']}<br>Alt: {row['alt_m']:.0f} m"
            ),
        ).add_to(layer)

    layer.add_to(map_obj)
    return [[lat, lon] for lat, lon in coords]


def add_maidenhead_grids(map_obj: folium.Map, track: pd.DataFrame) -> None:
    layer = folium.FeatureGroup(name="Maidenhead grid squares", show=True)
    seen = set()
    for _, row in track.iterrows():
        grid = str(row["maidenhead_grid"])
        if grid in seen:
            continue
        seen.add(grid)
        polygon = [[lat, lon] for lon, lat in grid_polygon_coords(grid)]
        folium.Polygon(
            locations=polygon,
            color="#555555",
            weight=1.5,
            fill=True,
            fill_color="#9e9e9e",
            fill_opacity=0.16,
            tooltip=f"Maidenhead grid {grid}",
        ).add_to(layer)
    layer.add_to(map_obj)


def add_radar_context(
    map_obj: folium.Map,
    config: dict,
    active_sites: set[str],
) -> list[list[float]]:
    layer = folium.FeatureGroup(name="Radar sites and range rings", show=True)
    bounds: list[list[float]] = []
    radar_sites = config.get("radar_sites", {})
    ring_distances = config.get("mapping", {}).get("range_rings_km", [50, 100, 150, 200])

    for site in sorted(active_sites):
        if site not in radar_sites:
            continue
        site_cfg = radar_sites[site]
        lat = float(site_cfg["lat"])
        lon = float(site_cfg["lon"])
        alt_m = float(site_cfg.get("alt_m", 0.0))
        color = site_color(site)
        bounds.append([lat, lon])

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red" if site == "KEMX" else "blue", icon="signal"),
            tooltip=f"{site} radar<br>Altitude: {alt_m:.0f} m",
        ).add_to(layer)

        for radius_km in ring_distances:
            folium.Circle(
                location=[lat, lon],
                radius=float(radius_km) * 1000.0,
                color=color,
                weight=1,
                fill=False,
                opacity=0.20,
                tooltip=f"{site} {radius_km} km range ring",
            ).add_to(layer)

    layer.add_to(map_obj)
    return bounds


def tracklet_popup(row: pd.Series, review_row: pd.Series | None = None) -> str:
    review_html = ""
    if review_row is not None:
        review_html = (
            f"<br><b>Review rank:</b> {int(review_row['review_rank'])}"
            f"<br><b>Review item:</b> {review_row['item_id']}"
            f"<br><b>Priority:</b> {float(review_row['review_priority_score']):.3f}"
        )
    return (
        f"<b>{row['tracklet_id']}</b><br>"
        f"Radar: {row['radar_site']}<br>"
        f"UTC: {row['scan_time_utc']}<br>"
        f"Candidate alt: {row['cluster_alt_m']:.0f} m<br>"
        f"Expected alt: {row['expected_alt_m']:.0f} m<br>"
        f"Vertical mismatch: {row['signed_vertical_m']:+.0f} m<br>"
        f"Reflectivity: {row['max_reflectivity_dbz']:.1f} dBZ"
        f"{review_html}"
    )


def add_review_tracklets(
    map_obj: folium.Map,
    review_queue: pd.DataFrame,
    points: pd.DataFrame,
    top_n: int,
) -> tuple[list[list[float]], dict[str, dict[str, float]]]:
    layer = folium.FeatureGroup(name=f"Top {top_n} review tracklets", show=True)
    bounds: list[list[float]] = []
    centroids: dict[str, dict[str, float]] = {}

    for _, review_row in review_queue.head(top_n).iterrows():
        rank = int(review_row["review_rank"])
        for tracklet_id in parse_ids(review_row["tracklet_ids"]):
            subset = points[points["tracklet_id"] == tracklet_id].sort_values("scan_time_utc")
            if subset.empty:
                continue

            site = str(subset["radar_site"].iloc[0])
            color = site_color(site)
            coords = list(zip(subset["cluster_lat_deg"], subset["cluster_lon_deg"], strict=True))
            bounds.extend([[lat, lon] for lat, lon in coords])
            centroids[tracklet_id] = {
                "lat": float(subset["cluster_lat_deg"].mean()),
                "lon": float(subset["cluster_lon_deg"].mean()),
            }

            folium.PolyLine(
                locations=coords,
                color=color,
                weight=5 if rank <= 3 else 3,
                opacity=0.88,
                tooltip=(
                    f"Rank {rank}: {review_row['item_id']}<br>"
                    f"{tracklet_id}<br>{review_row['review_reason']}"
                ),
            ).add_to(layer)

            first = subset.iloc[0]
            folium.Marker(
                location=[first["cluster_lat_deg"], first["cluster_lon_deg"]],
                icon=folium.DivIcon(
                    html=(
                        '<div style="font-size:12px;font-weight:700;color:#111;'
                        'background:white;border:1px solid #555;border-radius:10px;'
                        'padding:1px 5px;white-space:nowrap;">'
                        f"#{rank} {tracklet_id}</div>"
                    )
                ),
            ).add_to(layer)

            for _, point_row in subset.iterrows():
                folium.CircleMarker(
                    location=[point_row["cluster_lat_deg"], point_row["cluster_lon_deg"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.82,
                    popup=folium.Popup(tracklet_popup(point_row, review_row), max_width=340),
                ).add_to(layer)

    layer.add_to(map_obj)
    return bounds, centroids


def add_cross_radar_links(
    map_obj: folium.Map,
    review_queue: pd.DataFrame,
    centroids: dict[str, dict[str, float]],
    top_n: int,
) -> None:
    layer = folium.FeatureGroup(name="Review cross-radar links", show=True)
    cross_rows = review_queue[review_queue["item_type"] == "cross_radar_association"].head(top_n)

    for _, row in cross_rows.iterrows():
        tracklet_ids = parse_ids(row["tracklet_ids"])
        if len(tracklet_ids) < 2:
            continue
        first = centroids.get(tracklet_ids[0])
        second = centroids.get(tracklet_ids[1])
        if not first or not second:
            continue

        folium.PolyLine(
            locations=[[first["lat"], first["lon"]], [second["lat"], second["lon"]]],
            color="#7b3294",
            weight=3,
            opacity=0.82,
            dash_array="8, 8",
            tooltip=(
                f"Rank {int(row['review_rank'])}: {row['item_id']}<br>"
                f"{row['tracklet_ids']}<br>{row['review_reason']}"
            ),
        ).add_to(layer)

    layer.add_to(map_obj)


def add_legend(map_obj: folium.Map, case_id: str) -> None:
    legend_html = f"""
    <div style="
      position: fixed;
      top: 12px;
      left: 52px;
      z-index: 9999;
      background: rgba(255,255,255,0.94);
      border: 1px solid #999;
      border-radius: 6px;
      padding: 10px 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 12px;
      max-width: 390px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.22);
    ">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;">
        PicoCAST Review GIS Overlay - {case_id}
      </div>
      <div>
        <span style="color:#d73027;font-weight:700;">KEMX</span> and
        <span style="color:#1f78b4;font-weight:700;">KIWA</span>
        review tracklets over telemetry grid context.
      </div>
      <div style="margin-top:5px;color:#555;">
        Dashed purple lines connect cross-radar review candidates.
        This is for visual inspection only.
      </div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def write_review_geojson(
    output_path: Path,
    review_queue: pd.DataFrame,
    points: pd.DataFrame,
    top_n: int,
) -> None:
    features = []
    for _, review_row in review_queue.head(top_n).iterrows():
        for tracklet_id in parse_ids(review_row["tracklet_ids"]):
            subset = points[points["tracklet_id"] == tracklet_id].sort_values("scan_time_utc")
            if subset.empty:
                continue
            coords = [
                [float(row["cluster_lon_deg"]), float(row["cluster_lat_deg"])]
                for _, row in subset.iterrows()
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "review_rank": int(review_row["review_rank"]),
                        "review_item_id": review_row["item_id"],
                        "review_item_type": review_row["item_type"],
                        "tracklet_id": tracklet_id,
                        "radar_site": str(subset["radar_site"].iloc[0]),
                        "review_priority_score": float(review_row["review_priority_score"]),
                        "review_reason": review_row["review_reason"],
                    },
                }
            )

    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def make_review_packet_gis_overlay(config_path: Path, top_n: int = 10) -> Path:
    config = load_config(config_path)
    case_dir = config_path.parent
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    review_queue = pd.read_csv(review_dir / "tracklet_review_queue.csv")
    points = pd.read_csv(case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv")
    track = pd.read_csv(case_dir / "expected_track.csv")

    map_obj = folium.Map(
        location=[float(track["lat_deg"].mean()), float(track["lon_deg"].mean())],
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    add_tile_layers(map_obj)

    bounds = add_expected_track(map_obj, track)
    add_maidenhead_grids(map_obj, track)

    active_sites = {
        site for sites in review_queue["radar_sites"].dropna().map(parse_ids) for site in sites
    }
    bounds.extend(add_radar_context(map_obj, config, active_sites))
    review_bounds, centroids = add_review_tracklets(map_obj, review_queue, points, top_n)
    bounds.extend(review_bounds)
    add_cross_radar_links(map_obj, review_queue, centroids, top_n)

    add_legend(map_obj, str(config["case_id"]))
    folium.LayerControl(collapsed=False).add_to(map_obj)
    if bounds:
        map_obj.fit_bounds(bounds, padding=(20, 20))

    output_path = review_dir / "review_packet_gis_overlay.html"
    map_obj.save(output_path)
    write_review_geojson(
        review_dir / "review_packet_tracklets.geojson",
        review_queue,
        points,
        top_n,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--top-n", type=int, default=10, help="Number of review rows to overlay")
    args = parser.parse_args()

    output_path = make_review_packet_gis_overlay(args.config, top_n=args.top_n)
    print(f"Wrote review packet GIS overlay: {output_path}")


if __name__ == "__main__":
    main()
