#!/usr/bin/env python
"""Generate an interactive path dashboard with map + altitude panels.

Top panel: GIS map with grid squares, candidate path, and radar geometry.
Bottom panel: Time vs altitude plot with expected curve and candidate altitudes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    load_config,
    radar_site_from_config,
)

# Focus original ranks to label
LABEL_RANKS = {1, 3, 6, 7, 9}


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def mismatch_color(signed_m: float) -> str:
    """Map signed vertical mismatch to a color: green near 0, red far."""
    abs_m = abs(signed_m)
    if abs_m <= 250:
        return "#3fb950"  # green — excellent
    if abs_m <= 500:
        return "#58a6ff"  # blue — strong
    if abs_m <= 1000:
        return "#e3b341"  # yellow — moderate
    return "#f85149"      # red — poor


def score_radius(alt_score: float) -> float:
    """Map altitude consistency score to marker radius."""
    return 4 + alt_score * 8


# ---------------------------------------------------------------------------
# Altitude chart (embedded SVG/HTML)
# ---------------------------------------------------------------------------


def build_altitude_chart_html(
    path_df: pd.DataFrame,
    plausible: pd.DataFrame,
    track: pd.DataFrame,
    start_time: str,
    end_time: str,
) -> str:
    """Build an inline HTML/JS altitude-vs-time chart using canvas."""
    # Prepare data as JSON for the embedded chart
    track_sorted = track.sort_values("time_utc")
    track_times = pd.to_datetime(track_sorted["time_utc"])

    # Filter track to window with padding
    t0 = pd.Timestamp(start_time) - pd.Timedelta(minutes=15)
    t1 = pd.Timestamp(end_time) + pd.Timedelta(minutes=15)
    mask = (track_times >= t0) & (track_times <= t1)  # noqa: F841

    # Also create interpolated points covering the window
    interp_times = pd.date_range(
        max(t0, track_times.min()),
        min(t1, track_times.max()),
        periods=50,
    )
    interp_alts = np.interp(
        [t.timestamp() for t in interp_times],
        [t.timestamp() for t in track_times],
        track_sorted["alt_m"].astype(float),
    )

    # Build data arrays
    exp_data = [
        {"t": t.isoformat(), "alt": float(a)}
        for t, a in zip(interp_times, interp_alts, strict=True)
    ]

    plausible_data = []
    for _, r in plausible.iterrows():
        orig_rank = int(r["original_candidate_rank"])
        plausible_data.append({
            "t": r["scan_time_utc"],
            "alt": float(r["candidate_alt_m"]),
            "signed_v": float(r["signed_vertical_interp_m"]),
            "alt_score": float(r["altitude_consistency_score"]),
            "rank": orig_rank,
            "label": r["altitude_consistency_label"],
            "h_dist": float(r["horizontal_distance_km"]),
        })

    path_data = []
    for _, r in path_df.iterrows():
        path_data.append({
            "t": r["scan_time_utc"],
            "alt": float(r["candidate_alt_m"]),
            "exp_alt": float(r["expected_alt_m"]),
            "signed_v": float(r["signed_vertical_m"]),
            "rank": int(r["original_candidate_rank"]),
            "label": r["altitude_consistency_label"],
        })

    chart_data = json.dumps({
        "expected": exp_data,
        "plausible": plausible_data,
        "path": path_data,
    })

    return f"""
    <div id="alt-chart-container" style="
        background: #161b22;
        border-top: 2px solid #30363d;
        padding: 16px 20px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <div style="color: #f0f6fc; font-size: 14px; font-weight: 600;
                     margin-bottom: 8px;">
            Altitude vs Time — Expected Profile &amp; Candidate Points
        </div>
        <canvas id="altChart" style="width: 100%; height: 220px; display: block;">
        </canvas>
        <div style="color: #8b949e; font-size: 11px; margin-top: 6px;">
            Blue line = expected balloon altitude (interpolated).
            Green dots = excellent match (≤250 m).
            Blue dots = strong (≤500 m).
            Yellow = moderate. Red = poor.
            Connected line = selected candidate path.
        </div>
    </div>
    <script>
    (function() {{
        var data = {chart_data};
        var canvas = document.getElementById('altChart');
        var rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width - 40;
        canvas.height = 220;
        var ctx = canvas.getContext('2d');
        var W = canvas.width, H = canvas.height;
        var pad = {{l: 60, r: 20, t: 10, b: 30}};
        var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

        // Parse times
        function toTs(s) {{ return new Date(s).getTime(); }}

        // Collect all timestamps and altitudes for scaling
        var allT = [], allA = [];
        data.expected.forEach(function(d) {{ allT.push(toTs(d.t)); allA.push(d.alt); }});
        data.plausible.forEach(function(d) {{ allT.push(toTs(d.t)); allA.push(d.alt); }});
        data.path.forEach(function(d) {{
            allT.push(toTs(d.t)); allA.push(d.alt); allA.push(d.exp_alt);
        }});

        var tMin = Math.min.apply(null, allT), tMax = Math.max.apply(null, allT);
        var aMin = Math.min.apply(null, allA) - 500, aMax = Math.max.apply(null, allA) + 500;

        function xOf(t) {{ return pad.l + (t - tMin) / (tMax - tMin) * pw; }}
        function yOf(a) {{ return pad.t + ph - (a - aMin) / (aMax - aMin) * ph; }}

        // Background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, W, H);

        // ±500m band around expected
        ctx.fillStyle = 'rgba(88, 166, 255, 0.08)';
        ctx.beginPath();
        data.expected.forEach(function(d, i) {{
            var x = xOf(toTs(d.t)), y = yOf(d.alt + 500);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        for (var i = data.expected.length - 1; i >= 0; i--) {{
            var d = data.expected[i];
            ctx.lineTo(xOf(toTs(d.t)), yOf(d.alt - 500));
        }}
        ctx.closePath();
        ctx.fill();

        // ±250m band
        ctx.fillStyle = 'rgba(88, 166, 255, 0.1)';
        ctx.beginPath();
        data.expected.forEach(function(d, i) {{
            var x = xOf(toTs(d.t)), y = yOf(d.alt + 250);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        for (var i = data.expected.length - 1; i >= 0; i--) {{
            var d = data.expected[i];
            ctx.lineTo(xOf(toTs(d.t)), yOf(d.alt - 250));
        }}
        ctx.closePath();
        ctx.fill();

        // Expected altitude line
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        data.expected.forEach(function(d, i) {{
            var x = xOf(toTs(d.t)), y = yOf(d.alt);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        // All plausible candidates
        data.plausible.forEach(function(d) {{
            var x = xOf(toTs(d.t)), y = yOf(d.alt);
            var abs_v = Math.abs(d.signed_v);
            var color = abs_v <= 250 ? '#3fb950' : abs_v <= 500 ? '#58a6ff' :
                        abs_v <= 1000 ? '#e3b341' : '#f85149';
            var r = 3 + d.alt_score * 4;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.4;
            ctx.fill();
            ctx.globalAlpha = 1.0;
        }});

        // Selected path line
        if (data.path.length >= 2) {{
            ctx.strokeStyle = '#f0883e';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            data.path.forEach(function(d, i) {{
                var x = xOf(toTs(d.t)), y = yOf(d.alt);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            ctx.setLineDash([]);
        }}

        // Selected path points (larger)
        data.path.forEach(function(d) {{
            var x = xOf(toTs(d.t)), y = yOf(d.alt);
            var abs_v = Math.abs(d.signed_v);
            var color = abs_v <= 250 ? '#3fb950' : abs_v <= 500 ? '#58a6ff' :
                        abs_v <= 1000 ? '#e3b341' : '#f85149';
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#f0f6fc';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Label
            var labelRanks = [1, 3, 6, 7, 9];
            if (labelRanks.indexOf(d.rank) >= 0) {{
                ctx.fillStyle = '#f0f6fc';
                ctx.font = 'bold 10px sans-serif';
                ctx.fillText('R' + d.rank, x + 8, y - 4);
            }}
        }});

        // Y-axis labels
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px sans-serif';
        var nTicks = 5;
        for (var i = 0; i <= nTicks; i++) {{
            var alt = aMin + (aMax - aMin) * i / nTicks;
            var y = yOf(alt);
            ctx.fillText(Math.round(alt) + ' m', 2, y + 3);
            ctx.strokeStyle = '#21262d';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(pad.l, y);
            ctx.lineTo(W - pad.r, y);
            ctx.stroke();
        }}

        // X-axis labels
        var nXTicks = 5;
        for (var i = 0; i <= nXTicks; i++) {{
            var t = tMin + (tMax - tMin) * i / nXTicks;
            var x = xOf(t);
            var d = new Date(t);
            var label = ('0' + d.getUTCHours()).slice(-2) + ':' +
                        ('0' + d.getUTCMinutes()).slice(-2);
            ctx.fillStyle = '#8b949e';
            ctx.fillText(label, x - 12, H - 5);
        }}
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------


def make_path_dashboard(
    config_path: Path,
    radar_site: str | None = None,
) -> Path:
    """Build the interactive path dashboard HTML."""
    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_dir = config_path.parent
    cand_dir = candidates_dir(config_path, site)
    site_lower = site.lower()
    maps_dir = case_dir / "outputs" / "maps"
    path_dir = cand_dir / "path_fit"

    # Load data
    track = pd.read_csv(case_dir / "expected_track.csv")
    track = track.sort_values("time_utc").reset_index(drop=True)

    path_df = pd.read_csv(path_dir / "candidate_path.csv")

    alt_csv = cand_dir / "altitude_validation" / "altitude_prioritized_candidates.csv"
    plausible_all = pd.read_csv(alt_csv)

    # Filter plausible to window
    if not path_df.empty:
        t0 = pd.Timestamp(path_df["scan_time_utc"].min()) - pd.Timedelta(minutes=5)
        t1 = pd.Timestamp(path_df["scan_time_utc"].max()) + pd.Timedelta(minutes=5)
    else:
        t0 = pd.Timestamp("2026-03-22T20:00:00Z")
        t1 = pd.Timestamp("2026-03-22T20:30:00Z")

    plausible_all["scan_dt"] = pd.to_datetime(plausible_all["scan_time_utc"])
    plausible = plausible_all[
        (plausible_all["scan_dt"] >= t0) & (plausible_all["scan_dt"] <= t1)
        & (plausible_all["altitude_consistency_score"] >= 0.6)
    ].copy()

    with open(path_dir / "candidate_path.geojson", encoding="utf-8") as f:
        path_gj = json.load(f)
    with open(path_dir / "smoothed_candidate_path.geojson", encoding="utf-8") as f:
        smooth_gj = json.load(f)

    # Load map layers
    def load_gj(path: Path) -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {"type": "FeatureCollection", "features": []}

    grid_gj = load_gj(maps_dir / "grid_squares.geojson")
    track_gj = load_gj(maps_dir / "expected_track.geojson")
    radar_gj = load_gj(maps_dir / f"{site_lower}_radar_site.geojson")
    rings_gj = load_gj(maps_dir / f"{site_lower}_range_rings.geojson")

    # ---- Build folium map ----
    if not path_df.empty:
        center_lat = path_df["candidate_lat_deg"].mean()
        center_lon = path_df["candidate_lon_deg"].mean()
    else:
        center_lat, center_lon = 32.33, -110.57

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Topo",
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Satellite",
    ).add_to(fmap)

    # Grid squares
    grid_group = folium.FeatureGroup(name="Grid Squares")
    for feat in grid_gj.get("features", []):
        folium.GeoJson(
            feat,
            style_function=lambda _: {
                "color": "#58a6ff", "weight": 1.5,
                "fillOpacity": 0.05, "dashArray": "4 4",
            },
            tooltip=feat.get("properties", {}).get("maidenhead_grid", ""),
        ).add_to(grid_group)
    grid_group.add_to(fmap)

    # Expected track
    track_group = folium.FeatureGroup(name="Expected Track (grid centers)")
    for feat in track_gj.get("features", []):
        coords = feat["geometry"]["coordinates"]
        props = feat.get("properties", {})
        folium.CircleMarker(
            location=[coords[1], coords[0]],
            radius=5, color="#2196F3", fill=True,
            fill_color="#2196F3", fill_opacity=0.7,
            tooltip=f"{props.get('time_utc', '')} | alt={props.get('alt_m', '')} m",
        ).add_to(track_group)
    track_group.add_to(fmap)

    # All plausible candidates
    plausible_group = folium.FeatureGroup(name="All Plausible Candidates")
    for _, r in plausible.iterrows():
        color = mismatch_color(r["signed_vertical_interp_m"])
        radius = score_radius(r["altitude_consistency_score"])
        orig_rank = int(r["original_candidate_rank"])
        tooltip = (
            f"OrigR{orig_rank} | "
            f"vert={r['signed_vertical_interp_m']:+.0f}m | "
            f"h_dist={r['horizontal_distance_km']:.1f}km | "
            f"{r['altitude_consistency_label'].replace('_', ' ')}"
        )
        folium.CircleMarker(
            location=[r["candidate_lat_deg"], r["candidate_lon_deg"]],
            radius=radius, color=color, fill=True,
            fill_color=color, fill_opacity=0.4,
            tooltip=tooltip,
        ).add_to(plausible_group)
    plausible_group.add_to(fmap)

    # Selected path line
    path_line_group = folium.FeatureGroup(name="Selected Candidate Path")
    for feat in path_gj.get("features", []):
        if feat["geometry"]["type"] == "LineString":
            coords = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
            folium.PolyLine(
                coords, color="#f0883e", weight=3, opacity=0.9,
                tooltip="Selected candidate path",
            ).add_to(path_line_group)
    path_line_group.add_to(fmap)

    # Selected path points
    path_pts_group = folium.FeatureGroup(name="Selected Path Points")
    for _, r in path_df.iterrows():
        color = mismatch_color(r["signed_vertical_m"])
        orig_rank = int(r["original_candidate_rank"])
        popup_html = (
            f"<b>Path Step — OrigR{orig_rank}</b><br>"
            f"Time: {r['scan_time_utc']}<br>"
            f"Altitude: {r['candidate_alt_m']:.0f} m (gate beam-center)<br>"
            f"Expected: {r['expected_alt_m']:.0f} m (interpolated)<br>"
            f"Vert mismatch: {r['signed_vertical_m']:+.0f} m<br>"
            f"H-dist: {r['horizontal_distance_km']:.1f} km<br>"
            f"Alt label: {r['altitude_consistency_label'].replace('_', ' ')}<br>"
            f"Speed: {r['segment_speed_kmh']:.0f} km/h<br>"
            f"<i>Balloon position from Maidenhead grid squares, not GPS.</i>"
        )
        rank_label = f"R{orig_rank}" if orig_rank in LABEL_RANKS else ""
        folium.CircleMarker(
            location=[r["candidate_lat_deg"], r["candidate_lon_deg"]],
            radius=9, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"R{orig_rank}: {r['signed_vertical_m']:+.0f}m vert",
        ).add_to(path_pts_group)

        if rank_label:
            folium.Marker(
                location=[r["candidate_lat_deg"], r["candidate_lon_deg"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:11px;font-weight:bold;'
                         f'color:#f0f6fc;text-shadow:0 0 3px #000;'
                         f'white-space:nowrap;">{rank_label}</div>',
                    icon_size=(30, 15), icon_anchor=(0, -12),
                ),
            ).add_to(path_pts_group)
    path_pts_group.add_to(fmap)

    # Smoothed path
    smooth_group = folium.FeatureGroup(name="Smoothed Path", show=False)
    for feat in smooth_gj.get("features", []):
        if feat["geometry"]["type"] == "LineString":
            coords = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
            folium.PolyLine(
                coords, color="#a371f7", weight=2, opacity=0.7,
                dash_array="6 4",
                tooltip="Smoothed candidate path (rolling avg)",
            ).add_to(smooth_group)
    smooth_group.add_to(fmap)

    # Radar site
    radar_group = folium.FeatureGroup(name=f"{site} Radar Site")
    for feat in radar_gj.get("features", []):
        coords = feat["geometry"]["coordinates"]
        folium.Marker(
            location=[coords[1], coords[0]],
            icon=folium.Icon(
                color="darkblue", icon="tower-broadcast", prefix="fa"
            ),
            tooltip=site,
        ).add_to(radar_group)
    radar_group.add_to(fmap)

    # Range rings
    ring_group = folium.FeatureGroup(name="Range Rings", show=False)
    for feat in rings_gj.get("features", []):
        folium.GeoJson(
            feat,
            style_function=lambda _: {
                "color": "#666", "weight": 1,
                "fillOpacity": 0, "dashArray": "4 4",
            },
            tooltip=feat.get("properties", {}).get("label", ""),
        ).add_to(ring_group)
    ring_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # ---- Title banner ----
    n_pts = len(path_df)
    med_vert = path_df["abs_vertical_distance_m"].median() if n_pts else 0
    title_html = (
        '<div style="position:fixed;top:10px;left:60px;z-index:1000;'
        'background:#161b22;padding:10px 16px;border-radius:8px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.5);font-size:13px;'
        'color:#c9d1d9;max-width:520px;border:1px solid #30363d;">'
        '<b style="color:#58a6ff;font-size:15px;">'
        'PicoCAST Altitude-Constrained Candidate Trajectory</b><br>'
        f'K7UAZ 2026-03-22 · {site} · '
        f'{n_pts} path points · '
        f'median vert mismatch {med_vert:.0f} m<br>'
        '<span style="font-size:11px;color:#8b949e;">'
        'Balloon position from Maidenhead grid-square centers, not GPS. '
        'This is a radar-assisted candidate path, not a confirmed detection.'
        '</span></div>'
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    # ---- Bottom panel: altitude chart ----
    start_str = str(t0.isoformat())
    end_str = str(t1.isoformat())
    alt_chart = build_altitude_chart_html(
        path_df, plausible, track, start_str, end_str,
    )
    fmap.get_root().html.add_child(folium.Element(alt_chart))

    # ---- Save ----
    out_path = path_dir / "interactive_path_dashboard.html"
    fmap.save(str(out_path))
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site, default from config")
    args = parser.parse_args()
    make_path_dashboard(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
