#!/usr/bin/env python
# ruff: noqa: E501
"""Phase 7: Generate interactive regional discovery dashboard (GIS Map + Altitude Profile)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import folium
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    horizontal_distance_km,
)

haversine_distance_km = horizontal_distance_km


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="Create the interactive regional discovery dashboard.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites (primary + secondary)")
    parser.add_argument("--radar-site", type=str, help="Not used but allowed for interface consistency")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already generated dashboard")
    
    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent
    
    # Read geometry report
    geometry_csv_path = case_dir / "nexrad" / "regional_radar_geometry.csv"
    if not geometry_csv_path.exists():
        raise FileNotFoundError(f"Geometry report {geometry_csv_path} not found.")
        
    geom_df = pd.read_csv(geometry_csv_path)
    included_radars = geom_df[geom_df["geometry_status"] == "include"]["radar_site"].tolist()
    
    discovery = config.get("discovery", {})
    primary = discovery.get("radar_sites_primary", [])
    secondary = discovery.get("radar_sites_secondary", [])
    
    if args.all_sites:
        target_sites = primary + secondary
    else:
        target_sites = primary
        
    active_sites = [r for r in target_sites if r in included_radars]
    
    out_dir = case_dir / "outputs" / "discovery"
    out_html = out_dir / "regional_discovery_dashboard.html"
    
    if out_html.exists() and not args.overwrite:
        print(f"Reloading existing dashboard: {out_html}")
        return
        
    # Load track data
    track_path = case_dir / "expected_track.csv"
    track = pd.read_csv(track_path)
    
    # Center map on the midpoint of the expected track
    mean_lat = track["lat_deg"].mean()
    mean_lon = track["lon_deg"].mean()
    
    # Initialize folium map with custom style for bottom panel space
    # We set padding-bottom on the map div so the altitude panel fits nicely at the bottom
    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    
    # Add optional tile layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Topo",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
    ).add_to(m)
    
    # Draw expected balloon track
    track_coords = list(zip(track["lat_deg"], track["lon_deg"], strict=True))
    track_line = folium.PolyLine(
        track_coords,
        color="#58a6ff",
        weight=4,
        opacity=0.8,
        name="Expected Balloon Track",
        tooltip="K7UAZ Expected Ground Track (Maidenhead Grid Centers)",
    )
    track_line.add_to(m)
    
    # Draw Maidenhead Grid Squares from expected track points
    grid_layer = folium.FeatureGroup(name="Maidenhead Grids", show=True)
    from nexrad_picoballoon.maidenhead_grid import grid_polygon_coords
    
    seen_grids = set()
    for _, row in track.iterrows():
        grid = str(row["maidenhead_grid"])
        if grid in seen_grids:
            continue
        seen_grids.add(grid)
        
        # Get bounds
        coords = grid_polygon_coords(grid)
        # GeoJSON is [lon, lat], Folium wants [lat, lon]
        folium_coords = [[pt[1], pt[0]] for pt in coords]
        
        folium.Polygon(
            folium_coords,
            color="#30363d",
            weight=1.5,
            fill=True,
            fill_color="#21262d",
            fill_opacity=0.15,
            tooltip=f"Grid Square: {grid}",
        ).add_to(grid_layer)
    grid_layer.add_to(m)
    
    # Draw active radar sites and range rings
    radars_group = folium.FeatureGroup(name="Radar Sites & Ranges", show=True)
    radar_sites_cfg = config.get("radar_sites", {})
    
    site_colors = {
        "KEMX": "#ff7b72",  # coral/red
        "KIWA": "#79c0ff",  # skyblue
        "KFSX": "#d29922",  # gold
        "KYUX": "#bc8cff",  # purple
        "KEPZ": "#56d364",  # green
    }
    
    # Default color if not in primary
    def get_site_color(site_name: str) -> str:
        return site_colors.get(site_name, "#8b949e")
        
    for site in active_sites:
        if site not in radar_sites_cfg:
            continue
        scfg = radar_sites_cfg[site]
        lat, lon, alt_m = scfg["lat"], scfg["lon"], scfg["alt_m"]
        
        # Marker
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="darkred" if site == "KEMX" else "blue", icon="info-sign"),
            tooltip=f"Radar Site: {site}<br>Elevation: {alt_m:.0f} m",
        ).add_to(radars_group)
        
        # Range rings: 100km & 200km
        for r_km in [100, 200]:
            folium.Circle(
                location=[lat, lon],
                radius=r_km * 1000.0,
                color=get_site_color(site),
                weight=1,
                fill=False,
                opacity=0.25,
            ).add_to(radars_group)
    radars_group.add_to(m)
    
    # Collect candidate tracklets and points
    tracklets_df_list = []
    points_df_list = []
    
    for site in active_sites:
        s_path = out_dir / site / "candidate_tracklets.csv"
        p_path = out_dir / site / "tracklet_points.csv"
        
        if s_path.exists() and p_path.exists():
            sdf = pd.read_csv(s_path)
            if not sdf.empty:
                tracklets_df_list.append(sdf)
                points_df_list.append(pd.read_csv(p_path))
                
    # Draw tracklet layers on map
    tracklets_group = folium.FeatureGroup(name="Candidate Tracklets", show=True)
    
    tracklet_points_json = []
    tracklet_meta_json = []
    
    if points_df_list:
        merged_points = pd.concat(points_df_list, ignore_index=True)
        merged_summaries = pd.concat(tracklets_df_list, ignore_index=True)
        
        for _, t in merged_summaries.iterrows():
            tid = t["tracklet_id"]
            site = t["radar_site"]
            rank = int(t["tracklet_rank"])
            
            t_pts = merged_points[merged_points["tracklet_id"] == tid].sort_values("scan_time_utc")
            if t_pts.empty:
                continue
                
            pts_coords = list(zip(t_pts["cluster_lat_deg"], t_pts["cluster_lon_deg"], strict=True))
            
            # Polyline for tracklet path
            folium.PolyLine(
                pts_coords,
                color=get_site_color(site),
                weight=2.5,
                opacity=0.7,
                tooltip=f"Tracklet: {tid}<br>Radar: {site}<br>Points: {len(t_pts)}<br>Label: {t['tracklet_label']}",
            ).add_to(tracklets_group)
            
            # Markers for individual tracklet points
            for _, pt in t_pts.iterrows():
                folium.CircleMarker(
                    location=[pt["cluster_lat_deg"], pt["cluster_lon_deg"]],
                    radius=4,
                    color=get_site_color(site),
                    fill=True,
                    fill_color=get_site_color(site),
                    fill_opacity=0.8,
                    tooltip=(
                        f"Point in {tid}<br>"
                        f"Time: {pt['scan_time_utc'].split('T')[1][:8]} UTC<br>"
                        f"Alt: {pt['cluster_alt_m']:.0f}m (Expected: {pt['expected_alt_m']:.0f}m)<br>"
                        f"Mismatch: {pt['signed_vertical_m']:+.0f}m<br>"
                        f"Reflectivity: {pt['max_reflectivity_dbz']:.1f} dBZ"
                    ),
                ).add_to(tracklets_group)
                
                tracklet_points_json.append({
                    "tid": tid,
                    "site": site,
                    "t": pt["scan_time_utc"],
                    "alt": float(pt["cluster_alt_m"]),
                    "exp_alt": float(pt["expected_alt_m"]),
                    "signed_v": float(pt["signed_vertical_m"]),
                    "dbz": float(pt["max_reflectivity_dbz"]),
                    "score": float(pt["balloon_like_cluster_score"]),
                })
                
            tracklet_meta_json.append({
                "tid": tid,
                "site": site,
                "rank": rank,
                "label": t["tracklet_label"],
                "score": float(t["tracklet_score"]),
            })
            
    tracklets_group.add_to(m)
    
    # Load and draw cross-radar associations
    assoc_group = folium.FeatureGroup(name="Cross-Radar Associations", show=True)
    assoc_path = out_dir / "cross_radar_tracklet_associations.csv"
    
    associations_json = []
    
    if assoc_path.exists():
        assoc_df = pd.read_csv(assoc_path)
        if not assoc_df.empty:
            for _, r in assoc_df.iterrows():
                tid_pair = str(r["tracklet_ids"]).split(";")
                t1_id, t2_id = tid_pair[0], tid_pair[1]
                
                # Fetch points to draw connecting lines
                if points_df_list:
                    merged_points = pd.concat(points_df_list, ignore_index=True)
                    t1_pts = merged_points[merged_points["tracklet_id"] == t1_id].sort_values("scan_time_utc")
                    t2_pts = merged_points[merged_points["tracklet_id"] == t2_id].sort_values("scan_time_utc")
                    
                    if not t1_pts.empty and not t2_pts.empty:
                        # Find midpoints to draw association marker line
                        lat1, lon1 = t1_pts["cluster_lat_deg"].mean(), t1_pts["cluster_lon_deg"].mean()
                        lat2, lon2 = t2_pts["cluster_lat_deg"].mean(), t2_pts["cluster_lon_deg"].mean()
                        
                        # Dashed purple connecting line for associated tracklets
                        folium.PolyLine(
                            [[lat1, lon1], [lat2, lon2]],
                            color="#d8b4fe",
                            weight=2,
                            dash_array="5, 5",
                            tooltip=(
                                f"Association: {r['association_id']}<br>"
                                f"Tracklets: {t1_id} &amp; {t2_id}<br>"
                                f"Label: {r['association_label']}<br>"
                                f"Horiz Diff: {r['median_horizontal_difference_km']:.2f} km"
                            ),
                        ).add_to(assoc_group)
                        
                associations_json.append({
                    "aid": r["association_id"],
                    "tids": [t1_id, t2_id],
                    "label": r["association_label"],
                    "h_diff": float(r["median_horizontal_difference_km"]),
                    "v_diff": float(r["median_altitude_difference_m"]),
                })
                
    assoc_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    # -----------------------------------------------------------------------
    # Build chart data structures
    # -----------------------------------------------------------------------
    # expected telemetry curve
    exp_data = []
    for _, row in track.iterrows():
        exp_data.append({
            "t": row["time_utc"],
            "alt": float(row["alt_m"]),
        })
        
    chart_payload = {
        "expected": exp_data,
        "tracklet_points": tracklet_points_json,
        "tracklets": tracklet_meta_json,
        "associations": associations_json,
        "site_colors": site_colors,
    }
    
    chart_payload_json = json.dumps(chart_payload)
    
    # HTML and canvas chart injection code
    chart_html = f"""
    <div id="alt-chart-container" style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 290px;
        background: #0d1117;
        border-top: 2px solid #30363d;
        padding: 12px 20px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        z-index: 1000;
        box-sizing: border-box;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div style="color: #f0f6fc; font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                <span>🚀 Regional Discovery Dashboard — Altitude vs Time Profile</span>
                <span style="background: #ff7b7233; color: #ff7b72; font-size: 10px; padding: 1px 6px; border-radius: 4px; border: 1px solid #ff7b7233;">KEMX</span>
                <span style="background: #79c0ff33; color: #79c0ff; font-size: 10px; padding: 1px 6px; border-radius: 4px; border: 1px solid #79c0ff33;">KIWA</span>
            </div>
            <div style="color: #8b949e; font-size: 11px; font-style: italic;">
                Maidenhead Grid-Square Centers approximate expected altitude curve.
            </div>
        </div>
        
        <div style="display: flex; gap: 15px; height: calc(100% - 30px);">
            <!-- Canvas Plot -->
            <div style="flex-grow: 1; position: relative;">
                <canvas id="altChart" style="width: 100%; height: 210px; display: block;"></canvas>
            </div>
            
            <!-- Side Panel Details -->
            <div id="side-panel" style="
                width: 250px;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px;
                overflow-y: auto;
                color: #c9d1d9;
                font-size: 11px;
                line-height: 1.4;
            ">
                <div style="font-weight: 600; color: #58a6ff; margin-bottom: 6px; border-bottom: 1px solid #30363d; padding-bottom: 4px;">
                    Select a Candidate Tracklet
                </div>
                <div id="side-panel-content">
                    Hover over tracklet points on the chart or map to view details.
                </div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        var payload = {chart_payload_json};
        var canvas = document.getElementById('altChart');
        
        // Setup high DPI canvas
        function setupCanvas() {{
            var rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = 210 * window.devicePixelRatio;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = '210px';
        }}
        setupCanvas();
        window.addEventListener('resize', function() {{
            setupCanvas();
            draw();
        }});
        
        function toTs(isoStr) {{
            return new Date(isoStr).getTime();
        }}
        
        // Find bounds
        var allT = [], allA = [];
        payload.expected.forEach(function(d) {{ allT.push(toTs(d.t)); allA.push(d.alt); }});
        payload.tracklet_points.forEach(function(d) {{ allT.push(toTs(d.t)); allA.push(d.alt); }});
        
        var tMin = Math.min.apply(null, allT), tMax = Math.max.apply(null, allT);
        var aMin = Math.min.apply(null, allA) - 1000, aMax = Math.max.apply(null, allA) + 1000;
        
        function getX(ts) {{
            var pct = (ts - tMin) / (tMax - tMin);
            var padding = 50 * window.devicePixelRatio;
            var w = canvas.width - padding * 2;
            return padding + pct * w;
        }}
        
        function getY(alt) {{
            var pct = (alt - aMin) / (aMax - aMin);
            var padding = 20 * window.devicePixelRatio;
            var h = canvas.height - padding * 2;
            return canvas.height - padding - pct * h;
        }}
        
        function draw() {{
            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw axes
            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 1 * window.devicePixelRatio;
            ctx.beginPath();
            ctx.moveTo(getX(tMin), getY(aMin));
            ctx.lineTo(getX(tMax), getY(aMin));
            ctx.moveTo(getX(tMin), getY(aMin));
            ctx.lineTo(getX(tMin), getY(aMax));
            ctx.stroke();
            
            // Draw grid lines
            ctx.fillStyle = '#8b949e';
            ctx.font = (10 * window.devicePixelRatio) + 'px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            
            // Alt ticks
            var altStep = 2000;
            var startAlt = Math.ceil(aMin / altStep) * altStep;
            for (var a = startAlt; a <= aMax; a += altStep) {{
                var y = getY(a);
                ctx.strokeStyle = '#21262d';
                ctx.beginPath();
                ctx.moveTo(getX(tMin), y);
                ctx.lineTo(getX(tMax), y);
                ctx.stroke();
                ctx.fillText(a + 'm', getX(tMin) - 8 * window.devicePixelRatio, y);
            }}
            
            // Time ticks (every 30 mins)
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            var timeStep = 30 * 60 * 1000;
            var startTime = Math.ceil(tMin / timeStep) * timeStep;
            for (var t = startTime; t <= tMax; t += timeStep) {{
                var x = getX(t);
                var d = new Date(t);
                var timeStr = d.getUTCHours().toString().padStart(2, '0') + ':' + d.getUTCMinutes().toString().padStart(2, '0');
                ctx.strokeStyle = '#21262d';
                ctx.beginPath();
                ctx.moveTo(x, getY(aMin));
                ctx.lineTo(x, getY(aMax));
                ctx.stroke();
                ctx.fillText(timeStr, x, getY(aMin) + 8 * window.devicePixelRatio);
            }}
            
            // Draw expected altitude curve
            ctx.strokeStyle = '#58a6ff';
            ctx.lineWidth = 3 * window.devicePixelRatio;
            ctx.beginPath();
            payload.expected.forEach(function(d, idx) {{
                var x = getX(toTs(d.t));
                var y = getY(d.alt);
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            
            // Draw tracklet lines
            var grpPoints = {{}};
            payload.tracklet_points.forEach(function(pt) {{
                if (!grpPoints[pt.tid]) grpPoints[pt.tid] = [];
                grpPoints[pt.tid].push(pt);
            }});
            
            Object.keys(grpPoints).forEach(function(tid) {{
                var pts = grpPoints[tid].sort((a,b) => toTs(a.t) - toTs(b.t));
                var color = payload.site_colors[pts[0].site] || '#8b949e';
                ctx.strokeStyle = color;
                ctx.lineWidth = 2 * window.devicePixelRatio;
                ctx.beginPath();
                pts.forEach(function(pt, idx) {{
                    var x = getX(toTs(pt.t));
                    var y = getY(pt.alt);
                    if (idx === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }});
                ctx.stroke();
                
                // Draw nodes
                pts.forEach(function(pt) {{
                    var x = getX(toTs(pt.t));
                    var y = getY(pt.alt);
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(x, y, 4 * window.devicePixelRatio, 0, Math.PI * 2);
                    ctx.fill();
                }});
            }});
        }}
        
        draw();
        
        // Mouse hover interaction on Canvas
        canvas.addEventListener('mousemove', function(e) {{
            var rect = canvas.getBoundingClientRect();
            var mouseX = (e.clientX - rect.left) * window.devicePixelRatio;
            var mouseY = (e.clientY - rect.top) * window.devicePixelRatio;
            
            var nearestPt = null;
            var minDist = 20 * window.devicePixelRatio;
            
            payload.tracklet_points.forEach(function(pt) {{
                var px = getX(toTs(pt.t));
                var py = getY(pt.alt);
                var dist = Math.sqrt((px - mouseX)**2 + (py - mouseY)**2);
                if (dist < minDist) {{
                    minDist = dist;
                    nearestPt = pt;
                }}
            }});
            
            var panel = document.getElementById('side-panel-content');
            if (nearestPt) {{
                var metadata = payload.tracklets.find(t => t.tid === nearestPt.tid) || {{}};
                panel.innerHTML = `
                    <div style="font-weight:600; color:#58a6ff; font-size:12px; margin-bottom:4px;">${{nearestPt.tid}}</div>
                    <strong>Radar:</strong> ${{nearestPt.site}}<br>
                    <strong>Time:</strong> ${{nearestPt.t.split('T')[1].substring(0,8)}} UTC<br>
                    <strong>Alt:</strong> ${{nearestPt.alt.toFixed(0)}} m<br>
                    <strong>Exp Alt:</strong> ${{nearestPt.exp_alt.toFixed(0)}} m<br>
                    <strong>Mismatch:</strong> ${{nearestPt.signed_v > 0 ? '+' : ''}}${{nearestPt.signed_v.toFixed(0)}} m<br>
                    <strong>Reflectivity:</strong> ${{nearestPt.dbz.toFixed(1)}} dBZ<br>
                    <strong>Score:</strong> ${{nearestPt.score.toFixed(4)}}<br>
                    <strong>Label:</strong><br><span style="color:#79c0ff;">${{metadata.label || ''}}</span>
                `;
            }} else {{
                panel.innerHTML = 'Hover over tracklet points on the chart or map to view details.';
            }}
        }});
        
    }})();
    </script>
    """
    
    # Save the map HTML
    # We load the Folium map HTML string, inject our chart panel, and save it
    map_html = m._repr_html_()
    
    # Combine map HTML and the bottom chart injection panel
    # We modify the body tag or inject the chart absolute container at the end
    full_html = map_html.replace(
        "</body>",
        f"{chart_html}</body>"
    )
    
    # Adjust padding of folium map so it leaves space for the bottom panel
    # The default folium map style is height: 100%; width: 100%; we add bottom: 290px
    full_html = full_html.replace(
        "height:100%;",
        "height:calc(100% - 290px);"
    )
    
    with out_html.open("w", encoding="utf-8") as handle:
        handle.write(full_html)
        
    print(f"Wrote interactive regional discovery dashboard: {out_html}")


if __name__ == "__main__":
    main()
