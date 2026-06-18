#!/usr/bin/env python
# ruff: noqa: E501, E402
"""Create the strict plausible validation dashboard with default layer visibility and vertical mismatch bands."""

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

from scripts.candidate_utils import (
    load_config,
)


def main():
    parser = argparse.ArgumentParser(description="Create the plausible tracklets validation dashboard.")
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
    out_html = out_dir / "plausible_tracklet_dashboard.html"
    
    if out_html.exists() and not args.overwrite:
        print(f"Reloading existing dashboard: {out_html}")
        return
        
    # Load track data
    track_path = case_dir / "expected_track.csv"
    track = pd.read_csv(track_path)
    
    # Center map on expected track midpoint
    mean_lat = track["lat_deg"].mean()
    mean_lon = track["lon_deg"].mean()
    
    # Initialize folium map
    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    
    # Add tile layers
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
    
    # Draw Maidenhead Grid Squares
    grid_layer = folium.FeatureGroup(name="Maidenhead Grids", show=True)
    from nexrad_picoballoon.maidenhead_grid import grid_polygon_coords
    
    seen_grids = set()
    for _, row in track.iterrows():
        grid = str(row["maidenhead_grid"])
        if grid in seen_grids:
            continue
        seen_grids.add(grid)
        
        coords = grid_polygon_coords(grid)
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
    
    # Radar sites and ranges
    radars_group = folium.FeatureGroup(name="Radar Sites & Ranges", show=True)
    radar_sites_cfg = config.get("radar_sites", {})
    
    site_colors = {
        "KEMX": "#ff7b72",
        "KIWA": "#79c0ff",
        "KFSX": "#d29922",
        "KYUX": "#bc8cff",
        "KEPZ": "#56d364",
    }
    
    def get_site_color(site_name: str) -> str:
        return site_colors.get(site_name, "#8b949e")
        
    for site in active_sites:
        if site not in radar_sites_cfg:
            continue
        scfg = radar_sites_cfg[site]
        lat, lon, alt_m = scfg["lat"], scfg["lon"], scfg["alt_m"]
        
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="darkred" if site == "KEMX" else "blue", icon="info-sign"),
            tooltip=f"Radar Site: {site}<br>Elevation: {alt_m:.0f} m",
        ).add_to(radars_group)
        
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
    
    # Load quality diagnostics to classify tracklets
    diag_csv = out_dir / "tracklet_quality_diagnostics.csv"
    if not diag_csv.exists():
        raise FileNotFoundError(f"Diagnostics file {diag_csv} not found. Run filter script first.")
        
    diag_df = pd.read_csv(diag_csv)
    
    # Collect all points
    raw_points_list = []
    for site in active_sites:
        p_path = out_dir / site / "tracklet_points.csv"
        if p_path.exists():
            raw_points_list.append(pd.read_csv(p_path))
            
    all_points_df = pd.concat(raw_points_list, ignore_index=True) if raw_points_list else pd.DataFrame()
    
    # Folium Feature Groups for tracklets
    plausible_group = folium.FeatureGroup(name="Plausible Tracklets (Valid)", show=True)
    rejected_group = folium.FeatureGroup(name="Rejected Tracklets (Spaghetti/Noise)", show=False)
    
    tracklet_points_json = []
    tracklet_meta_json = []
    
    if not all_points_df.empty:
        for _, t in diag_df.iterrows():
            tid = t["tracklet_id"]
            site = t["radar_site"]
            status = t["status"]
            quality = t["quality_label"]
            spag = t["spaghetti_score"]
            reject_msg = t["reject_reason"] if pd.notna(t["reject_reason"]) else ""
            
            t_pts = all_points_df[all_points_df["tracklet_id"] == tid].sort_values("scan_time_utc")
            if t_pts.empty:
                continue
                
            pts_coords = list(zip(t_pts["cluster_lat_deg"], t_pts["cluster_lon_deg"], strict=True))
            
            # Select target group
            target_group = plausible_group if status == "plausible" else rejected_group
            
            # Polyline for tracklet
            folium.PolyLine(
                pts_coords,
                color=get_site_color(site) if status == "plausible" else "#8b949e",
                weight=3 if status == "plausible" else 1.5,
                opacity=0.85 if status == "plausible" else 0.4,
                tooltip=(
                    f"Tracklet: {tid}<br>"
                    f"Radar: {site}<br>"
                    f"Quality: {quality}<br>"
                    f"Spaghetti Score: {spag}<br>"
                    f"Status: {status.upper()}"
                    + (f"<br>Reason: {reject_msg}" if reject_msg else "")
                ),
            ).add_to(target_group)
            
            # Points marker
            for _, pt in t_pts.iterrows():
                folium.CircleMarker(
                    location=[pt["cluster_lat_deg"], pt["cluster_lon_deg"]],
                    radius=4.5 if status == "plausible" else 3.0,
                    color=get_site_color(site) if status == "plausible" else "#8b949e",
                    fill=True,
                    fill_color=get_site_color(site) if status == "plausible" else "#484f58",
                    fill_opacity=0.9 if status == "plausible" else 0.5,
                    tooltip=(
                        f"Point in {tid}<br>"
                        f"Time: {pt['scan_time_utc'].split('T')[1][:8]} UTC<br>"
                        f"Alt: {pt['cluster_alt_m']:.0f}m (Expected: {pt['expected_alt_m']:.0f}m)<br>"
                        f"Mismatch: {pt['signed_vertical_m']:+.0f}m<br>"
                        f"Reflectivity: {pt['max_reflectivity_dbz']:.1f} dBZ<br>"
                        f"Status: {status.upper()}"
                    ),
                ).add_to(target_group)
                
                tracklet_points_json.append({
                    "tid": tid,
                    "site": site,
                    "t": pt["scan_time_utc"],
                    "alt": float(pt["cluster_alt_m"]),
                    "exp_alt": float(pt["expected_alt_m"]),
                    "signed_v": float(pt["signed_vertical_m"]),
                    "dbz": float(pt["max_reflectivity_dbz"]),
                    "score": float(pt["balloon_like_cluster_score"]),
                    "status": status,
                })
                
            tracklet_meta_json.append({
                "tid": tid,
                "site": site,
                "label": quality,
                "spag": float(spag),
                "status": status,
            })
            
    plausible_group.add_to(m)
    rejected_group.add_to(m)
    
    # Load cross-radar associations
    assoc_csv = out_dir / "cross_radar_tracklet_associations.csv"
    pl_assoc_group = folium.FeatureGroup(name="Plausible Cross-Radar Associations", show=True)
    weak_assoc_group = folium.FeatureGroup(name="Weak/Rejected Associations", show=False)
    
    associations_json = []
    
    if assoc_csv.exists() and not all_points_df.empty:
        assoc_df = pd.read_csv(assoc_csv)
        if not assoc_df.empty:
            plausible_ids = set(diag_df[diag_df["status"] == "plausible"]["tracklet_id"])
            for _, r in assoc_df.iterrows():
                tids = str(r["tracklet_ids"]).split(";")
                t1_id, t2_id = tids[0], tids[1]
                label = r["association_label"]
                
                is_plausible_assoc = all(tid in plausible_ids for tid in tids)
                is_strong_or_mod = label in ["strong_cross_radar_candidate", "moderate_cross_radar_candidate"]
                
                t1_pts = all_points_df[all_points_df["tracklet_id"] == t1_id].sort_values("scan_time_utc")
                t2_pts = all_points_df[all_points_df["tracklet_id"] == t2_id].sort_values("scan_time_utc")
                
                if not t1_pts.empty and not t2_pts.empty:
                    lat1, lon1 = t1_pts["cluster_lat_deg"].mean(), t1_pts["cluster_lon_deg"].mean()
                    lat2, lon2 = t2_pts["cluster_lat_deg"].mean(), t2_pts["cluster_lon_deg"].mean()
                    
                    target_assoc_group = pl_assoc_group if (is_plausible_assoc and is_strong_or_mod) else weak_assoc_group
                    
                    folium.PolyLine(
                        [[lat1, lon1], [lat2, lon2]],
                        color="#d8b4fe" if is_plausible_assoc else "#484f58",
                        weight=2.5 if is_plausible_assoc else 1.0,
                        dash_array="6, 6" if is_plausible_assoc else "3, 6",
                        opacity=0.9 if is_plausible_assoc else 0.3,
                        tooltip=(
                            f"Association: {r['association_id']}<br>"
                            f"Tracklets: {t1_id} &amp; {t2_id}<br>"
                            f"Label: {label}<br>"
                            f"Overlap: {r['time_overlap_min']:.1f} min<br>"
                            f"Horiz Diff: {r['median_horizontal_difference_km']:.2f} km<br>"
                            f"Status: {'PLAUSIBLE' if is_plausible_assoc else 'WEAK/REJECTED'}"
                        ),
                    ).add_to(target_assoc_group)
                    
                associations_json.append({
                    "aid": r["association_id"],
                    "tids": [t1_id, t2_id],
                    "label": label,
                    "h_diff": float(r["median_horizontal_difference_km"]),
                    "v_diff": float(r["median_altitude_difference_m"]),
                    "is_plausible": is_plausible_assoc,
                    "is_strong_mod": is_strong_or_mod,
                })
                
    pl_assoc_group.add_to(m)
    weak_assoc_group.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Expected telemetry curve
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
    
    # HTML injection with Canvas and Side Panel
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
                <span>🚀 Plausible Tracklet Validation — Altitude vs Time Profile</span>
                <span style="background: #56d36433; color: #56d364; font-size: 10px; padding: 1px 6px; border-radius: 4px; border: 1px solid #56d36433;">Validated</span>
            </div>
            <div style="color: #8b949e; font-size: 11px; font-style: italic;">
                Shaded bands indicate vertical telemetry offsets: ±250m, ±500m, and ±1000m.
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
                    Select a Plausible Tracklet
                </div>
                <div id="side-panel-content">
                    Hover over tracklet points on the chart or map to view details. Plausible points are highlighted.
                </div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        var payload = {chart_payload_json};
        var canvas = document.getElementById('altChart');
        
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
        payload.tracklet_points.forEach(function(d) {{ 
            if (d.status === 'plausible') {{
                allT.push(toTs(d.t)); 
                allA.push(d.alt); 
            }}
        }});
        
        if (allT.length === 0) {{
            payload.expected.forEach(function(d) {{ allT.push(toTs(d.t)); allA.push(d.alt); }});
        }}
        
        var tMin = Math.min.apply(null, allT), tMax = Math.max.apply(null, allT);
        var aMin = Math.min.apply(null, allA) - 1200, aMax = Math.max.apply(null, allA) + 1200;
        
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
            
            // Draw vertical mismatch bands around expected balloon altitude
            var offsets = [1000, 500, 250];
            var opacities = [0.03, 0.06, 0.1];
            offsets.forEach(function(offset, oidx) {{
                ctx.fillStyle = 'rgba(88, 166, 255, ' + opacities[oidx] + ')';
                ctx.beginPath();
                // Go upper path (left to right)
                payload.expected.forEach(function(d, idx) {{
                    var x = getX(toTs(d.t));
                    var y = getY(d.alt + offset);
                    if (idx === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }});
                // Go lower path (right to left)
                for (var idx = payload.expected.length - 1; idx >= 0; idx--) {{
                    var d = payload.expected[idx];
                    var x = getX(toTs(d.t));
                    var y = getY(d.alt - offset);
                    ctx.lineTo(x, y);
                }}
                ctx.closePath();
                ctx.fill();
            }});
            
            // Draw axis lines
            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 1 * window.devicePixelRatio;
            ctx.beginPath();
            ctx.moveTo(getX(tMin), getY(aMin));
            ctx.lineTo(getX(tMax), getY(aMin));
            ctx.moveTo(getX(tMin), getY(aMin));
            ctx.lineTo(getX(tMin), getY(aMax));
            ctx.stroke();
            
            // Alt ticks
            ctx.fillStyle = '#8b949e';
            ctx.font = (10 * window.devicePixelRatio) + 'px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
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
            
            // Time ticks
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
            
            // Draw plausible tracklets only
            var grpPoints = {{}};
            payload.tracklet_points.forEach(function(pt) {{
                if (pt.status === 'plausible') {{
                    if (!grpPoints[pt.tid]) grpPoints[pt.tid] = [];
                    grpPoints[pt.tid].push(pt);
                }}
            }});
            
            Object.keys(grpPoints).forEach(function(tid) {{
                var pts = grpPoints[tid].sort((a,b) => toTs(a.t) - toTs(b.t));
                var color = payload.site_colors[pts[0].site] || '#8b949e';
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.5 * window.devicePixelRatio;
                ctx.beginPath();
                pts.forEach(function(pt, idx) {{
                    var x = getX(toTs(pt.t));
                    var y = getY(pt.alt);
                    if (idx === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }});
                ctx.stroke();
                
                pts.forEach(function(pt) {{
                    var x = getX(toTs(pt.t));
                    var y = getY(pt.alt);
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(x, y, 4.5 * window.devicePixelRatio, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // White center for visual distinctness
                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(x, y, 1.5 * window.devicePixelRatio, 0, Math.PI * 2);
                    ctx.fill();
                }});
            }});
        }}
        
        draw();
        
        // Mouse interaction on Canvas
        canvas.addEventListener('mousemove', function(e) {{
            var rect = canvas.getBoundingClientRect();
            var mouseX = (e.clientX - rect.left) * window.devicePixelRatio;
            var mouseY = (e.clientY - rect.top) * window.devicePixelRatio;
            
            var nearestPt = null;
            var minDist = 20 * window.devicePixelRatio;
            
            payload.tracklet_points.forEach(function(pt) {{
                if (pt.status === 'plausible') {{
                    var px = getX(toTs(pt.t));
                    var py = getY(pt.alt);
                    var dist = Math.sqrt((px - mouseX)**2 + (py - mouseY)**2);
                    if (dist < minDist) {{
                        minDist = dist;
                        nearestPt = pt;
                    }}
                }}
            }});
            
            var panel = document.getElementById('side-panel-content');
            if (nearestPt) {{
                var metadata = payload.tracklets.find(t => t.tid === nearestPt.tid) || {{}};
                panel.innerHTML = `
                    <div style="font-weight:600; color:#58a6ff; font-size:12px; margin-bottom:4px;">${{nearestPt.tid}}</div>
                    <strong>Radar Site:</strong> ${{nearestPt.site}}<br>
                    <strong>Time:</strong> ${{nearestPt.t.split('T')[1].substring(0,8)}} UTC<br>
                    <strong>Alt:</strong> ${{nearestPt.alt.toFixed(0)}} m<br>
                    <strong>Exp Alt:</strong> ${{nearestPt.exp_alt.toFixed(0)}} m<br>
                    <strong>Mismatch:</strong> ${{nearestPt.signed_v > 0 ? '+' : ''}}${{nearestPt.signed_v.toFixed(0)}} m<br>
                    <strong>Reflectivity:</strong> ${{nearestPt.dbz.toFixed(1)}} dBZ<br>
                    <strong>Diagnostic:</strong><br><span style="color:#56d364;">${{metadata.label.replace(/_/g, ' ')}}</span><br>
                    <strong>Spaghetti Score:</strong> ${{metadata.spag.toFixed(1)}}
                `;
            }} else {{
                panel.innerHTML = 'Hover over tracklet points on the chart or map to view details. Plausible points are highlighted.';
            }}
        }});
    }})();
    </script>
    """
    
    # Save the map HTML
    map_html = m._repr_html_()
    full_html = map_html.replace("</body>", f"{chart_html}</body>")
    full_html = full_html.replace("height:100%;", "height:calc(100% - 290px);")
    
    with out_html.open("w", encoding="utf-8") as handle:
        handle.write(full_html)
        
    print(f"Wrote validation plausible dashboard: {out_html}")
    
    # Mirror to docs/discovery
    import shutil
    docs_discovery_dir = ROOT / "docs" / "discovery"
    docs_discovery_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_html, docs_discovery_dir / "plausible_tracklet_dashboard.html")
    print("Copied plausible dashboard to docs/discovery/")


if __name__ == "__main__":
    main()
