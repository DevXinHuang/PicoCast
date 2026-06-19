#!/usr/bin/env python3
"""
make_balloon_siting_map.py — Generates balloon_siting_timeline_map.html

Creates a Leaflet.js map showing:
- All 12 balloon telemetry fixes (colored by altitude)
- All 20 radar candidate tracklets (colored by radar, sized by identification score)
- 90-minute data gap shading
- Both radar sites + range rings
- Time scrubber to highlight active tracklets per scan epoch
- Layer toggles

Usage:
    python scripts/make_balloon_siting_map.py --case-id k7uaz_20260322
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def build_html(telemetry: list, tracklets_geojson: dict, scores: list,
               radar_sites: dict, gap_start_utc: str, gap_end_utc: str) -> str:
    tel_json = json.dumps(telemetry)
    trkl_json = json.dumps(tracklets_geojson)
    scores_json = json.dumps(scores)
    radars_json = json.dumps(radar_sites)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PicoCAST — K7UAZ 3-22 Balloon Siting Timeline Map</title>
<meta name="description" content="Interactive timeline map of all balloon potential siting candidates for the K7UAZ 2026-03-22 NEXRAD case"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ── Header ───────────────────────────── */
  #header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-bottom: 1px solid #334155;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4);
  }}
  #header .badge {{
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #fff;
  }}
  #header h1 {{
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
    flex: 1;
  }}
  #header .meta {{
    font-size: 11px;
    color: #64748b;
    text-align: right;
    line-height: 1.5;
  }}

  /* ── Main layout ───────────────────────── */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Sidebar ───────────────────────────── */
  #sidebar {{
    width: 290px;
    flex-shrink: 0;
    background: #0f172a;
    border-right: 1px solid #1e293b;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 12px;
    gap: 12px;
  }}

  .panel {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
  }}
  .panel-title {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 10px;
  }}

  /* Time scrubber */
  #scrubber-time {{
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
    font-variant-numeric: tabular-nums;
    margin-bottom: 2px;
    letter-spacing: -0.02em;
  }}
  #scrubber-phase {{
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 10px;
  }}
  #scrubber {{
    width: 100%;
    accent-color: #38bdf8;
    cursor: pointer;
    margin-bottom: 8px;
  }}
  .scrubber-btns {{
    display: flex;
    gap: 6px;
    margin-top: 6px;
  }}
  .scrubber-btns button {{
    flex: 1;
    padding: 5px 0;
    border: 1px solid #334155;
    border-radius: 6px;
    background: #0f172a;
    color: #94a3b8;
    font-size: 11px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .scrubber-btns button:hover {{ background: #1e3a5f; color: #38bdf8; border-color: #38bdf8; }}
  .scrubber-btns button.active {{ background: #1e3a5f; color: #38bdf8; border-color: #38bdf8; }}

  /* Layer toggles */
  .toggle-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    cursor: pointer;
    border-radius: 5px;
    transition: background 0.1s;
  }}
  .toggle-row:hover {{ background: #0f172a; }}
  .toggle-row input[type=checkbox] {{
    width: 15px; height: 15px;
    accent-color: #38bdf8;
    cursor: pointer;
    flex-shrink: 0;
  }}
  .toggle-label {{ font-size: 12px; color: #cbd5e1; flex: 1; }}
  .toggle-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* Siting list */
  .siting-item {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
    position: relative;
    overflow: hidden;
  }}
  .siting-item::before {{
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
  }}
  .siting-item:hover {{ border-color: #38bdf8; background: #0a1628; }}
  .siting-item.highlight {{ border-color: #fbbf24; background: #1a1200; }}
  .siting-rank {{ font-size: 10px; color: #64748b; font-weight: 600; }}
  .siting-id {{ font-size: 12px; font-weight: 600; color: #f1f5f9; }}
  .siting-score-bar {{
    height: 4px;
    border-radius: 2px;
    background: #1e293b;
    margin: 5px 0 3px;
    overflow: hidden;
  }}
  .siting-score-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
  }}
  .siting-meta {{ font-size: 10px; color: #64748b; }}

  /* Altitude legend */
  .alt-legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
  }}
  .alt-gradient {{
    flex: 1;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, #3b82f6, #22d3ee, #84cc16, #f59e0b, #ef4444);
  }}
  .alt-legend-label {{ font-size: 10px; color: #64748b; }}

  /* ── Map ───────────────────────────────── */
  #map {{
    flex: 1;
    background: #0a0f1e;
  }}

  /* Leaflet overrides */
  .leaflet-popup-content-wrapper {{
    background: rgba(15, 23, 42, 0.96);
    border: 1px solid #334155;
    border-radius: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    backdrop-filter: blur(8px);
  }}
  .leaflet-popup-content {{ margin: 12px 16px; color: #e2e8f0; min-width: 200px; }}
  .leaflet-popup-tip {{ background: rgba(15, 23, 42, 0.96); }}
  .popup-header {{ font-weight: 700; font-size: 13px; margin-bottom: 8px; }}
  .popup-row {{ display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #1e293b; }}
  .popup-key {{ color: #64748b; }}
  .popup-val {{ color: #f1f5f9; font-weight: 500; }}
  .popup-tier {{
    display: inline-block;
    margin-top: 7px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  /* Score badge on markers */
  .score-badge {{
    background: rgba(15,23,42,0.9);
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 1px 4px;
    font-size: 9px;
    font-weight: 700;
    color: #38bdf8;
    white-space: nowrap;
  }}

  /* Gap info */
  .gap-notice {{
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 10px;
    color: #fca5a5;
    line-height: 1.5;
  }}
</style>
</head>
<body>

<div id="header">
  <div class="badge">PicoCAST</div>
  <h1>K7UAZ · 2026-03-22 · Balloon Siting Timeline</h1>
  <div class="meta">
    KEMX + KIWA · 20 candidates · 12 telemetry fixes<br/>
    19:18 – 22:28 UTC · 3.1 hr window
  </div>
</div>

<div id="main">
  <!-- ── SIDEBAR ────────────────────────────── -->
  <div id="sidebar">

    <!-- Time scrubber -->
    <div class="panel">
      <div class="panel-title">Time Scrubber (UTC)</div>
      <div id="scrubber-time">19:18</div>
      <div id="scrubber-phase">—</div>
      <input type="range" id="scrubber" min="0" max="100" value="0" step="1"/>
      <div class="scrubber-btns">
        <button id="btn-play" onclick="togglePlay()">▶ Play</button>
        <button id="btn-reset" onclick="resetScrubber()">↺ Reset</button>
      </div>
    </div>

    <!-- Layer toggles -->
    <div class="panel">
      <div class="panel-title">Layers</div>
      <label class="toggle-row">
        <input type="checkbox" id="tog-telemetry" checked onchange="toggleLayer('telemetry')"/>
        <div class="toggle-dot" style="background:#38bdf8;"></div>
        <span class="toggle-label">Balloon Telemetry Track</span>
      </label>
      <label class="toggle-row">
        <input type="checkbox" id="tog-kemx" checked onchange="toggleLayer('kemx')"/>
        <div class="toggle-dot" style="background:#f59e0b;"></div>
        <span class="toggle-label">KEMX Candidates (10)</span>
      </label>
      <label class="toggle-row">
        <input type="checkbox" id="tog-kiwa" checked onchange="toggleLayer('kiwa')"/>
        <div class="toggle-dot" style="background:#a78bfa;"></div>
        <span class="toggle-label">KIWA Candidates (10)</span>
      </label>
      <label class="toggle-row">
        <input type="checkbox" id="tog-gap" checked onchange="toggleLayer('gap')"/>
        <div class="toggle-dot" style="background:#ef4444;"></div>
        <span class="toggle-label">Data Gap (90 min)</span>
      </label>
      <label class="toggle-row">
        <input type="checkbox" id="tog-rings" checked onchange="toggleLayer('rings')"/>
        <div class="toggle-dot" style="background:#475569;"></div>
        <span class="toggle-label">Radar Range Rings</span>
      </label>
      <label class="toggle-row">
        <input type="checkbox" id="tog-sites" checked onchange="toggleLayer('sites')"/>
        <div class="toggle-dot" style="background:#22d3ee;"></div>
        <span class="toggle-label">Radar Sites</span>
      </label>
    </div>

    <!-- Altitude legend -->
    <div class="panel">
      <div class="panel-title">Telemetry Altitude</div>
      <div class="alt-legend">
        <span class="alt-legend-label">2 km</span>
        <div class="alt-gradient"></div>
        <span class="alt-legend-label">12 km</span>
      </div>
      <div style="font-size:10px;color:#64748b;margin-top:5px;">
        Circle size = fix index · Color = altitude
      </div>
    </div>

    <!-- Gap notice -->
    <div class="gap-notice">
      ⚠ <strong>90-min data gap</strong><br/>
      12:48 → 14:18 local (19:48 → 21:18 UTC)<br/>
      Balloon climbed from 4.3 km toward ~11 km float. No telemetry during ascent.
    </div>

    <!-- Siting score list -->
    <div class="panel" style="padding-bottom:4px;">
      <div class="panel-title">Siting Candidates (ranked)</div>
      <div id="siting-list"></div>
    </div>

  </div>

  <!-- ── MAP ──────────────────────────────── -->
  <div id="map"></div>
</div>

<script>
// ─── Embedded data ──────────────────────────────────────────────────────────
const TELEMETRY = {tel_json};
const TRACKLETS = {trkl_json};
const SCORES    = {scores_json};
const RADARS    = {radars_json};
const GAP_START = "{gap_start_utc}";
const GAP_END   = "{gap_end_utc}";

// ─── Map init ───────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [32.28, -110.4],
  zoom: 9,
  zoomControl: true,
}});

const tileLayers = {{
  'Esri WorldImagery': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{attribution: 'Esri WorldImagery', maxZoom: 19}}
  ),
  'OpenStreetMap': L.tileLayer(
    'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution: '© OpenStreetMap', maxZoom: 19}}
  ),
  'OpenTopoMap': L.tileLayer(
    'https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution: '© OpenTopoMap', maxZoom: 17}}
  ),
}};
tileLayers['Esri WorldImagery'].addTo(map);
L.control.layers(tileLayers, {{}}).addTo(map);

// ─── Layer groups ───────────────────────────────────────────────────────────
const layers = {{
  telemetry: L.layerGroup().addTo(map),
  kemx:      L.layerGroup().addTo(map),
  kiwa:      L.layerGroup().addTo(map),
  gap:       L.layerGroup().addTo(map),
  rings:     L.layerGroup().addTo(map),
  sites:     L.layerGroup().addTo(map),
}};

function toggleLayer(name) {{
  const cb = document.getElementById('tog-' + name);
  if (cb.checked) map.addLayer(layers[name]);
  else map.removeLayer(layers[name]);
}}

// ─── Utility ─────────────────────────────────────────────────────────────────
function altToColor(alt_m) {{
  const min = 2340, max = 12000;
  const t = Math.max(0, Math.min(1, (alt_m - min) / (max - min)));
  // Blue → Cyan → Green → Amber → Red
  if (t < 0.25) {{
    const s = t / 0.25;
    return `rgb(${{Math.round(59+s*(-59+34))}}, ${{Math.round(130+s*(211-130))}}, ${{Math.round(246+s*(238-246))}})`;
  }} else if (t < 0.5) {{
    const s = (t-0.25)/0.25;
    return `rgb(${{Math.round(34+s*(132-34))}}, ${{Math.round(211+s*(204-211))}}, ${{Math.round(238+s*(22-238))}})`;
  }} else if (t < 0.75) {{
    const s = (t-0.5)/0.25;
    return `rgb(${{Math.round(132+s*(245-132))}}, ${{Math.round(204+s*(158-204))}}, ${{Math.round(22+s*(11-22))}})`;
  }} else {{
    const s = (t-0.75)/0.25;
    return `rgb(${{Math.round(245+s*(239-245))}}, ${{Math.round(158+s*(68-158))}}, ${{Math.round(11+s*(68-11))}})`;
  }}
}}

function tierColor(tier) {{
  if (tier === 'high_confidence_siting')     return '#22c55e';
  if (tier === 'moderate_confidence_siting') return '#f59e0b';
  if (tier === 'low_confidence_siting')      return '#ef4444';
  return '#64748b';
}}

function tierBg(tier) {{
  if (tier === 'high_confidence_siting')     return 'rgba(34,197,94,0.15)';
  if (tier === 'moderate_confidence_siting') return 'rgba(245,158,11,0.15)';
  return 'rgba(100,116,139,0.15)';
}}

function scoreToRadius(score) {{
  return 6 + score * 14;
}}

function utcLabel(iso) {{
  return iso.replace('T','  ').replace('Z',' UTC');
}}

// ─── Telemetry track ─────────────────────────────────────────────────────────
const telCoords = TELEMETRY.map(p => [p.lat_deg, p.lon_deg]);

// Dashed connector line
L.polyline(telCoords, {{
  color: 'rgba(255,255,255,0.35)',
  weight: 1.5,
  dashArray: '5 7',
}}).addTo(layers.telemetry);

TELEMETRY.forEach((p, i) => {{
  const col = altToColor(p.alt_m);
  const r = 7 + (i / (TELEMETRY.length - 1)) * 6;

  const marker = L.circleMarker([p.lat_deg, p.lon_deg], {{
    radius: r,
    color: '#0f172a',
    weight: 1.5,
    fillColor: col,
    fillOpacity: 0.92,
  }});

  const popupHTML = `
    <div class="popup-header" style="color:${{col}}">📡 Telemetry Fix #${{i+1}}</div>
    <div class="popup-row"><span class="popup-key">Time UTC</span><span class="popup-val">${{p.time_utc.replace('T',' ').replace('Z','')}}</span></div>
    <div class="popup-row"><span class="popup-key">Time Local</span><span class="popup-val">${{p.time_local}}</span></div>
    <div class="popup-row"><span class="popup-key">Altitude</span><span class="popup-val">${{(p.alt_m/1000).toFixed(2)}} km (${{p.alt_m.toFixed(0)}} m)</span></div>
    <div class="popup-row"><span class="popup-key">Grid Square</span><span class="popup-val">${{p.maidenhead_grid}}</span></div>
    <div class="popup-row"><span class="popup-key">Position</span><span class="popup-val">${{p.lat_deg.toFixed(3)}}°N, ${{p.lon_deg.toFixed(3)}}°W</span></div>
    ${{p.speed_kmh ? `<div class="popup-row"><span class="popup-key">Speed</span><span class="popup-val">${{p.speed_kmh}} km/h</span></div>` : ''}}
    ${{p.vertical_speed_m_min ? `<div class="popup-row"><span class="popup-key">V-Speed</span><span class="popup-val">${{p.vertical_speed_m_min}} m/min</span></div>` : ''}}
    ${{p.temperature_c !== undefined ? `<div class="popup-row"><span class="popup-key">Temp</span><span class="popup-val">${{p.temperature_c}}°C</span></div>` : ''}}
  `;
  marker.bindPopup(popupHTML, {{maxWidth: 260}});
  marker.addTo(layers.telemetry);
}});

// ─── Data gap shading (rectangle spanning the gap longitude range) ─────────
// We shade the approximate lat/lon bounding box during the gap
// (approx: 32.35°N, -110.71°W → 32.31°N, -110.29°W)
const gapBounds = [[31.9, -111.2], [32.7, -110.05]];
const gapRect = L.rectangle(gapBounds, {{
  color: 'rgba(239,68,68,0.6)',
  weight: 1,
  fillColor: 'rgba(239,68,68,0.08)',
  fillOpacity: 1,
  dashArray: '4 6',
}});
gapRect.bindPopup(`
  <div class="popup-header" style="color:#ef4444">⚠ 90-Minute Data Gap</div>
  <div class="popup-row"><span class="popup-key">Gap Start</span><span class="popup-val">12:48 local / 19:48 UTC</span></div>
  <div class="popup-row"><span class="popup-key">Gap End</span><span class="popup-val">14:18 local / 21:18 UTC</span></div>
  <div style="font-size:11px;color:#94a3b8;margin-top:8px;">Balloon ascended from 4.3 km toward ~11 km float.<br/>No telemetry received during this window.</div>
`);
gapRect.addTo(layers.gap);

// Gap label
L.marker([32.6, -110.65], {{
  icon: L.divIcon({{
    className: '',
    html: '<div style="background:rgba(239,68,68,0.8);color:#fff;font-size:10px;font-weight:700;padding:3px 7px;border-radius:4px;white-space:nowrap;font-family:Inter,sans-serif;">90-MIN GAP · 19:48–21:18 UTC</div>',
    iconAnchor: [75, 0],
  }})
}}).addTo(layers.gap);

// ─── Radar sites + range rings ───────────────────────────────────────────────
const RING_KM = [50, 100, 150, 200];

Object.entries(RADARS).forEach(([site, info]) => {{
  const col = site === 'KEMX' ? '#22d3ee' : '#a78bfa';

  // Range rings
  RING_KM.forEach(km => {{
    L.circle([info.lat, info.lon], {{
      radius: km * 1000,
      color: col,
      weight: 0.5,
      opacity: 0.25,
      fillOpacity: 0,
      dashArray: '3 8',
    }}).addTo(layers.rings);
  }});

  // Site marker
  const siteIcon = L.divIcon({{
    className: '',
    html: `<div style="
      background: rgba(15,23,42,0.9);
      border: 2px solid ${{col}};
      border-radius: 50%;
      width: 28px; height: 28px;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px; font-weight: 800;
      color: ${{col}};
      font-family: Inter, sans-serif;
      box-shadow: 0 0 10px ${{col}}66;
    ">${{site.slice(1)}}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  }});

  L.marker([info.lat, info.lon], {{icon: siteIcon}})
    .bindPopup(`
      <div class="popup-header" style="color:${{col}}">${{site}} Radar Site</div>
      <div class="popup-row"><span class="popup-key">Location</span><span class="popup-val">${{info.lat.toFixed(3)}}°N, ${{info.lon.toFixed(3)}}°W</span></div>
      <div class="popup-row"><span class="popup-key">Elevation</span><span class="popup-val">${{info.alt_m.toFixed(0)}} m MSL</span></div>
      <div class="popup-row"><span class="popup-key">Candidates</span><span class="popup-val">10 tracklets</span></div>
    `)
    .addTo(layers.sites);
}});

// ─── Candidate tracklet halos ────────────────────────────────────────────────
const trackletMarkers = {{}}; // tracklet_id → marker

SCORES.forEach(s => {{
  const feat = TRACKLETS.features.find(f => f.properties.tracklet_id === s.tracklet_id);
  if (!feat) return;

  const props = feat.properties;
  const isKEMX = s.radar_site === 'KEMX';
  const radarColor = isKEMX ? '#f59e0b' : '#a78bfa';
  const cLat = props.centroid_lat;
  const cLon = props.centroid_lon;
  const radius = scoreToRadius(s.identification_score);
  const tcol = tierColor(s.identification_tier);

  // Outer halo (pulsing glow effect via CSS would need keyframes; use simple circle)
  const halo = L.circle([cLat, cLon], {{
    radius: radius * 800,
    color: radarColor,
    weight: 1,
    opacity: 0.3,
    fillColor: radarColor,
    fillOpacity: 0.06,
  }}).addTo(layers[isKEMX ? 'kemx' : 'kiwa']);

  // Score dot
  const dot = L.circleMarker([cLat, cLon], {{
    radius: radius,
    color: '#0f172a',
    weight: 1.5,
    fillColor: radarColor,
    fillOpacity: 0.85,
  }});

  const scoreBar = `<div style="height:6px;background:#1e293b;border-radius:3px;margin:4px 0;overflow:hidden;">
    <div style="width:${{(s.identification_score*100).toFixed(0)}}%;height:100%;background:${{tcol}};border-radius:3px;"></div>
  </div>`;

  const popupHTML = `
    <div class="popup-header" style="color:${{radarColor}}">${{s.tracklet_id}}</div>
    ${{scoreBar}}
    <div class="popup-row"><span class="popup-key">ID Score</span><span class="popup-val" style="color:${{tcol}};font-weight:700;">${{s.identification_score.toFixed(3)}}</span></div>
    <div class="popup-row"><span class="popup-key">Tier</span><span class="popup-val">${{s.identification_tier.replace(/_/g,' ')}}</span></div>
    <div class="popup-row"><span class="popup-key">Radar</span><span class="popup-val">${{s.radar_site}}</span></div>
    <div class="popup-row"><span class="popup-key">Points</span><span class="popup-val">${{s.n_points}}</span></div>
    <div class="popup-row"><span class="popup-key">Duration</span><span class="popup-val">${{s.duration_min.toFixed(1)}} min</span></div>
    <div class="popup-row"><span class="popup-key">Alt mismatch</span><span class="popup-val">${{s.median_alt_mismatch_m.toFixed(0)}} m</span></div>
    <div class="popup-row"><span class="popup-key">Corridor dist</span><span class="popup-val">${{s.mean_corridor_km.toFixed(1)}} km</span></div>
    <div class="popup-row"><span class="popup-key">Speed ratio</span><span class="popup-val">${{s.speed_ratio.toFixed(2)}}×</span></div>
    <div class="popup-row"><span class="popup-key">Window</span><span class="popup-val">${{props.start_time_utc.slice(11,16)}} → ${{props.end_time_utc.slice(11,16)}} UTC</span></div>
    <div style="margin-top:8px;">
      <div style="font-size:10px;color:#64748b;margin-bottom:4px;">Factor breakdown</div>
      ${{['altitude','horizontal','smoothness','speed','cross_radar','duration'].map(f => {{
        const v = s['factor_' + f];
        return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
          <span style="font-size:9px;color:#94a3b8;width:70px;">${{f}}</span>
          <div style="flex:1;height:4px;background:#1e293b;border-radius:2px;overflow:hidden;">
            <div style="width:${{(v*100).toFixed(0)}}%;height:100%;background:${{v>0.7?'#22c55e':v>0.4?'#f59e0b':'#ef4444'}};border-radius:2px;"></div>
          </div>
          <span style="font-size:9px;color:#94a3b8;width:28px;text-align:right;">${{(v*100).toFixed(0)}}%</span>
        </div>`;
      }}).join('')}}
    </div>
  `;

  dot.bindPopup(popupHTML, {{maxWidth: 280}});
  dot.addTo(layers[isKEMX ? 'kemx' : 'kiwa']);
  trackletMarkers[s.tracklet_id] = {{dot, halo, isKEMX}};

  // Draw tracklet path (faint)
  const lineCoords = feat.geometry.coordinates.map(c => [c[1], c[0]]);
  L.polyline(lineCoords, {{
    color: radarColor,
    weight: 1.5,
    opacity: 0.35,
    dashArray: '4 5',
  }}).addTo(layers[isKEMX ? 'kemx' : 'kiwa']);
}});

// ─── Sidebar siting list ─────────────────────────────────────────────────────
function buildSitingList() {{
  const el = document.getElementById('siting-list');
  el.innerHTML = '';
  SCORES.forEach((s, i) => {{
    const isKEMX = s.radar_site === 'KEMX';
    const rcol = isKEMX ? '#f59e0b' : '#a78bfa';
    const tcol = tierColor(s.identification_tier);
    const div = document.createElement('div');
    div.className = 'siting-item';
    div.id = 'siting-' + s.tracklet_id;
    div.style.borderLeftColor = rcol;
    div.style.setProperty('--radar-color', rcol);
    div.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
        <span class="siting-rank">#${{s.siting_rank}}</span>
        <span class="siting-id">${{s.tracklet_id}}</span>
        <span style="margin-left:auto;font-size:10px;color:${{tcol}};font-weight:700;">${{(s.identification_score*100).toFixed(0)}}%</span>
      </div>
      <div class="siting-score-bar">
        <div class="siting-score-fill" style="width:${{(s.identification_score*100).toFixed(0)}}%;background:${{tcol}};"></div>
      </div>
      <div class="siting-meta">${{s.radar_site}} · ${{s.duration_min.toFixed(0)}} min · ${{s.identification_tier.replace(/_/g,' ')}}</div>
    `;
    div.style.borderLeft = `3px solid ${{rcol}}`;
    div.addEventListener('click', () => flyToTracklet(s.tracklet_id));
    el.appendChild(div);
  }});
}}
buildSitingList();

function flyToTracklet(tid) {{
  const feat = TRACKLETS.features.find(f => f.properties.tracklet_id === tid);
  if (!feat) return;
  const p = feat.properties;
  map.flyTo([p.centroid_lat, p.centroid_lon], 11, {{duration: 1.2}});
  // Highlight in sidebar
  document.querySelectorAll('.siting-item').forEach(el => el.classList.remove('highlight'));
  const el = document.getElementById('siting-' + tid);
  if (el) {{ el.classList.add('highlight'); el.scrollIntoView({{behavior:'smooth',block:'nearest'}}); }}
  // Open popup
  setTimeout(() => {{
    const m = trackletMarkers[tid];
    if (m) m.dot.openPopup();
  }}, 1200);
}}

// ─── Time scrubber ────────────────────────────────────────────────────────────
const TIME_MIN_UTC = new Date('2026-03-22T19:18:00Z').getTime();
const TIME_MAX_UTC = new Date('2026-03-22T22:28:00Z').getTime();
const GAP_S = new Date('2026-03-22T19:48:00Z').getTime();
const GAP_E = new Date('2026-03-22T21:18:00Z').getTime();

const PHASES = [
  {{start: TIME_MIN_UTC, end: GAP_S, label: 'Ascent Phase · 2.3 – 4.3 km'}},
  {{start: GAP_S, end: GAP_E, label: '⚠ Data Gap · Balloon ascending to float'}},
  {{start: GAP_E, end: TIME_MAX_UTC, label: 'Float Phase · ~11.5 – 12 km'}},
];

let playInterval = null;
let isPlaying = false;

function currentTimeMs(val) {{
  return TIME_MIN_UTC + (val / 100) * (TIME_MAX_UTC - TIME_MIN_UTC);
}}

function formatUTC(ms) {{
  const d = new Date(ms);
  return d.toISOString().slice(11,16);
}}

function getPhase(ms) {{
  for (const ph of PHASES) {{
    if (ms >= ph.start && ms <= ph.end) return ph.label;
  }}
  return '';
}}

function updateScrubberDisplay(val) {{
  const ms = currentTimeMs(val);
  document.getElementById('scrubber-time').textContent = formatUTC(ms) + ' UTC';
  document.getElementById('scrubber-phase').textContent = getPhase(ms);

  // Highlight tracklets active at this time
  SCORES.forEach(s => {{
    const st = new Date(s.start_time_utc).getTime();
    const en = new Date(s.end_time_utc).getTime();
    const m = trackletMarkers[s.tracklet_id];
    if (!m) return;
    const active = ms >= st - 5*60000 && ms <= en + 5*60000;
    const isKEMX = s.radar_site === 'KEMX';
    const rcol = isKEMX ? '#f59e0b' : '#a78bfa';
    m.dot.setStyle({{
      fillOpacity: active ? 0.95 : 0.2,
      opacity: active ? 1 : 0.2,
      weight: active ? 2.5 : 1,
    }});
    m.halo.setStyle({{
      opacity: active ? 0.5 : 0.05,
      fillOpacity: active ? 0.12 : 0.01,
    }});
  }});
}}

document.getElementById('scrubber').addEventListener('input', function() {{
  updateScrubberDisplay(parseInt(this.value));
}});

function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById('btn-play').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  document.getElementById('btn-play').classList.toggle('active', isPlaying);
  if (isPlaying) {{
    playInterval = setInterval(() => {{
      const s = document.getElementById('scrubber');
      const newVal = parseInt(s.value) + 1;
      if (newVal > 100) {{ resetScrubber(); return; }}
      s.value = newVal;
      updateScrubberDisplay(newVal);
    }}, 80);
  }} else {{
    clearInterval(playInterval);
  }}
}}

function resetScrubber() {{
  clearInterval(playInterval);
  isPlaying = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  const s = document.getElementById('scrubber');
  s.value = 0;
  updateScrubberDisplay(0);
}}

// Init display
updateScrubberDisplay(0);
</script>
</body>
</html>"""


def load_telemetry(case_dir: Path) -> list:
    df = pd.read_csv(case_dir / "expected_track.csv")
    records = []
    for _, r in df.iterrows():
        records.append({
            "point_id": str(r.get("point_id", "")),
            "time_local": str(r.get("time_local", "")),
            "time_utc": str(r.get("time_utc", "")),
            "lat_deg": float(r["lat_deg"]),
            "lon_deg": float(r["lon_deg"]),
            "alt_m": float(r["alt_m"]),
            "maidenhead_grid": str(r.get("maidenhead_grid", "")),
            "speed_kmh": float(r["speed_kmh"]) if pd.notna(r.get("speed_kmh")) else None,
            "vertical_speed_m_min": float(r["vertical_speed_m_min"]) if pd.notna(r.get("vertical_speed_m_min")) else None,
            "temperature_c": float(r["temperature_c"]) if pd.notna(r.get("temperature_c")) else None,
        })
    return records


def load_scores(discovery_dir: Path) -> list:
    df = pd.read_csv(discovery_dir / "siting_scores.csv")
    return df.to_dict(orient="records")


def main(case_id: str, project_root: Path) -> None:
    case_dir = project_root / "cases" / case_id
    discovery_dir = case_dir / "outputs" / "discovery"
    maps_dir = project_root / "docs" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("Loading telemetry...")
    telemetry = load_telemetry(case_dir)
    print(f"  {len(telemetry)} telemetry fixes")

    print("Loading tracklet GeoJSON...")
    with open(discovery_dir / "siting_scores_geojson.json") as f:
        tracklets_geojson = json.load(f)

    print("Loading siting scores...")
    scores = load_scores(discovery_dir)

    radar_sites = {
        "KEMX": {"lat": 31.89365, "lon": -110.63025, "alt_m": 1621.2},
        "KIWA": {"lat": 33.28923, "lon": -111.66991, "alt_m": 434.6},
    }

    print("Building HTML...")
    html = build_html(
        telemetry=telemetry,
        tracklets_geojson=tracklets_geojson,
        scores=scores,
        radar_sites=radar_sites,
        gap_start_utc="2026-03-22T19:48:00Z",
        gap_end_utc="2026-03-22T21:18:00Z",
    )

    out = maps_dir / "balloon_siting_timeline_map.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n✓ Wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="k7uaz_20260322")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    main(args.case_id, Path(args.project_root).resolve())
