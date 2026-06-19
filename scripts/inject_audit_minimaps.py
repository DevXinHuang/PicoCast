#!/usr/bin/env python3
"""
inject_audit_minimaps.py — Injects a mini Leaflet map into every evidence-audit candidate HTML card.

For each card, we embed a 260px Leaflet map showing:
  - The specific tracklet paths for this candidate
  - The telemetry track segment that overlaps the candidate's time window
  - Both radar sites
  - A "View on Timeline Map" button

Usage:
    python scripts/inject_audit_minimaps.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TELEMETRY_RAW = [
    {"time_utc": "2026-03-22T19:18:00Z", "lat_deg": 32.27, "lon_deg": -110.96, "alt_m": 2340.0},
    {"time_utc": "2026-03-22T19:28:00Z", "lat_deg": 32.27, "lon_deg": -110.88, "alt_m": 2880.0},
    {"time_utc": "2026-03-22T19:38:00Z", "lat_deg": 32.31, "lon_deg": -110.79, "alt_m": 3660.0},
    {"time_utc": "2026-03-22T19:48:00Z", "lat_deg": 32.35, "lon_deg": -110.71, "alt_m": 4280.0},
    {"time_utc": "2026-03-22T21:18:00Z", "lat_deg": 32.31, "lon_deg": -110.29, "alt_m": 11280.0},
    {"time_utc": "2026-03-22T21:28:00Z", "lat_deg": 32.31, "lon_deg": -110.21, "alt_m": 11440.0},
    {"time_utc": "2026-03-22T21:38:00Z", "lat_deg": 32.27, "lon_deg": -110.13, "alt_m": 11740.0},
    {"time_utc": "2026-03-22T21:48:00Z", "lat_deg": 32.27, "lon_deg": -110.13, "alt_m": 11740.0},
    {"time_utc": "2026-03-22T21:58:00Z", "lat_deg": 32.27, "lon_deg": -110.04, "alt_m": 11780.0},
    {"time_utc": "2026-03-22T22:08:00Z", "lat_deg": 32.27, "lon_deg": -109.96, "alt_m": 11760.0},
    {"time_utc": "2026-03-22T22:18:00Z", "lat_deg": 32.23, "lon_deg": -109.88, "alt_m": 12000.0},
    {"time_utc": "2026-03-22T22:28:00Z", "lat_deg": 32.23, "lon_deg": -109.79, "alt_m": 11780.0},
]

RADAR_SITES = {
    "KEMX": {"lat": 31.89365, "lon": -110.63025},
    "KIWA": {"lat": 33.28923, "lon": -111.66991},
}


def load_geojson(discovery_dir: Path) -> dict:
    path = discovery_dir / "siting_scores_geojson.json"
    with open(path) as f:
        return json.load(f)


def load_scores(discovery_dir: Path) -> dict:
    """Returns dict keyed by tracklet_id."""
    import csv
    result = {}
    with open(discovery_dir / "siting_scores.csv") as f:
        for row in csv.DictReader(f):
            result[row["tracklet_id"]] = row
    return result


def extract_candidate_tracklets(card_filename: str) -> list[str]:
    """Parse tracklet IDs from the card filename / HTML content."""
    # Filename patterns: candidate_A001.html → A001, candidate_KEMX_F001.html → KEMX_F001
    stem = card_filename.replace("candidate_", "").replace(".html", "")
    return [stem]  # will be matched against association data


def load_association_tracklets(discovery_dir: Path, assoc_id: str) -> list[str]:
    """Load tracklet_ids for a given association ID like A001."""
    import csv
    path = discovery_dir / "plausible_cross_radar_associations.csv"
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["association_id"] == assoc_id:
                    return row["tracklet_ids"].split(";")
    except Exception:
        pass
    return []


def build_minimap_html(candidate_id: str, tracklet_ids: list[str],
                       geojson: dict, scores: dict) -> str:
    """Build the mini-map HTML block to inject into the card."""
    # Filter geojson features for relevant tracklets
    features = [f for f in geojson["features"]
                if f["properties"]["tracklet_id"] in tracklet_ids]

    tel_json = json.dumps(TELEMETRY_RAW)
    features_json = json.dumps(features)
    radars_json = json.dumps(RADAR_SITES)
    tid_list_json = json.dumps(tracklet_ids)

    # Compute center of all candidate points
    all_lats = [p["lat_deg"] for p in TELEMETRY_RAW]
    all_lons = [p["lon_deg"] for p in TELEMETRY_RAW]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Score info for display
    score_rows = []
    for tid in tracklet_ids:
        if tid in scores:
            s = scores[tid]
            score_rows.append(
                f'<div style="font-size:11px;margin-bottom:3px;">'
                f'<strong style="color:{"#f59e0b" if "KEMX" in tid else "#a78bfa"}">{tid}</strong> '
                f'— ID Score: <strong style="color:#22c55e">{float(s["identification_score"]):.3f}</strong> '
                f'({s["identification_tier"].replace("_", " ")})</div>'
            )
    score_html = "".join(score_rows)

    return f"""
  <!-- ── Mini Siting Map ── -->
  <div class="card" style="margin-top:1.5rem;">
    <h2 style="margin-top:0">Siting Map — {candidate_id}</h2>
    <div style="margin-bottom:10px;">{score_html}</div>
    <div id="minimap-{candidate_id}" style="height:280px;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;"></div>
    <div style="margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <a href="../maps/balloon_siting_timeline_map.html"
         target="_blank"
         style="display:inline-block;background:#1e3a5f;color:#38bdf8;border:1px solid #38bdf8;
                border-radius:6px;padding:6px 14px;font-size:12px;font-weight:600;text-decoration:none;">
        🗺 View on Full Timeline Map
      </a>
      <span style="font-size:11px;color:#94a3b8;">
        Dashed cyan = telemetry track · Colored paths = radar candidate tracklets
      </span>
    </div>
  </div>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
  (function() {{
    const TELEMETRY = {tel_json};
    const FEATURES  = {features_json};
    const RADARS    = {radars_json};

    const m = L.map('minimap-{candidate_id}', {{
      center: [{center_lat:.4f}, {center_lon:.4f}],
      zoom: 8,
      zoomControl: true,
      attributionControl: false,
    }});

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{maxZoom: 18}}).addTo(m);

    // Telemetry track
    const telCoords = TELEMETRY.map(p => [p.lat_deg, p.lon_deg]);
    L.polyline(telCoords, {{color:'#22d3ee', weight:2, dashArray:'5 6', opacity:0.7}}).addTo(m);
    TELEMETRY.forEach(p => {{
      L.circleMarker([p.lat_deg, p.lon_deg], {{
        radius: 4, color:'#0f172a', weight:1,
        fillColor: '#38bdf8', fillOpacity: 0.85,
      }}).addTo(m);
    }});

    // Candidate tracklets
    const COLORS = {{'KEMX':'#f59e0b','KIWA':'#a78bfa'}};
    FEATURES.forEach(feat => {{
      const radar = feat.properties.radar_site;
      const col = COLORS[radar] || '#94a3b8';
      const coords = feat.geometry.coordinates.map(c => [c[1], c[0]]);
      L.polyline(coords, {{color: col, weight: 2.5, opacity: 0.85}}).addTo(m);
      // Centroid dot
      L.circleMarker([feat.properties.centroid_lat, feat.properties.centroid_lon], {{
        radius: 7 + feat.properties.identification_score * 10,
        color: '#0f172a', weight: 1.5,
        fillColor: col, fillOpacity: 0.9,
      }}).bindTooltip(feat.properties.tracklet_id + ' · ' + (feat.properties.identification_score*100).toFixed(0) + '%', {{permanent: false}}).addTo(m);
    }});

    // Radar sites
    Object.entries(RADARS).forEach(([site, r]) => {{
      L.circleMarker([r.lat, r.lon], {{
        radius: 6, color: '#22d3ee', weight: 2, fillColor: '#0f172a', fillOpacity: 0.9,
      }}).bindTooltip(site).addTo(m);
    }});

    // Fit bounds
    if (FEATURES.length > 0) {{
      const allPts = FEATURES.flatMap(f => f.geometry.coordinates.map(c => [c[1], c[0]]));
      allPts.push(...telCoords);
      m.fitBounds(L.latLngBounds(allPts).pad(0.15));
    }}
  }})();
  </script>
"""


def inject_into_card(html_path: Path, minimap_html: str) -> bool:
    """Insert minimap HTML before closing </div> of .page or before </body>."""
    content = html_path.read_text(encoding="utf-8")

    # Skip if already injected
    if "minimap-" in content:
        print(f"  [skip] {html_path.name} — already has minimap")
        return False

    # Inject before the footer paragraph or before </div>\n</body>
    target = '<p style="font-size:0.8rem;color:#94a3b8;margin-top:2rem">'
    if target in content:
        content = content.replace(target, minimap_html + "\n  " + target)
        html_path.write_text(content, encoding="utf-8")
        return True

    # Fallback: inject before </body>
    content = content.replace("</body>", minimap_html + "\n</body>")
    html_path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    discovery_dir = PROJECT_ROOT / "cases" / "k7uaz_20260322" / "outputs" / "discovery"
    audit_dir = PROJECT_ROOT / "docs" / "evidence_audit"

    print("Loading GeoJSON and scores...")
    geojson = load_geojson(discovery_dir)
    scores = load_scores(discovery_dir)

    cards = sorted(audit_dir.glob("candidate_*.html"))
    print(f"Found {len(cards)} candidate cards")

    for card_path in cards:
        name = card_path.stem  # e.g. "candidate_A001"
        cid = name.replace("candidate_", "")  # e.g. "A001"

        # Determine which tracklets to show
        if cid.startswith("A"):
            # Cross-radar association → load both tracklets
            tracklet_ids = load_association_tracklets(discovery_dir, cid)
            if not tracklet_ids:
                print(f"  [warn] {card_path.name} — no tracklets found for {cid}")
                continue
        elif "_F" in cid:
            # Single-radar false-positive candidate
            # Map e.g. KEMX_F001 → first KEMX tracklet in scores
            radar = cid.split("_F")[0]
            idx = int(cid.split("_F")[1]) - 1
            radar_tracklets = [k for k in scores if k.startswith(radar + "_T")]
            if idx < len(radar_tracklets):
                tracklet_ids = [sorted(radar_tracklets)[idx]]
            else:
                tracklet_ids = radar_tracklets[:1]
        else:
            tracklet_ids = [cid]

        print(f"  {card_path.name} → tracklets: {tracklet_ids}")
        minimap_html = build_minimap_html(cid, tracklet_ids, geojson, scores)
        injected = inject_into_card(card_path, minimap_html)
        if injected:
            print(f"    ✓ Injected mini-map")

    print("\n✓ All evidence audit cards updated.")


if __name__ == "__main__":
    main()
