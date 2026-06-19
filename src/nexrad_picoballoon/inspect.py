"""Candidate gate inspection framework.

Provides functions to load a specific candidate by T-label, compute gate
footprint polygons for plan-view display, and build folium maps with full
geo-overlay support (satellite / OSM basemaps, layer toggles, annotation).

Typical usage::

    from nexrad_picoballoon.inspect import load_candidate, make_gate_map

    row, gates = load_candidate(case_dir, "KEMX_T015")
    m = make_gate_map(row, gates, config)
    display(m)          # in a Jupyter notebook
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import branca.colormap as bcm
import folium
import folium.plugins as fp
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_DR_KM = 0.25          # NEXRAD super-res range resolution
GATE_DAZ_DEG = 1.0         # nominal beam width used for footprint polygon
EARTH_R_KM = 6371.0

# Field display metadata: (display_name, unit, vmin, vmax, palette)
FIELD_META: dict[str, tuple[str, str, float, float, str]] = {
    "reflectivity_dbz":   ("Reflectivity",    "dBZ",  -10,  25, "RdYlBu_r"),
    "velocity_ms":        ("Radial Velocity", "m/s",  -15,  15, "RdBu_r"),
    "spectrum_width_ms":  ("Spectrum Width",  "m/s",    0,   8, "YlOrRd"),
    "rhohv":              ("ρHV",             "",     0.4, 1.0, "RdYlGn"),
    "zdr_db":             ("ZDR",             "dB",    -2,   4, "Spectral"),
    "phidp_deg":          ("PhiDP",           "°",      0, 180, "twilight"),
}

BASEMAPS = {
    "Esri Satellite":  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap":   "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "OpenTopoMap":     "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    "CartoDB Dark":    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
}

BASEMAP_ATTR = {
    "Esri Satellite":  "Esri WorldImagery",
    "OpenStreetMap":   "© OpenStreetMap contributors",
    "OpenTopoMap":     "© OpenTopoMap contributors",
    "CartoDB Dark":    "© CARTO",
}


# ---------------------------------------------------------------------------
# T-label resolution
# ---------------------------------------------------------------------------

def build_tlabel_index(candidates_csv: Path) -> pd.DataFrame:
    """Return candidates sorted chronologically with T-labels assigned.

    The index column ``t_label`` uses the format ``<SITE>_T<NNN>`` where NNN
    is the zero-padded chronological position among all scan times that
    produced at least one candidate for that site.
    """
    df = pd.read_csv(candidates_csv)
    df["scan_time_utc"] = pd.to_datetime(df["scan_time_utc"], utc=True)

    rows = []
    for site, grp in df.groupby("radar_site"):
        times = sorted(grp["scan_time_utc"].unique())
        time_to_idx = {t: i + 1 for i, t in enumerate(times)}
        grp = grp.copy()
        grp["t_label"] = grp["scan_time_utc"].map(
            lambda t: f"{site}_T{time_to_idx[t]:03d}"
        )
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


def _lookup_expected_position(case_dir: Path, scan_time_utc: str) -> tuple[float, float, float]:
    """Look up expected lat, lon, alt from expected_track.csv for the closest time."""
    track_path = case_dir / "expected_track.csv"
    if not track_path.exists():
        return 0.0, 0.0, 0.0
    track_df = pd.read_csv(track_path)
    track_df["time_utc"] = pd.to_datetime(track_df["time_utc"], utc=True)
    scan_dt = pd.to_datetime(scan_time_utc, utc=True)
    idx = (track_df["time_utc"] - scan_dt).abs().idxmin()
    row = track_df.loc[idx]
    return float(row["lat_deg"]), float(row["lon_deg"]), float(row["alt_m"])


def load_candidate(
    case_dir: Path | str,
    candidate_id: str,
    gate_radius_km: float = 12.0,
    point_index: int = 0,
    top_n_clusters: int = 1,
) -> tuple[pd.Series, pd.DataFrame]:
    """Load a candidate (either single detection T-label or siting tracklet) and return (row, nearby_gates).

    Parameters
    ----------
    case_dir:
        Path to the case directory (e.g. ``cases/k7uaz_20260322``).
    candidate_id:
        T-label string like ``"KEMX_T015"`` or siting tracklet like ``"KEMX_T004"``.
    gate_radius_km:
        Spatial radius around the candidate to pull from the gate cache.
    point_index:
        If candidate_id is a siting tracklet, which point (0 to n-1) to load.
    top_n_clusters:
        Legacy parameter for single detection search compatibility.

    Returns
    -------
    row:
        pandas Series with candidate metadata.
    gates:
        DataFrame of all gate cache rows within ``gate_radius_km`` of the
        candidate centre for that scan, plus the ``is_candidate`` boolean
        column marking which gates belong to this cluster.
    """
    case_dir = Path(case_dir)
    site = candidate_id.split("_")[0]           # e.g. "KEMX"

    # Try loading as a tracklet first
    tracklet_pts_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"
    if tracklet_pts_path.exists():
        pts_df = pd.read_csv(tracklet_pts_path)
        tracklet_pts = pts_df[pts_df["tracklet_id"] == candidate_id]
        if not tracklet_pts.empty:
            if point_index < 0 or point_index >= len(tracklet_pts):
                raise ValueError(
                    f"Point index {point_index} out of range for tracklet '{candidate_id}' "
                    f"(contains {len(tracklet_pts)} points)."
                )
            # Pick the specified point from the tracklet, sort chronologically
            tracklet_pts = tracklet_pts.sort_values("scan_time_utc")
            pt_row = tracklet_pts.iloc[point_index].copy()
            
            # Lookup expected position from telemetry track file
            scan_time_utc = pt_row["scan_time_utc"]
            exp_lat, exp_lon, exp_alt = _lookup_expected_position(case_dir, scan_time_utc)

            # Map columns to match candidate scores schema
            row = pd.Series({
                "radar_site": pt_row["radar_site"],
                "scan_time_utc": scan_time_utc,
                "candidate_lat_deg": pt_row["cluster_lat_deg"],
                "candidate_lon_deg": pt_row["cluster_lon_deg"],
                "candidate_alt_m": pt_row["cluster_alt_m"],
                "expected_lat_deg": exp_lat,
                "expected_lon_deg": exp_lon,
                "expected_alt_m": exp_alt,
                "candidate_score": pt_row["balloon_like_cluster_score"],
                "horizontal_distance_km": pt_row["distance_to_track_corridor_km"],
                "vertical_distance_m": pt_row["signed_vertical_m"],
                "n_gates": pt_row["n_gates"],
                "mean_reflectivity_dbz": pt_row["mean_reflectivity_dbz"],
                "t_label": f"{candidate_id}_P{point_index+1}",
                "candidate_label": pt_row["notes"],
                "cluster_id": pt_row["cluster_id"],
            })
            
            ts_str = pd.Timestamp(scan_time_utc).strftime("%Y%m%d_%H%M%S")
            gate_file = case_dir / "cache" / "gates" / site / f"{site}_{ts_str}.parquet"
            if not gate_file.exists():
                raise FileNotFoundError(
                    f"Gate cache not found: {gate_file}\n"
                    "Run build_gate_feature_cache.py first."
                )
            gates_all = pd.read_parquet(gate_file)
            
            cand_lat = row["candidate_lat_deg"]
            cand_lon = row["candidate_lon_deg"]
            dlat_km = gate_radius_km / 111.0
            dlon_km = gate_radius_km / (111.0 * math.cos(math.radians(cand_lat)))
            nearby = gates_all[
                (gates_all["gate_lat_deg"].between(cand_lat - dlat_km, cand_lat + dlat_km))
                & (gates_all["gate_lon_deg"].between(cand_lon - dlon_km, cand_lon + dlon_km))
            ].copy()
            
            cand_alt = row["candidate_alt_m"]
            nearby["is_candidate"] = (
                _haversine_km(
                    nearby["gate_lat_deg"].to_numpy(),
                    nearby["gate_lon_deg"].to_numpy(),
                    cand_lat,
                    cand_lon,
                ) < 1.5
            ) & (
                (nearby["gate_alt_m"] - cand_alt).abs() < 600
            )
            return row, nearby

    # Otherwise fall back to single-scan candidates
    candidates_csv = case_dir / "outputs" / "candidates" / site / "candidate_scores.csv"
    if not candidates_csv.exists():
        raise FileNotFoundError(f"No candidate scores found: {candidates_csv}")

    index = build_tlabel_index(candidates_csv)
    matches = index[index["t_label"] == candidate_id]
    if matches.empty:
        valid = sorted(index["t_label"].unique())
        raise ValueError(
            f"Candidate '{candidate_id}' not found. "
            f"Valid T-labels: {valid}"
        )

    # Pick the top-scoring cluster at this time slot
    row = matches.sort_values("candidate_score", ascending=False).iloc[0]
    scan_time_utc = row["scan_time_utc"]

    # Resolve the gate cache file for this scan
    ts_str = pd.Timestamp(scan_time_utc).strftime("%Y%m%d_%H%M%S")
    gate_file = case_dir / "cache" / "gates" / site / f"{site}_{ts_str}.parquet"
    if not gate_file.exists():
        raise FileNotFoundError(
            f"Gate cache not found: {gate_file}\n"
            "Run build_gate_feature_cache.py first."
        )

    gates_all = pd.read_parquet(gate_file)

    # Spatial filter around candidate centre
    cand_lat = row["candidate_lat_deg"]
    cand_lon = row["candidate_lon_deg"]
    dlat_km = gate_radius_km / 111.0
    dlon_km = gate_radius_km / (111.0 * math.cos(math.radians(cand_lat)))
    nearby = gates_all[
        (gates_all["gate_lat_deg"].between(cand_lat - dlat_km, cand_lat + dlat_km))
        & (gates_all["gate_lon_deg"].between(cand_lon - dlon_km, cand_lon + dlon_km))
    ].copy()

    # Mark candidate gates using cluster_id from the candidate row
    cluster_id = row.get("cluster_id", "")
    cand_alt = row["candidate_alt_m"]
    nearby["is_candidate"] = (
        _haversine_km(
            nearby["gate_lat_deg"].to_numpy(),
            nearby["gate_lon_deg"].to_numpy(),
            cand_lat,
            cand_lon,
        ) < 1.5
    ) & (
        (nearby["gate_alt_m"] - cand_alt).abs() < 600
    )

    return row, nearby


# ---------------------------------------------------------------------------
# Gate polygon geometry
# ---------------------------------------------------------------------------

def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorised haversine distance in km."""
    R = EARTH_R_KM
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(math.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _offset_latlon(
    lat: float, lon: float, bearing_deg: float, dist_km: float
) -> tuple[float, float]:
    """Return lat/lon after moving dist_km along bearing_deg from (lat, lon)."""
    R = EARTH_R_KM
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    b_r = math.radians(bearing_deg)
    d = dist_km / R

    lat2 = math.asin(
        math.sin(lat_r) * math.cos(d)
        + math.cos(lat_r) * math.sin(d) * math.cos(b_r)
    )
    lon2 = lon_r + math.atan2(
        math.sin(b_r) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def gate_polygon_latlon(
    radar_lat: float,
    radar_lon: float,
    azimuth_deg: float,
    range_km: float,
    dr_km: float = GATE_DR_KM,
    daz_deg: float = GATE_DAZ_DEG,
) -> list[tuple[float, float]]:
    """Return the 4-corner polygon (lat, lon) for one radar gate footprint.

    The polygon is an arc-sector approximated as a quadrilateral using the
    bearing/distance method.  Corners go: near-left → near-right →
    far-right → far-left (closing ring).
    """
    r_near = range_km - dr_km / 2.0
    r_far  = range_km + dr_km / 2.0
    az_l   = azimuth_deg - daz_deg / 2.0
    az_r   = azimuth_deg + daz_deg / 2.0

    corners = [
        _offset_latlon(radar_lat, radar_lon, az_l, r_near),
        _offset_latlon(radar_lat, radar_lon, az_r, r_near),
        _offset_latlon(radar_lat, radar_lon, az_r, r_far),
        _offset_latlon(radar_lat, radar_lon, az_l, r_far),
    ]
    return corners  # list of (lat, lon) tuples


def compute_gate_polygons(
    gates: pd.DataFrame,
    radar_lat: float,
    radar_lon: float,
    daz_deg: float = GATE_DAZ_DEG,
) -> list[list[tuple[float, float]]]:
    """Batch-compute gate polygons for every row in gates."""
    return [
        gate_polygon_latlon(
            radar_lat, radar_lon,
            row["azimuth_deg"], row["range_km"],
            daz_deg=daz_deg,
        )
        for _, row in gates.iterrows()
    ]


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

def _field_colormap(field: str, steps: int = 10) -> bcm.LinearColormap:
    """Return a branca LinearColormap for the given field."""
    name, unit, vmin, vmax, palette = FIELD_META.get(
        field,
        (field, "", 0, 1, "viridis"),
    )
    # branca palette names match matplotlib
    cmap = bcm.linear._schemes.get(palette)
    if cmap is None:
        import matplotlib
        import matplotlib.colors as mcolors
        if hasattr(matplotlib, "colormaps"):
            c = matplotlib.colormaps.get_cmap(palette)
        else:
            import matplotlib.cm as mcm
            c = mcm.get_cmap(palette)
        colors = [mcolors.to_hex(c(i / (steps - 1))) for i in range(steps)]
        return bcm.LinearColormap(colors, vmin=vmin, vmax=vmax, caption=f"{name} ({unit})")
    return bcm.linear.__dict__[palette].scale(vmin, vmax).to_step(steps)


def _rgba_hex(colormap: bcm.LinearColormap, value: float | None) -> str:
    """Map a value to a hex colour string; grey if NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#888888"
    return colormap(value)


# ---------------------------------------------------------------------------
# Range-ring utility
# ---------------------------------------------------------------------------

def _range_ring(lat: float, lon: float, radius_km: float, n_pts: int = 180) -> list[list[float]]:
    """Return a list of [lat, lon] for a range ring."""
    return [
        list(_offset_latlon(lat, lon, az, radius_km))
        for az in np.linspace(0, 360, n_pts, endpoint=False)
    ]


def _maidenhead_box(grid_square: str) -> list[list[float]] | None:
    """Return the 4 corners of a Maidenhead grid square as [[lat,lon],...]."""
    try:
        # 4-char grid: field (2) + square (2)
        g = grid_square.upper()
        lon0 = (ord(g[0]) - ord('A')) * 20 - 180
        lat0 = (ord(g[1]) - ord('A')) * 10 - 90
        lon0 += (int(g[2])) * 2
        lat0 += (int(g[3])) * 1
        # corners
        return [
            [lat0,       lon0],
            [lat0 + 1,   lon0],
            [lat0 + 1,   lon0 + 2],
            [lat0,       lon0 + 2],
            [lat0,       lon0],
        ]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main map builder
# ---------------------------------------------------------------------------

def make_gate_map(
    row: pd.Series,
    gates: pd.DataFrame,
    config: dict,
    *,
    field: str = "reflectivity_dbz",
    basemap: str = "Esri Satellite",
    show_range_rings: bool = True,
    show_grid_square: bool = True,
    show_nearby_gates: bool = True,
    show_candidate_gates: bool = True,
    show_distance_line: bool = True,
    show_radar_site: bool = True,
    daz_deg: float = GATE_DAZ_DEG,
    gate_opacity: float = 0.75,
    zoom_start: int = 11,
    annotations: list[dict] | None = None,
) -> folium.Map:
    """Build a folium map overlaying radar gates on geo-imagery.

    Parameters
    ----------
    row:
        Candidate row (from load_candidate).
    gates:
        Gate DataFrame (from load_candidate), must have ``is_candidate`` col.
    config:
        Parsed case config.yaml as a dict.
    field:
        Which gate field to colour by (see FIELD_META).
    basemap:
        Starting basemap name (key in BASEMAPS).
    show_*:
        Layer visibility defaults.
    daz_deg:
        Azimuthal width for gate polygons.
    gate_opacity:
        Polygon fill opacity (0–1).
    zoom_start:
        Initial Leaflet zoom level.
    annotations:
        Optional list of dicts with keys ``lat``, ``lon``, ``label``,
        ``color`` (default red) for user-defined annotation markers.

    Returns
    -------
    folium.Map ready to display in a Jupyter notebook cell.
    """
    radar_sites = config.get("radar_sites", {})
    site = str(row.get("radar_site", "KEMX"))
    site_lat = radar_sites.get(site, {}).get("lat", 31.89365)
    site_lon = radar_sites.get(site, {}).get("lon", -110.63025)

    cand_lat  = float(row["candidate_lat_deg"])
    cand_lon  = float(row["candidate_lon_deg"])
    cand_alt  = float(row["candidate_alt_m"])
    exp_lat   = float(row["expected_lat_deg"])
    exp_lon   = float(row["expected_lon_deg"])
    exp_alt   = float(row["expected_alt_m"])
    score     = float(row["candidate_score"])
    h_dist    = float(row["horizontal_distance_km"])
    v_dist    = float(row.get("vertical_distance_m", row.get("signed_vertical_m", 0.0)))
    scan_time = str(row["scan_time_utc"])

    # ── Map object with chosen basemap ───────────────────────────────────────
    bm_url  = BASEMAPS.get(basemap, BASEMAPS["Esri Satellite"])
    bm_attr = BASEMAP_ATTR.get(basemap, "")
    m = folium.Map(
        location=[cand_lat, cand_lon],
        zoom_start=zoom_start,
        tiles=bm_url,
        attr=bm_attr,
        max_zoom=18,
    )

    # ── Additional basemap tiles (switchable) ────────────────────────────────
    for name, url in BASEMAPS.items():
        if name == basemap:
            continue
        folium.TileLayer(
            tiles=url,
            attr=BASEMAP_ATTR.get(name, ""),
            name=name,
            show=False,
        ).add_to(m)

    # ── Colour map for the chosen field ──────────────────────────────────────
    cmap = _field_colormap(field)
    field_label, field_unit, vmin, vmax, _ = FIELD_META.get(
        field, (field, "", 0, 1, "viridis")
    )

    # ── Gate layers ─────────────────────────────────────────────────────────
    nearby_gates = gates[~gates["is_candidate"]]
    cand_gates   = gates[gates["is_candidate"]]

    if show_nearby_gates and not nearby_gates.empty:
        fg_nearby = folium.FeatureGroup(name="Nearby Gates", show=show_nearby_gates)
        _add_gate_polygons(
            fg_nearby, nearby_gates, radar_lat=site_lat, radar_lon=site_lon,
            field=field, cmap=cmap, vmin=vmin, vmax=vmax,
            opacity=gate_opacity * 0.65, daz_deg=daz_deg, weight=0,
            fill_color_fallback="#aaaaaa",
        )
        fg_nearby.add_to(m)

    if show_candidate_gates and not cand_gates.empty:
        fg_cand = folium.FeatureGroup(name="Candidate Gates", show=show_candidate_gates)
        _add_gate_polygons(
            fg_cand, cand_gates, radar_lat=site_lat, radar_lon=site_lon,
            field=field, cmap=cmap, vmin=vmin, vmax=vmax,
            opacity=gate_opacity, daz_deg=daz_deg, weight=1.5,
            edge_color="#ff3300",
        )
        fg_cand.add_to(m)

    # ── Range rings ─────────────────────────────────────────────────────────
    if show_range_rings:
        fg_rings = folium.FeatureGroup(name="Range Rings", show=True)
        for r_km in [25, 50, 75, 100]:
            pts = _range_ring(site_lat, site_lon, r_km)
            folium.PolyLine(
                pts, color="#ffffff", weight=0.8, opacity=0.4,
                tooltip=f"{r_km} km",
            ).add_to(fg_rings)
        fg_rings.add_to(m)

    # ── Radar site marker ────────────────────────────────────────────────────
    if show_radar_site:
        fg_site = folium.FeatureGroup(name=f"{site} Radar Site", show=True)
        folium.Marker(
            location=[site_lat, site_lon],
            icon=folium.Icon(icon="tower-broadcast", prefix="fa", color="darkblue"),
            tooltip=f"{site} radar  ({site_lat:.4f}, {site_lon:.4f})",
        ).add_to(fg_site)
        fg_site.add_to(m)

    # ── Expected balloon position ────────────────────────────────────────────
    fg_expected = folium.FeatureGroup(name="Expected Balloon Position", show=True)
    folium.Marker(
        location=[exp_lat, exp_lon],
        icon=folium.DivIcon(
            html="""<div style="
                font-size:22px; color:#00ccff; font-weight:bold;
                text-shadow:1px 1px 3px #000;
                line-height:1;">✕</div>""",
            icon_size=(22, 22),
            icon_anchor=(11, 11),
        ),
        tooltip=f"Expected  {exp_lat:.4f}°, {exp_lon:.4f}°  alt {exp_alt:.0f} m",
    ).add_to(fg_expected)
    fg_expected.add_to(m)

    # ── Candidate centre ─────────────────────────────────────────────────────
    fg_cand_pt = folium.FeatureGroup(name="Candidate Centre", show=True)
    folium.CircleMarker(
        location=[cand_lat, cand_lon],
        radius=7,
        color="#ff3300",
        fill=True,
        fill_color="#ff3300",
        fill_opacity=0.9,
        tooltip=(
            f"<b>Candidate</b><br>"
            f"Score: {score:.3f}<br>"
            f"Alt: {cand_alt:.0f} m<br>"
            f"Scan: {scan_time}"
        ),
        popup=folium.Popup(
            _candidate_popup_html(row, cand_gates),
            max_width=320,
        ),
    ).add_to(fg_cand_pt)
    fg_cand_pt.add_to(m)

    # ── Distance line ────────────────────────────────────────────────────────
    if show_distance_line:
        fg_line = folium.FeatureGroup(name="Distance Line", show=True)
        folium.PolyLine(
            [[exp_lat, exp_lon], [cand_lat, cand_lon]],
            color="#ff3300", weight=1.5, opacity=0.8, dash_array="6 4",
            tooltip=f"H-dist: {h_dist:.2f} km  V-dist: {v_dist:+.0f} m",
        ).add_to(fg_line)
        fg_line.add_to(m)

    # ── Maidenhead grid square ───────────────────────────────────────────────
    if show_grid_square:
        grid_sq = _infer_grid_square(exp_lat, exp_lon)
        if grid_sq:
            corners = _maidenhead_box(grid_sq)
            if corners:
                fg_grid = folium.FeatureGroup(name="Maidenhead Grid Square", show=True)
                folium.Polygon(
                    locations=corners,
                    color="#00ccff",
                    weight=1.5,
                    fill=True,
                    fill_color="#00ccff",
                    fill_opacity=0.06,
                    tooltip=f"Grid square {grid_sq}",
                ).add_to(fg_grid)
                fg_grid.add_to(m)

    # ── User annotations ─────────────────────────────────────────────────────
    if annotations:
        fg_ann = folium.FeatureGroup(name="Annotations", show=True)
        for ann in annotations:
            color = ann.get("color", "red")
            label = ann.get("label", "")
            folium.CircleMarker(
                location=[ann["lat"], ann["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=label,
                popup=folium.Popup(label, max_width=200),
            ).add_to(fg_ann)
            if label:
                folium.Marker(
                    location=[ann["lat"], ann["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:11px;color:{color};'
                             f'font-weight:bold;white-space:nowrap;'
                             f'text-shadow:1px 1px 2px #000">{label}</div>',
                        icon_size=(120, 20),
                        icon_anchor=(0, 10),
                    ),
                ).add_to(fg_ann)
        fg_ann.add_to(m)

    # ── Draw / measure controls ──────────────────────────────────────────────
    fp.Draw(
        export=False,
        draw_options={
            "polyline": True,
            "polygon": False,
            "rectangle": True,
            "circle": True,
            "marker": True,
            "circlemarker": False,
        },
    ).add_to(m)
    fp.MeasureControl(position="bottomleft").add_to(m)

    # ── Colour bar legend ────────────────────────────────────────────────────
    cmap.caption = f"{field_label} ({field_unit})" if field_unit else field_label
    cmap.add_to(m)

    # ── Info box (top-left panel) ────────────────────────────────────────────
    info_html = _info_box_html(row, site)
    m.get_root().html.add_child(folium.Element(info_html))

    # ── Layer control ────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    return m


# ---------------------------------------------------------------------------
# Score breakdown chart
# ---------------------------------------------------------------------------

def score_breakdown_chart(row: pd.Series):
    """Return a matplotlib Figure showing sub-score contributions."""
    import matplotlib.pyplot as plt

    score_cols = [
        ("distance_score",           "Distance",      "#4fc3f7"),
        ("altitude_score",           "Altitude",      "#81c784"),
        ("reflectivity_score",       "Reflectivity",  "#ffb74d"),
        ("compactness_score",        "Compactness",   "#ce93d8"),
        ("isolation_score",          "Isolation",     "#f48fb1"),
        ("temporal_continuity_score","Temporal Cont.","#80cbc4"),
    ]

    labels = [s[1] for s in score_cols]
    values = [float(row.get(s[0], 0.0) or 0.0) for s in score_cols]
    colors = [s[2] for s in score_cols]
    total  = float(row.get("candidate_score", 0.0) or 0.0)

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    ax.axvline(total, color="#ff5252", linewidth=1.5, linestyle="--",
               label=f"Total score: {total:.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score component value", fontsize=9)
    ax.set_title(
        f"Score breakdown — {row.get('t_label', '')} ({row.get('radar_site','')})",
        fontsize=10, fontweight="bold",
    )
    for bar, val in zip(bars, values):
        if val > 0.05:
            ax.text(val - 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="right", fontsize=8, color="#111")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_gate_polygons(
    fg: folium.FeatureGroup,
    gates: pd.DataFrame,
    *,
    radar_lat: float,
    radar_lon: float,
    field: str,
    cmap,
    vmin: float,
    vmax: float,
    opacity: float,
    daz_deg: float,
    weight: float,
    edge_color: str = "#555555",
    fill_color_fallback: str = "#888888",
) -> None:
    """Add coloured gate polygons to a FeatureGroup."""
    for _, g in gates.iterrows():
        val = g.get(field, None)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            clamped = max(vmin, min(vmax, float(val)))
            fill = cmap(clamped)
            if fill and len(fill) == 9:
                fill = fill[:7]
        else:
            fill = fill_color_fallback

        corners = gate_polygon_latlon(
            radar_lat, radar_lon,
            float(g["azimuth_deg"]), float(g["range_km"]),
            daz_deg=daz_deg,
        )
        # folium Polygon wants [[lat,lon],...]
        folium.Polygon(
            locations=corners,
            color=edge_color if weight > 0 else fill,
            weight=weight,
            fill=True,
            fill_color=fill,
            fill_opacity=opacity,
            tooltip=(
                f"Az {g['azimuth_deg']:.1f}°  R {g['range_km']:.2f} km<br>"
                f"Alt {g.get('gate_alt_m', '?'):.0f} m<br>"
                f"{field}: {val if val is not None else 'NaN'}"
            ),
        ).add_to(fg)


def _candidate_popup_html(row: pd.Series, cand_gates: pd.DataFrame) -> str:
    alt   = float(row.get("candidate_alt_m", 0))
    score = float(row.get("candidate_score", 0))
    n     = int(row.get("n_gates", len(cand_gates)))
    hdist = float(row.get("horizontal_distance_km", 0))
    vdist = float(row.get("vertical_distance_m", row.get("signed_vertical_m", 0)))
    refl  = float(row.get("mean_reflectivity_dbz", float("nan")))
    label = str(row.get("t_label", ""))

    refl_str = f"{refl:.1f} dBZ" if not math.isnan(refl) else "N/A"
    return (
        f"<b style='font-size:13px'>{label}</b><br>"
        f"Score: <b>{score:.3f}</b><br>"
        f"Alt: {alt:.0f} m &nbsp;|&nbsp; Gates: {n}<br>"
        f"H-dist: {hdist:.2f} km &nbsp;|&nbsp; V-dist: {vdist:+.0f} m<br>"
        f"Mean Z: {refl_str}<br>"
        f"Scan: {row.get('scan_time_utc','')}"
    )


def _info_box_html(row: pd.Series, site: str) -> str:
    score  = float(row.get("candidate_score", 0))
    hdist  = float(row.get("horizontal_distance_km", 0))
    vdist  = float(row.get("vertical_distance_m", row.get("signed_vertical_m", 0)))
    refl   = float(row.get("mean_reflectivity_dbz", float("nan")))
    refl_s = f"{refl:.1f}" if not math.isnan(refl) else "N/A"
    label  = str(row.get("t_label", ""))
    clabel = str(row.get("candidate_label", ""))
    match  = "strong altitude match" if abs(vdist) < 600 else "altitude mismatch"

    return f"""
    <style>
    .picocast-info {{
        position: absolute;
        top: 10px; left: 50px;
        background: rgba(20,26,38,0.88);
        color: #e8ecf4;
        padding: 10px 14px;
        border-radius: 8px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 12px;
        z-index: 999;
        max-width: 320px;
        border: 1px solid rgba(255,255,255,0.15);
    }}
    .picocast-info b {{ color: #fff; }}
    .picocast-info .tag {{
        display: inline-block;
        background: #1e88e5;
        color: #fff;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 11px;
        margin-bottom: 4px;
    }}
    </style>
    <div class="picocast-info">
      <div class="tag">{clabel}</div>
      <b style="font-size:14px">{label}</b><br>
      H-dist {hdist:.2f} km &nbsp;·&nbsp; V-dist {vdist:+.0f} m &nbsp;·&nbsp; max dBZ {refl_s}<br>
      Alt mismatch: {vdist:+.0f} m &nbsp;·&nbsp; <em>{match}</em><br>
      <span style="color:#aaa;font-size:10px">
        Score {score:.3f} &nbsp;·&nbsp; {site}
      </span>
    </div>
    """


def _infer_grid_square(lat: float, lon: float) -> str | None:
    """Approximate the 4-char Maidenhead grid square for a lat/lon."""
    try:
        lon_adj = lon + 180.0
        lat_adj = lat + 90.0
        f1 = chr(ord('A') + int(lon_adj / 20))
        f2 = chr(ord('A') + int(lat_adj / 10))
        s1 = str(int((lon_adj % 20) / 2))
        s2 = str(int(lat_adj % 10))
        return f"{f1}{f2}{s1}{s2}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(case_dir: Path | str) -> dict:
    """Load the case config.yaml as a dict."""
    path = Path(case_dir) / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
