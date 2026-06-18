#!/usr/bin/env python
# ruff: noqa: E501
"""Generate a Radar Science Visual Lab page for a PicoCAST case.

This creates an educational HTML page that helps users understand the existing
candidate validation maps.  It does NOT make detection claims – all language
describes *candidate radar features* for visual review.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

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


# ---------------------------------------------------------------------------
# GeoJSON loader
# ---------------------------------------------------------------------------


def load_geojson(path: Path) -> dict:
    """Load a GeoJSON file, returning an empty FeatureCollection if missing."""
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_lab_data(
    config_path: Path,
    radar_site: str | None = None,
    rank: int = 1,
) -> dict:
    """Collect all data needed for the Radar Science Lab from existing outputs."""
    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_dir = config_path.parent
    case_id = config.get("case_id", case_dir.name)
    maps_dir = case_dir / "outputs" / "maps"
    cand_dir = candidates_dir(config_path, site)

    # Radar site info from config
    site_info = config.get("radar_sites", {}).get(site, {})
    radar_lat = site_info.get("lat", 0.0)
    radar_lon = site_info.get("lon", 0.0)
    radar_alt_m = site_info.get("alt_m", 0.0)

    # Load GeoJSON layers
    expected_track = load_geojson(maps_dir / "expected_track.geojson")
    top_candidates = load_geojson(maps_dir / "top_candidates.geojson")

    # Select the candidate by rank
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

    # Find closest expected-track point
    track_point = find_closest_track_point(
        expected_track.get("features", []), scan_time
    )
    if track_point is None:
        raise SystemExit("No expected-track point found for the candidate scan time.")
    tp_props = track_point["properties"]
    tp_lon, tp_lat = track_point["geometry"]["coordinates"][:2]

    # Altitude validation data
    alt_csv = cand_dir / "altitude_validation" / "altitude_prioritized_candidates.csv"
    alt_info: dict = {}
    if alt_csv.exists():
        alt_df = pd.read_csv(alt_csv)
        match = alt_df[alt_df["original_candidate_rank"] == rank]
        if not match.empty:
            row = match.iloc[0]
            alt_info = {
                "expected_alt_m": _safe_float(row.get("interpolated_expected_alt_m")),
                "candidate_alt_m": _safe_float(row.get("candidate_alt_m")),
                "signed_vertical_m": _safe_float(row.get("signed_vertical_interp_m")),
                "abs_vertical_m": _safe_float(row.get("abs_vertical_interp_m")),
                "altitude_label": str(row.get("altitude_consistency_label", "")),
                "altitude_priority_rank": _safe_int(
                    row.get("altitude_priority_rank")
                ),
            }

    # Expected balloon altitude from track
    expected_alt_m = tp_props.get("alt_m", 0.0)
    if alt_info and alt_info.get("expected_alt_m") is not None:
        expected_alt_m = alt_info["expected_alt_m"]

    # Candidate altitude from top_candidates or altitude CSV
    candidate_alt_m = cand_props.get("candidate_alt_m", 0.0)
    if alt_info and alt_info.get("candidate_alt_m") is not None:
        candidate_alt_m = alt_info["candidate_alt_m"]

    # Compute horizontal distance from radar to candidate
    radar_to_cand_km = _haversine_km(radar_lat, radar_lon, cand_lat, cand_lon)

    # Cross-radar info
    cross_radar_csv = (
        case_dir / "outputs" / "discovery" / "review_packet" / "cross_radar_review_queue.csv"
    )
    cross_radar_note = "Not available for this candidate."
    if cross_radar_csv.exists():
        try:
            cr_df = pd.read_csv(cross_radar_csv)
            # Check if this candidate's scan time appears in cross-radar data
            if "scan_time_utc" in cr_df.columns:
                cr_match = cr_df[cr_df["scan_time_utc"] == scan_time_str]
                if not cr_match.empty:
                    cross_radar_note = (
                        "This scan time appears in the cross-radar review queue, "
                        "suggesting possible multi-radar association. "
                        "Cross-radar consistency is suggestive, not proof."
                    )
        except Exception:
            pass

    # Relative path to validation map
    validation_map_path = maps_dir / f"rank_{rank:02d}_validation_map.html"
    validation_map_rel = ""
    if validation_map_path.exists():
        validation_map_rel = str(
            validation_map_path.relative_to(case_dir)
        )

    return {
        "case_id": case_id,
        "radar_site": site,
        "radar_lat": radar_lat,
        "radar_lon": radar_lon,
        "radar_alt_m": radar_alt_m,
        "rank": rank,
        "candidate_score": cand_props.get("candidate_score", 0.0),
        "candidate_label": cand_props.get("candidate_label", ""),
        "scan_time_utc": scan_time_str,
        "search_window": cand_props.get("search_window", ""),
        "cand_lat": cand_lat,
        "cand_lon": cand_lon,
        "candidate_alt_m": candidate_alt_m,
        "expected_lat": tp_lat,
        "expected_lon": tp_lon,
        "expected_alt_m": expected_alt_m,
        "maidenhead_grid": tp_props.get("maidenhead_grid", ""),
        "horizontal_distance_km": cand_props.get("horizontal_distance_km", 0.0),
        "vertical_distance_m": cand_props.get("vertical_distance_m", 0.0),
        "max_reflectivity_dbz": cand_props.get("max_reflectivity_dbz"),
        "n_gates": cand_props.get("n_gates", 0),
        "alt_info": alt_info,
        "radar_to_cand_km": radar_to_cand_km,
        "cross_radar_note": cross_radar_note,
        "validation_map_rel": validation_map_rel,
    }


def _safe_float(v) -> float | None:
    """Return a float or None for NaN/missing values."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> int | None:
    """Return an int or None for NaN/missing values."""
    if v is None:
        return None
    try:
        f = float(v)
        return int(f) if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Haversine distance in kilometers."""
    r = 6371.0088
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html_mod.escape(str(text))


def _fmt(value, fmt: str = ".1f") -> str:
    """Format a number, returning '—' for None/NaN."""
    if value is None:
        return "—"
    try:
        f = float(value)
        if not math.isfinite(f):
            return "—"
        return format(f, fmt)
    except (ValueError, TypeError):
        return "—"


def _build_vertical_beam_svg(data: dict) -> str:
    """Build an SVG showing the vertical beam / altitude intuition diagram."""
    # All in meters, but display as km where useful
    radar_alt_m = data["radar_alt_m"]
    range_km = data["radar_to_cand_km"]
    expected_alt_m = data["expected_alt_m"]
    candidate_alt_m = data["candidate_alt_m"]

    # SVG dimensions
    w, h = 700, 400
    margin_left, margin_bottom, margin_top, margin_right = 80, 50, 30, 40

    plot_w = w - margin_left - margin_right
    plot_h = h - margin_top - margin_bottom

    # Determine altitude range for display
    all_alts = [radar_alt_m, expected_alt_m, candidate_alt_m]
    all_alts = [a for a in all_alts if a is not None]
    if not all_alts:
        all_alts = [0, 10000]
    min_alt = min(0, min(all_alts) - 500)
    max_alt = max(all_alts) + 1500

    # Approximate beam geometry
    # Standard atmosphere: beam center altitude approximation
    # h ≈ r * sin(elev) + r² / (2 * R_e * k)
    # For a 0.5° elevation, k = 4/3 earth radius model
    earth_r_km = 6371.0
    k = 4.0 / 3.0
    effective_r = earth_r_km * k

    # We don't know exact elevation, so compute approximate beam-center
    # altitude at the candidate range using the candidate_alt_m
    # as a reference. Show a plausible beam envelope.
    beam_width_deg = 1.0  # approximate NEXRAD beam width

    # Compute what elevation angle would put beam-center at candidate_alt_m
    # at the given range: alt = radar_alt + r*sin(e) + r²/(2*kR)
    range_m = range_km * 1000.0
    earth_curv_m = (range_m ** 2) / (2.0 * effective_r * 1000.0)
    if range_m > 0:
        sin_elev = (candidate_alt_m - radar_alt_m - earth_curv_m) / range_m
        sin_elev = max(-1.0, min(1.0, sin_elev))
        elev_rad = math.asin(sin_elev)
        elev_deg = math.degrees(elev_rad)
    else:
        elev_deg = 0.5
        elev_rad = math.radians(elev_deg)

    # Generate beam centerline points
    n_pts = 50
    beam_center_pts = []
    beam_upper_pts = []
    beam_lower_pts = []
    half_bw = math.radians(beam_width_deg / 2.0)

    for i in range(n_pts + 1):
        r_km = range_km * i / n_pts
        r_m = r_km * 1000.0
        curv = (r_m ** 2) / (2.0 * effective_r * 1000.0)
        center_alt = radar_alt_m + r_m * math.sin(elev_rad) + curv
        upper_alt = radar_alt_m + r_m * math.sin(elev_rad + half_bw) + curv
        lower_alt = radar_alt_m + r_m * math.sin(elev_rad - half_bw) + curv
        beam_center_pts.append((r_km, center_alt))
        beam_upper_pts.append((r_km, upper_alt))
        beam_lower_pts.append((r_km, lower_alt))

    # Coordinate transforms
    def x_coord(r_km_val: float) -> float:
        return margin_left + (r_km_val / range_km) * plot_w if range_km > 0 else margin_left

    def y_coord(alt_m_val: float) -> float:
        frac = (alt_m_val - min_alt) / (max_alt - min_alt) if max_alt > min_alt else 0.5
        return margin_top + plot_h * (1.0 - frac)

    # Build SVG
    lines: list[str] = []
    lines.append(
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:{w}px;width:100%;height:auto;background:#1a1a2e;'
        f'border-radius:8px;font-family:Inter,sans-serif;">'
    )

    # Ground line
    ground_y = y_coord(radar_alt_m)
    lines.append(
        f'<line x1="{margin_left}" y1="{ground_y}" x2="{w - margin_right}" '
        f'y2="{ground_y}" stroke="#5a5a7a" stroke-width="2" stroke-dasharray="6,3"/>'
    )
    lines.append(
        f'<text x="{w - margin_right + 4}" y="{ground_y + 4}" '
        f'fill="#8a8aaa" font-size="10">ground ({_fmt(radar_alt_m, ".0f")} m)</text>'
    )

    # Beam envelope (filled polygon)
    envelope_pts = []
    for r_km_val, alt_val in beam_upper_pts:
        envelope_pts.append(f"{x_coord(r_km_val):.1f},{y_coord(alt_val):.1f}")
    for r_km_val, alt_val in reversed(beam_lower_pts):
        envelope_pts.append(f"{x_coord(r_km_val):.1f},{y_coord(alt_val):.1f}")
    lines.append(
        f'<polygon points="{" ".join(envelope_pts)}" '
        f'fill="rgba(100,180,255,0.12)" stroke="none"/>'
    )

    # Beam centerline
    center_path_pts = " ".join(
        f"{x_coord(r):.1f},{y_coord(a):.1f}" for r, a in beam_center_pts
    )
    lines.append(
        f'<polyline points="{center_path_pts}" '
        f'fill="none" stroke="#64b4ff" stroke-width="1.5" stroke-dasharray="4,3"/>'
    )

    # Beam upper/lower edges
    for edge_pts, _label in [(beam_upper_pts, "upper"), (beam_lower_pts, "lower")]:
        edge_path = " ".join(
            f"{x_coord(r):.1f},{y_coord(a):.1f}" for r, a in edge_pts
        )
        lines.append(
            f'<polyline points="{edge_path}" '
            f'fill="none" stroke="#64b4ff" stroke-width="0.8" opacity="0.5"/>'
        )

    # Radar icon at origin
    rx, ry = x_coord(0), y_coord(radar_alt_m)
    lines.append(
        f'<circle cx="{rx}" cy="{ry}" r="6" fill="#4488cc" stroke="white" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{rx}" y="{ry - 12}" text-anchor="middle" '
        f'fill="#88bbff" font-size="11" font-weight="600">{_esc(data["radar_site"])}</text>'
    )

    # Expected balloon altitude line
    exp_y = y_coord(expected_alt_m)
    lines.append(
        f'<line x1="{margin_left}" y1="{exp_y}" x2="{w - margin_right}" '
        f'y2="{exp_y}" stroke="#44dd88" stroke-width="1.5" stroke-dasharray="8,4"/>'
    )
    lines.append(
        f'<text x="{margin_left + 4}" y="{exp_y - 6}" '
        f'fill="#44dd88" font-size="10" font-weight="600">'
        f'Expected balloon alt ≈ {_fmt(expected_alt_m, ".0f")} m</text>'
    )

    # Candidate gate position
    cx_pos = x_coord(range_km)
    cy_pos = y_coord(candidate_alt_m)
    lines.append(
        f'<circle cx="{cx_pos}" cy="{cy_pos}" r="7" '
        f'fill="#ff4444" stroke="white" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{cx_pos - 10}" y="{cy_pos - 12}" text-anchor="end" '
        f'fill="#ff6666" font-size="10" font-weight="600">'
        f'Candidate gate ({_fmt(candidate_alt_m, ".0f")} m)</text>'
    )

    # Altitude mismatch arrow
    alt_mismatch = data.get("vertical_distance_m", 0)
    if expected_alt_m and candidate_alt_m:
        mid_x = cx_pos + 14
        lines.append(
            f'<line x1="{mid_x}" y1="{exp_y}" x2="{mid_x}" y2="{cy_pos}" '
            f'stroke="#ffaa44" stroke-width="2"/>'
        )
        mid_y = (exp_y + cy_pos) / 2
        lines.append(
            f'<text x="{mid_x + 6}" y="{mid_y + 4}" '
            f'fill="#ffaa44" font-size="10" font-weight="600">'
            f'Δalt ≈ {_fmt(abs(alt_mismatch), ".0f")} m</text>'
        )

    # Axis labels
    lines.append(
        f'<text x="{w / 2}" y="{h - 8}" text-anchor="middle" '
        f'fill="#aaaacc" font-size="11">Range from radar (km)</text>'
    )
    lines.append(
        f'<text x="14" y="{h / 2}" text-anchor="middle" '
        f'fill="#aaaacc" font-size="11" '
        f'transform="rotate(-90, 14, {h / 2})">Altitude (m)</text>'
    )

    # Range tick marks
    n_ticks = 5
    for i in range(n_ticks + 1):
        r_val = range_km * i / n_ticks
        tx = x_coord(r_val)
        lines.append(
            f'<line x1="{tx}" y1="{h - margin_bottom}" '
            f'x2="{tx}" y2="{h - margin_bottom + 5}" stroke="#8a8aaa" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{tx}" y="{h - margin_bottom + 16}" text-anchor="middle" '
            f'fill="#8a8aaa" font-size="9">{_fmt(r_val, ".0f")}</text>'
        )

    # Altitude tick marks
    alt_range = max_alt - min_alt
    alt_step = 2000 if alt_range > 8000 else (1000 if alt_range > 3000 else 500)
    alt_tick = math.ceil(min_alt / alt_step) * alt_step
    while alt_tick <= max_alt:
        ty = y_coord(alt_tick)
        lines.append(
            f'<line x1="{margin_left - 5}" y1="{ty}" '
            f'x2="{margin_left}" y2="{ty}" stroke="#8a8aaa" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 8}" y="{ty + 3}" text-anchor="end" '
            f'fill="#8a8aaa" font-size="9">{int(alt_tick)}</text>'
        )
        alt_tick += alt_step

    # Label for beam envelope
    last_upper = beam_upper_pts[-1]
    last_lower = beam_lower_pts[-1]
    beam_width_at_range_m = last_upper[1] - last_lower[1]
    label_y = y_coord(last_upper[1]) - 8
    lines.append(
        f'<text x="{x_coord(range_km) - 4}" y="{label_y}" text-anchor="end" '
        f'fill="#64b4ff" font-size="9" opacity="0.8">'
        f'beam width ≈ {_fmt(beam_width_at_range_m, ".0f")} m</text>'
    )

    # Approximate elevation label
    lines.append(
        f'<text x="{margin_left + 40}" y="{y_coord(radar_alt_m) - 20}" '
        f'fill="#64b4ff" font-size="9" opacity="0.7">'
        f'approx elev ≈ {_fmt(elev_deg, ".1f")}°</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


def _build_doppler_svg() -> str:
    """Build an SVG showing the Doppler radial velocity concept."""
    w, h = 500, 320
    cx, cy = 80, 200  # radar position
    bx, by = 380, 100  # balloon position

    lines: list[str] = []
    lines.append(
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:{w}px;width:100%;height:auto;background:#1a1a2e;'
        f'border-radius:8px;font-family:Inter,sans-serif;">'
    )

    # Radar-to-balloon line (LOS)
    lines.append(
        f'<line x1="{cx}" y1="{cy}" x2="{bx}" y2="{by}" '
        f'stroke="#555577" stroke-width="1.5" stroke-dasharray="6,4"/>'
    )

    # Radar icon
    lines.append(
        f'<circle cx="{cx}" cy="{cy}" r="8" fill="#4488cc" stroke="white" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" '
        f'fill="#88bbff" font-size="11" font-weight="600">Radar</text>'
    )

    # Balloon icon
    lines.append(
        f'<circle cx="{bx}" cy="{by}" r="7" fill="#44dd88" stroke="white" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{bx}" y="{by - 14}" text-anchor="middle" '
        f'fill="#66eebb" font-size="11" font-weight="600">Balloon</text>'
    )

    # Balloon motion vector (mostly sideways, slightly toward radar)
    # The motion is roughly perpendicular to LOS with a small radial component
    motion_dx, motion_dy = 70, 30  # eastward, slightly south
    mx, my = bx + motion_dx, by + motion_dy
    lines.append(
        f'<line x1="{bx}" y1="{by}" x2="{mx}" y2="{my}" '
        f'stroke="#ffaa44" stroke-width="2.5" marker-end="url(#arrowMotion)"/>'
    )
    lines.append(
        f'<text x="{mx + 8}" y="{my - 2}" fill="#ffaa44" font-size="10" '
        f'font-weight="600">Balloon motion</text>'
    )

    # Compute radial component (projection onto LOS direction)
    los_dx, los_dy = bx - cx, by - cy
    los_len = math.sqrt(los_dx ** 2 + los_dy ** 2)
    los_ux, los_uy = los_dx / los_len, los_dy / los_len

    # Radial projection
    radial_proj = motion_dx * los_ux + motion_dy * los_uy
    rad_end_x = bx + radial_proj * los_ux
    rad_end_y = by + radial_proj * los_uy

    # Sideways component (radial end → motion tip drawn directly below)

    # Draw radial component
    lines.append(
        f'<line x1="{bx}" y1="{by}" x2="{rad_end_x:.1f}" y2="{rad_end_y:.1f}" '
        f'stroke="#ff4444" stroke-width="2" marker-end="url(#arrowRad)"/>'
    )
    lines.append(
        f'<text x="{(bx + rad_end_x) / 2 - 30}" y="{(by + rad_end_y) / 2 + 20}" '
        f'fill="#ff6666" font-size="10" font-weight="600">'
        f'toward/away</text>'
    )
    lines.append(
        f'<text x="{(bx + rad_end_x) / 2 - 30}" y="{(by + rad_end_y) / 2 + 32}" '
        f'fill="#ff6666" font-size="9">'
        f'= Doppler-visible</text>'
    )

    # Draw sideways component from radial end to motion end
    lines.append(
        f'<line x1="{rad_end_x:.1f}" y1="{rad_end_y:.1f}" '
        f'x2="{mx}" y2="{my}" '
        f'stroke="#44aaff" stroke-width="2" marker-end="url(#arrowSide)"/>'
    )
    lines.append(
        f'<text x="{(rad_end_x + mx) / 2 + 8:.0f}" '
        f'y="{(rad_end_y + my) / 2 - 8:.0f}" '
        f'fill="#66bbff" font-size="10" font-weight="600">'
        f'sideways</text>'
    )
    lines.append(
        f'<text x="{(rad_end_x + mx) / 2 + 8:.0f}" '
        f'y="{(rad_end_y + my) / 2 + 4:.0f}" '
        f'fill="#66bbff" font-size="9">'
        f'= weak Doppler</text>'
    )

    # Dashed line connecting radial end to motion end (right angle indicator)
    lines.append(
        f'<line x1="{rad_end_x:.1f}" y1="{rad_end_y:.1f}" '
        f'x2="{rad_end_x:.1f}" y2="{rad_end_y:.1f}" '
        f'stroke="transparent" stroke-width="0"/>'
    )

    # LOS label
    mid_los_x = (cx + bx) / 2
    mid_los_y = (cy + by) / 2
    lines.append(
        f'<text x="{mid_los_x - 30}" y="{mid_los_y - 10}" '
        f'fill="#8888aa" font-size="9" '
        f'transform="rotate({math.degrees(math.atan2(by - cy, bx - cx)):.1f}, '
        f'{mid_los_x - 30}, {mid_los_y - 10})">'
        f'line of sight</text>'
    )

    # Arrow markers
    lines.append("""<defs>
<marker id="arrowMotion" markerWidth="8" markerHeight="6" refX="8" refY="3"
        orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ffaa44"/></marker>
<marker id="arrowRad" markerWidth="8" markerHeight="6" refX="8" refY="3"
        orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ff4444"/></marker>
<marker id="arrowSide" markerWidth="8" markerHeight="6" refX="8" refY="3"
        orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#44aaff"/></marker>
</defs>""")

    # Legend
    lines.append(
        f'<text x="{w / 2}" y="{h - 10}" text-anchor="middle" '
        f'fill="#8888aa" font-size="10">'
        f'Doppler measures only the toward/away component of motion</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


def render_lab_html(data: dict, iframe_rel_path: str | None = None) -> str:
    """Render the full Radar Science Visual Lab HTML page."""
    rank = data["rank"]
    site = data["radar_site"]
    case_id = data["case_id"]
    h_dist = data["horizontal_distance_km"]
    v_dist = data["vertical_distance_m"]
    max_dbz = data["max_reflectivity_dbz"]
    score = data["candidate_score"]
    label = data["candidate_label"]
    scan_time = data["scan_time_utc"]
    maidenhead = data["maidenhead_grid"]
    alt_info = data.get("alt_info", {})

    vertical_svg = _build_vertical_beam_svg(data)
    doppler_svg = _build_doppler_svg()

    # Validation map embed
    validation_map_embed = ""
    if data["validation_map_rel"]:
        if iframe_rel_path is not None:
            rel_path = iframe_rel_path
        else:
            map_filename = Path(data["validation_map_rel"]).name
            rel_path = "./" + map_filename
        validation_map_embed = f"""
        <div class="map-embed-wrapper">
          <iframe src="{_esc(rel_path)}" class="map-iframe"
                  title="Rank {rank} validation map"
                  sandbox="allow-scripts allow-same-origin"></iframe>
          <p class="map-link">
            <a href="{_esc(rel_path)}" target="_blank" rel="noopener">
              Open full validation map in new tab ↗
            </a>
          </p>
        </div>
"""
    else:
        validation_map_embed = """
        <p class="note-box">
          Validation map not found. Run
          <code>make_candidate_validation_map.py</code> first to generate it.
        </p>
"""

    # Altitude info lines
    alt_lines_html = ""
    if alt_info:
        exp_a = alt_info.get("expected_alt_m")
        cand_a = alt_info.get("candidate_alt_m")
        signed_v = alt_info.get("signed_vertical_m")
        alt_label = alt_info.get("altitude_label", "").replace("_", " ")
        alt_rank = alt_info.get("altitude_priority_rank")
        exp_a_str = _fmt(exp_a, '.0f')
        cand_a_str = _fmt(cand_a, '.0f')
        sv_str = _fmt(signed_v, '+.0f')
        al_str = _esc(alt_label)
        ar_str = _fmt(alt_rank, '.0f')
        alt_lines_html = (
            f'<tr><td>Expected balloon alt (interp.)</td>'
            f'<td>{exp_a_str} m</td></tr>'
            f'<tr><td>Candidate gate alt (beam-center)</td>'
            f'<td>{cand_a_str} m</td></tr>'
            f'<tr><td>Vertical mismatch (signed)</td>'
            f'<td>{sv_str} m</td></tr>'
            f'<tr><td>Altitude consistency</td>'
            f'<td>{al_str}</td></tr>'
            f'<tr><td>Altitude priority rank</td>'
            f'<td>{ar_str}</td></tr>'
        )

    page = f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Radar Science Visual Lab – {_esc(case_id)}</title>
  <style>
    :root {{
      --bg: #0f0f1a;
      --panel: #1a1a2e;
      --panel-border: #2a2a44;
      --ink: #e0e0ee;
      --muted: #8888aa;
      --accent-blue: #4488cc;
      --accent-green: #44dd88;
      --accent-red: #ff4444;
      --accent-amber: #ffaa44;
      --accent-purple: #9966ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
      padding: 0;
    }}
    .hero {{
      background:
        linear-gradient(135deg, rgba(16, 24, 40, 0.97), rgba(26, 26, 46, 0.95)),
        radial-gradient(circle at 80% 20%, rgba(68, 136, 204, 0.2), transparent 40%),
        radial-gradient(circle at 20% 80%, rgba(68, 221, 136, 0.15), transparent 40%);
      padding: 40px 32px 32px;
      border-bottom: 3px solid var(--accent-amber);
    }}
    .hero-inner {{
      max-width: 900px;
      margin: 0 auto;
    }}
    .hero h1 {{
      font-size: 26px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 8px;
    }}
    .hero .subtitle {{
      color: var(--accent-amber);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .hero p {{
      color: var(--muted);
      font-size: 14px;
      max-width: 700px;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      padding: 24px 32px 60px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 28px 28px 24px;
      margin-bottom: 24px;
    }}
    .section h2 {{
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 16px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--panel-border);
    }}
    .section h3 {{
      font-size: 15px;
      font-weight: 600;
      color: var(--accent-blue);
      margin: 16px 0 8px;
    }}
    .section p, .section li {{
      font-size: 14px;
      color: var(--ink);
      margin-bottom: 8px;
    }}
    .section ul, .section ol {{
      padding-left: 22px;
      margin-bottom: 12px;
    }}
    .section li {{
      margin-bottom: 6px;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px 16px;
      align-items: start;
      margin: 16px 0;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 24px;
      height: 24px;
      border-radius: 4px;
      border: 1.5px solid rgba(255,255,255,0.2);
      vertical-align: middle;
      flex-shrink: 0;
    }}
    .legend-swatch.circle {{
      border-radius: 50%;
    }}
    .legend-swatch.line {{
      width: 36px;
      height: 3px;
      border-radius: 2px;
      align-self: center;
    }}
    .legend-desc {{
      font-size: 13px;
      color: var(--ink);
      line-height: 1.5;
    }}
    .legend-desc .detail {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
    }}
    .caveat {{
      background: rgba(255, 170, 68, 0.08);
      border-left: 3px solid var(--accent-amber);
      padding: 12px 16px;
      border-radius: 0 6px 6px 0;
      margin: 16px 0;
      font-size: 13px;
      color: var(--accent-amber);
    }}
    .caveat strong {{
      color: #ffc966;
    }}
    .info-box {{
      background: rgba(68, 136, 204, 0.08);
      border-left: 3px solid var(--accent-blue);
      padding: 12px 16px;
      border-radius: 0 6px 6px 0;
      margin: 16px 0;
      font-size: 13px;
      color: #88bbff;
    }}
    .note-box {{
      background: rgba(136, 136, 170, 0.08);
      border-left: 3px solid var(--muted);
      padding: 12px 16px;
      border-radius: 0 6px 6px 0;
      margin: 16px 0;
      font-size: 13px;
      color: var(--muted);
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 13px;
    }}
    table.data-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid var(--panel-border);
    }}
    table.data-table td:first-child {{
      color: var(--muted);
      width: 45%;
    }}
    table.data-table td:last-child {{
      color: var(--ink);
      font-weight: 500;
    }}
    .svg-container {{
      margin: 20px 0;
      text-align: center;
    }}
    .map-embed-wrapper {{
      margin: 16px 0;
    }}
    .map-iframe {{
      width: 100%;
      height: 450px;
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      background: #222;
    }}
    .map-link {{
      text-align: center;
      margin-top: 8px;
    }}
    .map-link a {{
      color: var(--accent-blue);
      text-decoration: none;
      font-size: 13px;
    }}
    .map-link a:hover {{
      text-decoration: underline;
    }}
    .mistakes-list {{
      counter-reset: mistake;
      list-style: none;
      padding-left: 0;
    }}
    .mistakes-list li {{
      counter-increment: mistake;
      padding: 10px 16px 10px 48px;
      position: relative;
      background: rgba(255, 68, 68, 0.04);
      border-radius: 6px;
      margin-bottom: 8px;
      border: 1px solid rgba(255, 68, 68, 0.1);
    }}
    .mistakes-list li::before {{
      content: counter(mistake);
      position: absolute;
      left: 14px;
      top: 10px;
      width: 22px;
      height: 22px;
      background: rgba(255, 68, 68, 0.15);
      color: var(--accent-red);
      border-radius: 50%;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    code {{
      background: rgba(255,255,255,0.06);
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 12px;
      color: var(--accent-purple);
    }}
    .footer {{
      text-align: center;
      padding: 20px;
      color: var(--muted);
      font-size: 11px;
      border-top: 1px solid var(--panel-border);
      margin-top: 32px;
    }}
    a {{ color: var(--accent-blue); }}
  </style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div class="subtitle">PicoCAST · Radar Review Aid</div>
    <h1>🔬 Radar Science Visual Lab</h1>
    <p>
      An educational guide for understanding candidate radar feature maps.
      This page explains what the map symbols mean, how radar geometry works,
      and common interpretation pitfalls. All features described here are
      <strong>plausible candidates for visual review</strong> – not confirmed
      balloon detections.
    </p>
  </div>
</div>

<div class="container">

  <!-- ============================================================ -->
  <!-- SECTION 1: How to read the candidate map                      -->
  <!-- ============================================================ -->
  <div class="section" id="sec-map-guide">
    <h2>§1 — How to read the candidate map</h2>
    <p>
      The validation map shows a single ranked candidate radar feature and
      its spatial relationship to the expected balloon region. Here is what
      each element means:
    </p>

    <div class="legend-grid">
      <span class="legend-swatch" style="background:rgba(51,136,255,0.2);
            border-color:#3388ff;"></span>
      <div class="legend-desc">
        <strong>Blue shaded square</strong> — Maidenhead grid-square region
        from balloon telemetry.
        <span class="detail">
          This is an approximate expected balloon region derived from the
          telemetry grid locator, not from exact GPS coordinates. The true
          balloon position could be anywhere within this square.
        </span>
      </div>

      <span class="legend-swatch circle" style="background:#000;
            border-color:#444;"></span>
      <div class="legend-desc">
        <strong>Black circle</strong> — Expected grid-square center.
        <span class="detail">
          The geometric center of the Maidenhead grid square. Used as the
          reference point for distance calculations, but the actual balloon
          position within the grid square is unknown.
        </span>
      </div>

      <span class="legend-swatch circle" style="background:#ff4444;
            border-color:rgba(255,255,255,0.4);"></span>
      <div class="legend-desc">
        <strong>Red marker</strong> — Candidate radar feature center.
        <span class="detail">
          The location of the candidate radar return. This is a radar feature
          that scored well in the candidate search – it has not been confirmed
          as the balloon.
        </span>
      </div>

      <span class="legend-swatch line" style="background:#ff4444;"></span>
      <div class="legend-desc">
        <strong>Red line</strong> — Horizontal separation.
        <span class="detail">
          The great-circle distance between the expected grid-square center
          and the candidate radar feature. Shorter distances are more
          interesting but do not prove association.
        </span>
      </div>

      <span class="legend-swatch circle" style="background:#ccc;
            border-color:#999;"></span>
      <div class="legend-desc">
        <strong>White/gray dots</strong> — Nearby NEXRAD radar gates.
        <span class="detail">
          Individual radar sample volumes (gates) from the same scan, colored
          by reflectivity. These show the local radar context around the
          candidate feature.
        </span>
      </div>

      <span class="legend-swatch circle" style="background:#88bbff;
            border-color:#4488cc;"></span>
      <div class="legend-desc">
        <strong>Blue highlighted points</strong> — Candidate-related gates.
        <span class="detail">
          Gates that are part of or near the candidate cluster. These are
          highlighted for review context – they are not confirmed balloon
          detections.
        </span>
      </div>
    </div>

    <h3>Key metrics shown on the map</h3>
    <ul>
      <li>
        <strong>max dBZ</strong>: Strongest reflectivity in the local
        candidate group. Picoballoons are weak radar targets, so values
        near or below 0 dBZ are expected if present at all.
      </li>
      <li>
        <strong>Altitude mismatch</strong>: Difference between the expected
        balloon altitude (from telemetry) and the radar gate/beam-center
        altitude. Beam-center altitude is an approximation – the actual
        beam has width that increases with range.
      </li>
    </ul>

    <div class="caveat">
      <strong>⚠ Important caveat:</strong> The balloon position is estimated
      from Maidenhead grid-square centers, not exact GPS. The grid square is
      roughly 5–9 km across. Distance and altitude figures use this estimated
      center, so the true separation could be larger or smaller.
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- SECTION 2: Plan-view geometry                                 -->
  <!-- ============================================================ -->
  <div class="section" id="sec-plan-view">
    <h2>§2 — Plan-view geometry</h2>
    <p>
      The plan-view (top-down) map answers a key question: <em>Is the radar
      candidate horizontally near the expected balloon region?</em>
    </p>

    {validation_map_embed}

    <div class="info-box">
      <strong>What to look for:</strong> The red candidate marker should be
      near or within the blue grid-square region. The red line shows the
      horizontal separation. Range rings (if shown) help you gauge the
      distance from the radar site. Closer candidates are more interesting,
      but horizontal closeness alone is not sufficient – altitude must also
      be considered.
    </div>

    <h3>Geometry summary for Rank {rank}</h3>
    <table class="data-table">
      <tr><td>Radar site</td><td>{_esc(site)}</td></tr>
      <tr><td>Scan time (UTC)</td><td>{_esc(scan_time)}</td></tr>
      <tr><td>Expected grid square</td><td>{_esc(maidenhead)}</td></tr>
      <tr><td>Horizontal distance to candidate</td><td>{_fmt(h_dist, '.2f')} km</td></tr>
      <tr><td>Range from radar to candidate</td><td>{_fmt(data['radar_to_cand_km'], '.1f')} km</td></tr>
    </table>
  </div>

  <!-- ============================================================ -->
  <!-- SECTION 3: Vertical beam / altitude intuition                 -->
  <!-- ============================================================ -->
  <div class="section" id="sec-vertical">
    <h2>§3 — Vertical beam / altitude intuition</h2>
    <p>
      Horizontal closeness alone is not enough. The candidate also needs
      to be close to the expected balloon altitude. At longer ranges, the
      radar beam is higher and wider, so altitude interpretation has
      greater uncertainty.
    </p>

    <div class="svg-container">
      {vertical_svg}
    </div>

    <div class="caveat">
      <strong>⚠ Approximate teaching diagram.</strong> The beam geometry
      shown is computed from approximate radar equations using a 4/3 effective
      Earth radius model. The actual elevation angle used by the radar for
      this scan may differ. Beam width and altitude are approximate.
    </div>

    <h3>What this diagram shows</h3>
    <ul>
      <li>
        <strong>Blue line and shading</strong>: The approximate radar beam
        centerline and beam-width envelope from the radar site to the
        candidate range.
      </li>
      <li>
        <strong>Green dashed line</strong>: The expected balloon altitude
        from telemetry (grid-square center interpolation).
      </li>
      <li>
        <strong>Red dot</strong>: The candidate radar gate altitude
        (beam-center approximation at the gate range).
      </li>
      <li>
        <strong>Orange bar</strong>: The altitude mismatch between the
        expected balloon altitude and the candidate gate altitude.
      </li>
    </ul>

    <div class="info-box">
      <strong>Key insight:</strong> Beam width grows with range. At
      {_fmt(data['radar_to_cand_km'], '.0f')} km range, the beam is
      approximately ±0.5° wide, meaning the altitude uncertainty of a
      single gate is significant. A candidate that is close in beam-center
      altitude is interesting, but the true reflecting object could be
      anywhere within the beam width.
    </div>

    <h3>Altitude metrics for Rank {rank}</h3>
    <table class="data-table">
      <tr><td>Candidate altitude (beam-center)</td><td>{_fmt(data['candidate_alt_m'], '.0f')} m</td></tr>
      <tr><td>Expected balloon altitude</td><td>{_fmt(data['expected_alt_m'], '.0f')} m</td></tr>
      <tr><td>Vertical mismatch</td><td>{_fmt(abs(v_dist), '.0f')} m</td></tr>
      {alt_lines_html}
    </table>
  </div>

  <!-- ============================================================ -->
  <!-- SECTION 4: Doppler intuition preview                          -->
  <!-- ============================================================ -->
  <div class="section" id="sec-doppler">
    <h2>§4 — Doppler intuition preview</h2>
    <p>
      NEXRAD measures <em>radial velocity</em> — the component of motion
      directly toward or away from the radar, not the full speed of the
      object. Understanding this is crucial for interpreting Doppler
      evidence.
    </p>

    <div class="svg-container">
      {doppler_svg}
    </div>

    <h3>Key Doppler concepts</h3>
    <ul>
      <li>
        <strong>Radial velocity</strong> is motion along the radar's
        line of sight — toward or away from the radar antenna.
      </li>
      <li>
        If a balloon moves mostly <strong>sideways</strong> relative to
        the radar (perpendicular to the line of sight), its radial velocity
        can be <strong>near zero</strong> even though it is moving quickly.
      </li>
      <li>
        <strong>Low radial velocity does not mean the object is
        stationary.</strong> It may simply mean the motion is mostly
        sideways.
      </li>
      <li>
        <strong>Near-zero Doppler</strong> can make velocity evidence
        weak or harder to interpret for balloon-like targets.
      </li>
    </ul>

    <div class="info-box">
      <strong>Future development:</strong> The next PicoCAST development
      phase should compute the expected radial velocity for each radar/scan
      geometry, enabling direct comparison between expected and observed
      Doppler. This would strengthen or weaken the candidate association.
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- SECTION 5: Candidate explanation card                         -->
  <!-- ============================================================ -->
  <div class="section" id="sec-candidate-card">
    <h2>Rank {rank} candidate: why it is interesting</h2>

    <table class="data-table">
      <tr><td>Candidate rank</td><td>{rank}</td></tr>
      <tr><td>Candidate score</td><td>{_fmt(score, '.3f')}</td></tr>
      <tr><td>Candidate label</td><td>{_esc(label.replace('_', ' '))}</td></tr>
      <tr><td>Radar site</td><td>{_esc(site)}</td></tr>
      <tr><td>Scan time (UTC)</td><td>{_esc(scan_time)}</td></tr>
      <tr><td>Search window</td><td>{_esc(data['search_window'])}</td></tr>
      <tr><td>Horizontal distance</td><td>{_fmt(h_dist, '.2f')} km</td></tr>
      <tr><td>Vertical mismatch</td><td>{_fmt(abs(v_dist), '.0f')} m</td></tr>
      <tr><td>Max reflectivity</td><td>{_fmt(max_dbz, '.1f')} dBZ</td></tr>
      <tr><td>Number of gates</td><td>{data['n_gates']}</td></tr>
      <tr><td>Range from radar</td><td>{_fmt(data['radar_to_cand_km'], '.1f')} km</td></tr>
      <tr><td>Expected grid square</td><td>{_esc(maidenhead)}</td></tr>
      <tr><td>Cross-radar association</td><td>{_esc(data['cross_radar_note'])}</td></tr>
    </table>

    <p>
      This candidate is interesting because it is close to the expected
      balloon region both horizontally ({_fmt(h_dist, '.1f')} km) and
      vertically ({_fmt(abs(v_dist), '.0f')} m altitude mismatch).
      However, the balloon position is estimated from a Maidenhead
      grid-square center (not exact GPS), and the radar gate altitude is a
      beam-center approximation. <strong>This should be treated as a review
      candidate, not a confirmed balloon detection.</strong>
    </p>

    <div class="caveat">
      <strong>⚠ Grid-square caveat:</strong> The expected balloon position
      is the center of a {_esc(maidenhead)} Maidenhead grid square, which
      is approximately 5–9 km across. The actual balloon could be anywhere
      within this square, meaning the true horizontal distance could be
      significantly different from {_fmt(h_dist, '.2f')} km.
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- SECTION 6: Common radar mistakes                              -->
  <!-- ============================================================ -->
  <div class="section" id="sec-mistakes">
    <h2>Common ways to fool yourself with radar</h2>
    <p>
      When reviewing radar candidates for possible balloon association,
      be aware of these common interpretation pitfalls:
    </p>

    <ol class="mistakes-list">
      <li>
        <strong>Close horizontally, wrong altitude.</strong> A radar
        feature can appear at the same latitude/longitude as the expected
        balloon but at a completely different altitude. Always check both
        horizontal and vertical proximity.
      </li>
      <li>
        <strong>Beam width increases with range.</strong> At short range
        (e.g., 20 km), the NEXRAD beam is narrow (~350 m). At long range
        (e.g., 200 km), it is much wider (~3,500 m). This means altitude
        estimates from a single gate become less precise at greater
        distances from the radar.
      </li>
      <li>
        <strong>Radar scans are not instantaneous.</strong> A single
        volume scan takes several minutes. The balloon telemetry timestamp
        and the radar scan time may not perfectly overlap, and the
        balloon moves during this interval. Small timing mismatches
        can introduce position errors.
      </li>
      <li>
        <strong>Radial velocity is not full speed.</strong> NEXRAD
        Doppler measures only the toward/away component of motion.
        If the object moves mostly sideways relative to the radar,
        observed radial velocity will be small regardless of actual
        speed.
      </li>
      <li>
        <strong>Near-zero Doppler ≠ stationary object.</strong> A
        target moving perpendicular to the radar line of sight has
        near-zero radial velocity. Do not conclude that an object
        with low Doppler is not moving.
      </li>
      <li>
        <strong>Weak point targets have many possible explanations.</strong>
        A radar return near the expected balloon region could be weather,
        biological scatter (birds, insects), ground clutter, anomalous
        propagation, side-lobe contamination, or processing artifacts.
        The candidate scoring helps prioritize, but does not rule out
        alternative explanations.
      </li>
      <li>
        <strong>Cross-radar consistency is stronger, but still not
        proof.</strong> Seeing similar features from multiple radars
        at the same time and location significantly increases
        confidence, but coincidental agreement is possible.
        Multi-radar association should be treated as strong evidence,
        not definitive proof.
      </li>
    </ol>
  </div>

  <div class="footer">
    PicoCAST Radar Science Visual Lab · {_esc(case_id)} ·
    Generated for educational review purposes only ·
    All features described are candidate radar returns requiring visual
    inspection – not confirmed balloon detections
  </div>

</div>

</body>
</html>
"""
    return page


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def make_radar_science_lab(
    config_path: Path,
    radar_site: str | None = None,
    rank: int = 1,
) -> Path:
    """Build the Radar Science Visual Lab HTML page for a case."""
    data = collect_lab_data(config_path, radar_site=radar_site, rank=rank)

    case_dir = config_path.parent
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    review_dir.mkdir(parents=True, exist_ok=True)

    # Write to discovery review_packet
    map_filename = Path(data["validation_map_rel"]).name if data["validation_map_rel"] else ""
    html_discovery = render_lab_html(data, iframe_rel_path=f"./{map_filename}" if map_filename else None)
    output_path = review_dir / "radar_science_lab.html"
    output_path.write_text(html_discovery, encoding="utf-8")
    print(f"Wrote {output_path}")

    # Copy validation map if it exists
    if data["validation_map_rel"]:
        src_map = case_dir / data["validation_map_rel"]
        if src_map.exists():
            import shutil
            shutil.copy2(src_map, review_dir / map_filename)
            print(f"Copied validation map {src_map.name} to {review_dir}")

    # Also write to case-level review_packet for convenience
    case_review_dir = case_dir / "review_packet"
    case_review_dir.mkdir(parents=True, exist_ok=True)
    html_case = render_lab_html(data, iframe_rel_path=f"./{map_filename}" if map_filename else None)
    case_output_path = case_review_dir / "radar_science_lab.html"
    case_output_path.write_text(html_case, encoding="utf-8")
    print(f"Wrote {case_output_path}")

    if data["validation_map_rel"]:
        src_map = case_dir / data["validation_map_rel"]
        if src_map.exists():
            import shutil
            shutil.copy2(src_map, case_review_dir / map_filename)
            print(f"Copied validation map {src_map.name} to {case_review_dir}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument(
        "--radar-site",
        help="Radar site to use (default: primary from config)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Candidate rank to feature (default: 1)",
    )
    args = parser.parse_args()
    make_radar_science_lab(args.config, radar_site=args.radar_site, rank=args.rank)


if __name__ == "__main__":
    main()
