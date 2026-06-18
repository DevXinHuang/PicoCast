#!/usr/bin/env python
# ruff: noqa: E501
"""Build a presentation-ready review dashboard for the K7UAZ case."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexrad_picoballoon.maidenhead_grid import grid_polygon_coords  # noqa: E402
from scripts.candidate_utils import horizontal_distance_km, load_config  # noqa: E402
from scripts.make_review_packet_gis_overlay import parse_ids, site_color  # noqa: E402

GAP_THRESHOLD_MIN = 20.0
SITE_COLORS = {
    "KEMX": "#d73027",
    "KIWA": "#1f78b4",
}


def iso_to_utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, tuple):
        return [clean_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date | datetime):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def compute_grid_center_speeds(track: pd.DataFrame) -> pd.DataFrame:
    track = track.sort_values("time_utc").copy().reset_index(drop=True)
    times = pd.to_datetime(track["time_utc"], utc=True)
    speeds = [np.nan]
    gap_minutes = [np.nan]
    gap_flags = [False]

    for idx in range(1, len(track)):
        dt_h = (times.iloc[idx] - times.iloc[idx - 1]).total_seconds() / 3600.0
        dist_km = horizontal_distance_km(
            float(track.loc[idx - 1, "lat_deg"]),
            float(track.loc[idx - 1, "lon_deg"]),
            float(track.loc[idx, "lat_deg"]),
            float(track.loc[idx, "lon_deg"]),
        )
        gap_min = dt_h * 60.0
        speeds.append(float(dist_km / dt_h) if dt_h > 0 else np.nan)
        gap_minutes.append(float(gap_min))
        gap_flags.append(bool(gap_min > GAP_THRESHOLD_MIN))

    track["grid_center_speed_kmh"] = speeds
    track["gap_from_previous_min"] = gap_minutes
    track["is_gap_from_previous"] = gap_flags
    return track


def telemetry_gap_flags(track: pd.DataFrame, threshold_min: float = GAP_THRESHOLD_MIN) -> list[dict]:
    track = track.sort_values("time_utc").reset_index(drop=True)
    times = pd.to_datetime(track["time_utc"], utc=True)
    gaps = []
    for idx in range(len(track) - 1):
        gap_min = (times.iloc[idx + 1] - times.iloc[idx]).total_seconds() / 60.0
        if gap_min > threshold_min:
            gaps.append(
                {
                    "start_time_utc": str(track.loc[idx, "time_utc"]),
                    "end_time_utc": str(track.loc[idx + 1, "time_utc"]),
                    "gap_min": round(float(gap_min), 1),
                    "start_grid": str(track.loc[idx, "maidenhead_grid"]),
                    "end_grid": str(track.loc[idx + 1, "maidenhead_grid"]),
                }
            )
    return gaps


def interpolate_track_value(track: pd.DataFrame, time_utc: str, column: str) -> tuple[float | None, bool]:
    track = track.sort_values("time_utc").reset_index(drop=True)
    times = pd.to_datetime(track["time_utc"], utc=True)
    target = pd.Timestamp(time_utc)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")

    values = pd.to_numeric(track[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return None, True

    valid_times = times[valid]
    valid_values = values[valid]
    if target <= valid_times.iloc[0]:
        return float(valid_values[0]), True
    if target >= valid_times.iloc[-1]:
        return float(valid_values[-1]), True

    insertion = int(np.searchsorted(valid_times.astype("int64"), target.value))
    left_idx = max(insertion - 1, 0)
    right_idx = min(insertion, len(valid_times) - 1)
    left_t = valid_times.iloc[left_idx]
    right_t = valid_times.iloc[right_idx]
    gap_min = (right_t - left_t).total_seconds() / 60.0
    value = float(
        np.interp(
            target.value,
            [left_t.value, right_t.value],
            [valid_values[left_idx], valid_values[right_idx]],
        )
    )
    return value, bool(gap_min > GAP_THRESHOLD_MIN)


def compute_candidate_segment_speeds(points: pd.DataFrame) -> pd.DataFrame:
    enriched = []
    for _, group in points.groupby("tracklet_id"):
        group = group.sort_values("scan_time_utc").copy().reset_index(drop=True)
        times = pd.to_datetime(group["scan_time_utc"], utc=True)
        speeds = [np.nan]
        for idx in range(1, len(group)):
            dt_h = (times.iloc[idx] - times.iloc[idx - 1]).total_seconds() / 3600.0
            dist_km = horizontal_distance_km(
                float(group.loc[idx - 1, "cluster_lat_deg"]),
                float(group.loc[idx - 1, "cluster_lon_deg"]),
                float(group.loc[idx, "cluster_lat_deg"]),
                float(group.loc[idx, "cluster_lon_deg"]),
            )
            speeds.append(float(dist_km / dt_h) if dt_h > 0 else np.nan)
        group["candidate_segment_speed_kmh"] = speeds
        enriched.append(group)
    return pd.concat(enriched, ignore_index=True) if enriched else points.copy()


def telemetry_records(track: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in track.iterrows():
        rows.append(
            {
                "time_utc": row["time_utc"],
                "lat_deg": float(row["lat_deg"]),
                "lon_deg": float(row["lon_deg"]),
                "alt_m": float(row["alt_m"]),
                "speed_kmh": float(row["speed_kmh"]) if pd.notna(row["speed_kmh"]) else None,
                "grid_center_speed_kmh": (
                    float(row["grid_center_speed_kmh"])
                    if pd.notna(row["grid_center_speed_kmh"])
                    else None
                ),
                "vertical_speed_m_min": (
                    float(row["vertical_speed_m_min"])
                    if pd.notna(row["vertical_speed_m_min"])
                    else None
                ),
                "maidenhead_grid": row["maidenhead_grid"],
                "gap_from_previous_min": (
                    float(row["gap_from_previous_min"])
                    if pd.notna(row["gap_from_previous_min"])
                    else None
                ),
                "is_gap_from_previous": bool(row["is_gap_from_previous"]),
            }
        )
    return rows


def grid_records(track: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in track.drop_duplicates("maidenhead_grid").iterrows():
        coords = grid_polygon_coords(str(row["maidenhead_grid"]))
        records.append(
            {
                "grid": row["maidenhead_grid"],
                "coordinates": [[lat, lon] for lon, lat in coords],
            }
        )
    return records


def radar_site_records(config: dict, sites: set[str]) -> list[dict]:
    records = []
    for site in sorted(sites):
        site_cfg = config.get("radar_sites", {}).get(site)
        if not site_cfg:
            continue
        records.append(
            {
                "site": site,
                "lat": float(site_cfg["lat"]),
                "lon": float(site_cfg["lon"]),
                "alt_m": float(site_cfg.get("alt_m", 0.0)),
                "color": site_color(site),
            }
        )
    return records


def enrich_candidate_points(points: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    points = compute_candidate_segment_speeds(points)
    enriched_rows = []
    for _, row in points.iterrows():
        expected_speed, speed_gap = interpolate_track_value(track, row["scan_time_utc"], "speed_kmh")
        grid_speed, grid_speed_gap = interpolate_track_value(
            track,
            row["scan_time_utc"],
            "grid_center_speed_kmh",
        )
        enriched = row.to_dict()
        enriched["expected_speed_kmh"] = expected_speed
        enriched["expected_grid_center_speed_kmh"] = grid_speed
        enriched["is_low_confidence_speed"] = bool(speed_gap or grid_speed_gap)
        enriched["vertical_mismatch_m"] = float(row["cluster_alt_m"]) - float(row["expected_alt_m"])
        enriched_rows.append(enriched)
    return pd.DataFrame(enriched_rows)


def point_to_record(row: pd.Series) -> dict:
    return {
        "tracklet_id": row["tracklet_id"],
        "radar_site": row["radar_site"],
        "scan_time_utc": row["scan_time_utc"],
        "lat": float(row["cluster_lat_deg"]),
        "lon": float(row["cluster_lon_deg"]),
        "alt_m": float(row["cluster_alt_m"]),
        "expected_alt_m": float(row["expected_alt_m"]),
        "vertical_mismatch_m": float(row["vertical_mismatch_m"]),
        "candidate_segment_speed_kmh": (
            float(row["candidate_segment_speed_kmh"])
            if pd.notna(row["candidate_segment_speed_kmh"])
            else None
        ),
        "expected_speed_kmh": (
            float(row["expected_speed_kmh"]) if pd.notna(row["expected_speed_kmh"]) else None
        ),
        "expected_grid_center_speed_kmh": (
            float(row["expected_grid_center_speed_kmh"])
            if pd.notna(row["expected_grid_center_speed_kmh"])
            else None
        ),
        "is_low_confidence_speed": bool(row["is_low_confidence_speed"]),
        "reflectivity_dbz": (
            float(row["max_reflectivity_dbz"]) if pd.notna(row["max_reflectivity_dbz"]) else None
        ),
        "distance_to_track_corridor_km": (
            float(row["distance_to_track_corridor_km"])
            if pd.notna(row["distance_to_track_corridor_km"])
            else None
        ),
        "nearest_grid": row.get("nearest_grid", ""),
        "doppler_consistency_label": row.get("doppler_consistency_label", "missing_doppler"),
        "doppler_notes": row.get("doppler_notes", ""),
    }


def summarize_review_item(
    review_row: pd.Series,
    item_points: pd.DataFrame,
    cross_lookup: dict[str, dict],
) -> dict:
    speeds = pd.to_numeric(item_points["candidate_segment_speed_kmh"], errors="coerce").dropna()
    mismatch = item_points["vertical_mismatch_m"].abs()
    corridor = pd.to_numeric(item_points["distance_to_track_corridor_km"], errors="coerce").dropna()
    reflectivity = pd.to_numeric(item_points["max_reflectivity_dbz"], errors="coerce").dropna()
    cross = cross_lookup.get(str(review_row["item_id"]), {})

    return {
        "median_abs_vertical_mismatch_m": float(mismatch.median()) if not mismatch.empty else None,
        "max_abs_vertical_mismatch_m": float(mismatch.max()) if not mismatch.empty else None,
        "median_candidate_speed_kmh": float(speeds.median()) if not speeds.empty else None,
        "max_candidate_speed_kmh": float(speeds.max()) if not speeds.empty else None,
        "median_corridor_distance_km": float(corridor.median()) if not corridor.empty else None,
        "reflectivity_min_dbz": float(reflectivity.min()) if not reflectivity.empty else None,
        "reflectivity_max_dbz": float(reflectivity.max()) if not reflectivity.empty else None,
        "low_confidence_speed_points": int(item_points["is_low_confidence_speed"].sum()),
        "time_overlap_min": cross.get("time_overlap_min"),
        "median_horizontal_difference_km": cross.get("median_horizontal_difference_km"),
        "median_altitude_difference_m": cross.get("median_altitude_difference_m"),
        "association_label": cross.get("association_label"),
    }


def build_review_items(
    review_queue: pd.DataFrame,
    points: pd.DataFrame,
    cross_radar: pd.DataFrame,
    top_n: int,
) -> list[dict]:
    cross_lookup = (
        cross_radar.set_index("association_id").to_dict("index") if not cross_radar.empty else {}
    )
    items = []
    for _, row in review_queue.head(top_n).iterrows():
        tracklet_ids = parse_ids(row["tracklet_ids"])
        item_points = points[points["tracklet_id"].isin(tracklet_ids)].copy()
        point_records = [point_to_record(point_row) for _, point_row in item_points.iterrows()]
        by_tracklet = {}
        for tracklet_id in tracklet_ids:
            by_tracklet[tracklet_id] = [
                point_to_record(point_row)
                for _, point_row in item_points[item_points["tracklet_id"] == tracklet_id].iterrows()
            ]
        items.append(
            {
                "review_rank": int(row["review_rank"]),
                "item_type": row["item_type"],
                "item_id": row["item_id"],
                "tracklet_ids": tracklet_ids,
                "radar_sites": parse_ids(row["radar_sites"]),
                "family_ids": parse_ids(row["family_ids"]),
                "review_priority_score": float(row["review_priority_score"]),
                "review_reason": row["review_reason"],
                "plot_path": row.get("plot_path", ""),
                "stats": summarize_review_item(row, item_points, cross_lookup),
                "points": point_records,
                "points_by_tracklet": by_tracklet,
            }
        )
    return items


def build_dashboard_data(config_path: Path, top_n: int = 10) -> dict:
    config = load_config(config_path)
    case_dir = config_path.parent
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    review_queue = pd.read_csv(review_dir / "tracklet_review_queue.csv")
    cross_radar = pd.read_csv(review_dir / "cross_radar_review_queue.csv")
    
    points_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"
    points = pd.read_csv(points_path)
    
    doppler_points_path = case_dir / "outputs" / "discovery" / "doppler_validation" / "doppler_validated_points.csv"
    if doppler_points_path.exists():
        doppler_points = pd.read_csv(doppler_points_path)
        # Only merge the necessary columns if they exist
        cols_to_merge = ["tracklet_id", "scan_time_utc"]
        for col in ["doppler_consistency_label", "doppler_notes", "observed_radial_velocity_ms"]:
            if col in doppler_points.columns:
                cols_to_merge.append(col)
        points = points.merge(
            doppler_points[cols_to_merge],
            on=["tracklet_id", "scan_time_utc"],
            how="left"
        )
        if "doppler_consistency_label" in points.columns:
            points["doppler_consistency_label"] = points["doppler_consistency_label"].fillna("missing_doppler")
    
    track = compute_grid_center_speeds(pd.read_csv(case_dir / "expected_track.csv"))
    points = enrich_candidate_points(points, track)

    active_sites = {
        site for sites in review_queue["radar_sites"].dropna().map(parse_ids) for site in sites
    }
    doppler_summary_path = case_dir / "outputs" / "discovery" / "doppler_validation" / "doppler_tracklet_summary.csv"
    doppler_summaries = []
    if doppler_summary_path.exists():
        doppler_summaries = pd.read_csv(doppler_summary_path).to_dict("records")

    wind_summary_path = case_dir / "outputs" / "wind_context" / "wind_validated_tracklets.csv"
    wind_points = []
    if wind_summary_path.exists():
        wind_points = pd.read_csv(wind_summary_path).to_dict("records")

    items = build_review_items(review_queue, points, cross_radar, top_n)

    top_item = items[0] if items else {}
    data = {
        "case": {
            "case_id": config["case_id"],
            "case_name": config.get("case_name", config["case_id"]),
            "date_local": config.get("date_local"),
            "timezone": config.get("timezone"),
            "top_candidate": top_item.get("item_id"),
            "total_review_items": int(len(review_queue)),
            "shown_review_items": int(min(top_n, len(review_queue))),
        },
        "caveats": [
            "Candidate radar feature, not a detection claim.",
            "Horizontal balloon position comes from Maidenhead grid centers, not GPS.",
            "Speed comparison is approximate, especially during telemetry gaps.",
        ],
        "telemetry": telemetry_records(track),
        "telemetry_gaps": telemetry_gap_flags(track),
        "grid_squares": grid_records(track),
        "radar_sites": radar_site_records(config, active_sites),
        "range_rings_km": config.get("mapping", {}).get("range_rings_km", [50, 100, 150, 200]),
        "review_items": items,
        "doppler_summaries": doppler_summaries,
        "wind_points": wind_points,
        "colors": SITE_COLORS,
    }
    return clean_value(data)


def render_dashboard_html(data: dict) -> str:
    payload = json.dumps(data, allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PicoCAST K7UAZ Review Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #d7dce3;
      --blue: #1f78b4;
      --red: #d73027;
      --purple: #7b3294;
      --green: #18864b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f3f6f9 0%, #eef2f6 42%, #f8fafc 100%);
      color: var(--ink);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .hero {{
      padding: 28px 32px 26px;
      background:
        linear-gradient(135deg, rgba(16, 24, 32, 0.96), rgba(32, 45, 56, 0.94)),
        radial-gradient(circle at 82% 24%, rgba(215, 48, 39, 0.28), transparent 30%),
        radial-gradient(circle at 70% 86%, rgba(31, 120, 180, 0.28), transparent 34%);
      color: white;
      border-bottom: 4px solid #d29922;
    }}
    .hero-inner {{
      max-width: 1380px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: #f3c969;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 52px;
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .hero p {{ margin: 0; color: #d7e0ea; max-width: 920px; line-height: 1.5; font-size: 16px; }}
    .hero-card {{
      min-width: 260px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 10px;
      padding: 14px;
      backdrop-filter: blur(10px);
    }}
    .hero-card span {{ display: block; color: #ccd6e0; font-size: 12px; margin-bottom: 4px; }}
    .hero-card strong {{ font-size: 22px; }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px 24px 28px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric strong {{ font-size: 20px; }}
    .story {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 16px;
    }}
    .story-step {{
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 14px 13px;
      min-height: 132px;
      box-shadow: 0 2px 12px rgba(16, 24, 32, 0.05);
    }}
    .story-step::before {{
      content: attr(data-step);
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: #17202a;
      color: white;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 10px;
    }}
    .story-step h2 {{ margin: 0 0 6px; font-size: 15px; }}
    .story-step p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.42; }}
    .story-step.final {{ border-color: #d29922; box-shadow: 0 6px 18px rgba(210, 153, 34, 0.13); }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .section-heading h2 {{ margin: 0 0 4px; font-size: 16px; }}
    .section-heading p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
      border: 1px solid #e0b33c;
      background: #fff8e5;
      color: #614500;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 800;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(520px, 1.6fr) minmax(340px, 0.8fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(16, 24, 32, 0.06);
      overflow: hidden;
    }}
    #map {{ height: 610px; width: 100%; }}
    .side {{ padding: 16px; }}
    .side h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .side p {{ color: var(--muted); line-height: 1.45; }}
    .rank-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0 16px;
    }}
    .rank-btn {{
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 6px;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    .rank-btn.active {{
      border-color: #2f80ed;
      background: #eaf3ff;
      box-shadow: inset 0 0 0 1px #2f80ed;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}
    .detail {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }}
    .detail span {{ display: block; color: var(--muted); font-size: 11px; }}
    .detail strong {{ font-size: 15px; }}
    .caveat {{
      border-left: 4px solid #d29922;
      background: #fff8e5;
      padding: 10px 12px;
      color: #604100;
      line-height: 1.4;
      border-radius: 4px;
      margin-top: 12px;
    }}
    .evidence-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .evidence-strip div {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }}
    .evidence-strip span {{ display: block; color: var(--muted); font-size: 11px; }}
    .evidence-strip strong {{ font-size: 14px; }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 12px 16px 0;
      border-bottom: 1px solid var(--line);
    }}
    .tab-btn {{
      border: 1px solid var(--line);
      border-bottom: 0;
      background: #eef2f6;
      border-radius: 6px 6px 0 0;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }}
    .tab-btn.active {{ background: white; color: #0b63ce; font-weight: 700; }}
    .chart-wrap {{ padding: 16px; height: 360px; }}
    .chart-wrap canvas {{ width: 100%; height: 300px; }}
    .table-wrap {{ padding: 16px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: #f2f5f8; color: #344054; position: sticky; top: 0; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      background: #edf2f7;
    }}
    .site-kemx {{ color: var(--red); }}
    .site-kiwa {{ color: var(--blue); }}
    footer {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 0 24px 28px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .layout, .summary, .story, .hero-inner, .evidence-strip {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 24px 24px 26px; }}
      .hero h1 {{ font-size: 40px; }}
      #map {{ height: 520px; }}
    }}
    @media (max-width: 520px) {{
      .hero h1 {{ font-size: 34px; }}
      .hero p {{ font-size: 15px; }}
      .shell {{ padding: 16px 16px 24px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <div>
        <p class="eyebrow">PicoCAST review packet</p>
        <h1>K7UAZ Candidate Radar Feature Review</h1>
        <p>
          A chronological research view: start with the known balloon telemetry,
          compare the strongest KEMX/KIWA near-track features, then inspect altitude
          and speed consistency before making any science claim.
        </p>
      </div>
      <div style="display: flex; gap: 12px; align-items: stretch; flex-wrap: wrap;">
        <div class="hero-card">
          <span>Primary artifact</span>
          <strong>Open this page first</strong>
        </div>
        <a href="radar_science_lab.html" class="hero-card" style="text-decoration: none; color: inherit; border-color: #d29922; transition: background 0.2s, transform 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.15)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.background='rgba(255,255,255,0.1)'; this.style.transform='none'">
          <span style="color: #f3c969; font-weight: bold;">Radar Science Lab</span>
          <strong>Learning Module ↗</strong>
        </a>
      </div>
    </div>
  </header>
  <div class="shell">
    <section class="summary">
      <div class="metric"><span>Case</span><strong id="caseMetric"></strong></div>
      <div class="metric"><span>Top candidate</span><strong id="topMetric"></strong></div>
      <div class="metric"><span>Review items shown</span><strong id="countMetric"></strong></div>
      <div class="metric"><span>Major telemetry gap</span><strong id="gapMetric"></strong></div>
    </section>
    <section class="story" aria-label="Research path">
      <article class="story-step" data-step="1">
        <h2>Telemetry anchored the search</h2>
        <p>K7UAZ launch-day records provide time, grid-center position, altitude, climb rate, and speed context.</p>
      </article>
      <article class="story-step" data-step="2">
        <h2>KEMX/KIWA scans were matched</h2>
        <p>Radar scans were compared against the expected track, using the known flight window before any blind search.</p>
      </article>
      <article class="story-step" data-step="3">
        <h2>Tracklets were deduplicated</h2>
        <p>Near-duplicate families were collapsed so the review queue favors inspectable evidence over repeated variants.</p>
      </article>
      <article class="story-step final" data-step="4">
        <h2>This page is the review packet</h2>
        <p>Ranked candidates, GIS context, altitude mismatch, speed comparison, and caveats are combined here.</p>
      </article>
    </section>
    <main class="layout">
      <section class="panel">
        <div class="section-heading">
          <div>
            <h2>Geospatial Review</h2>
            <p>Telemetry grid centers, radar sites, range rings, top tracklets, and cross-radar association links.</p>
          </div>
          <span class="status-pill">Visual inspection only</span>
        </div>
        <div id="map"></div>
      </section>
    <aside class="panel side">
      <h2>Candidate Review</h2>
      <p id="selectedReason"></p>
      <div class="rank-list" id="rankList"></div>
      <div class="detail-grid" id="detailGrid"></div>
      <div class="evidence-strip">
        <div><span>Default focus</span><strong>A001</strong></div>
        <div><span>Radars</span><strong>KEMX + KIWA</strong></div>
        <div><span>Status</span><strong>Needs inspection</strong></div>
      </div>
      <div class="caveat">
        Candidate radar feature, not a detection claim. Horizontal balloon position comes from
        Maidenhead grid centers, not GPS. Speed comparison is approximate during telemetry gaps.
      </div>
    </aside>
    </main>
    <section class="panel">
      <div class="section-heading">
        <div>
          <h2>Telemetry Consistency Checks</h2>
          <p>Altitude is the strongest comparison. Speed is approximate because balloon positions are grid-center estimates.</p>
        </div>
        <span class="status-pill">No confirmed association</span>
      </div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="altitude">Altitude</button>
        <button class="tab-btn" data-tab="mismatch">Vertical mismatch</button>
        <button class="tab-btn" data-tab="speed">Speed</button>
        <button class="tab-btn" data-tab="doppler">Doppler</button>
        <button class="tab-btn" data-tab="wind">Atmospheric Wind</button>
        <button class="tab-btn" data-tab="table">Point table</button>
      </div>
      <div class="chart-wrap tab-panel" id="altitudePanel"><canvas id="altitudeChart"></canvas></div>
      <div class="chart-wrap tab-panel" id="mismatchPanel" style="display:none;"><canvas id="mismatchChart"></canvas></div>
      <div class="chart-wrap tab-panel" id="speedPanel" style="display:none;"><canvas id="speedChart"></canvas></div>
      <div class="table-wrap tab-panel" id="dopplerPanel" style="display:none;">
        <div style="padding: 1rem; color: var(--ink);">
          <h3 style="margin-top: 0;">Doppler Validation Context</h3>
          <p>Geometry/altitude remain interesting. Doppler does not currently strengthen the association.</p>
          <ul>
            <li>Most points have missing Doppler velocity.</li>
            <li>Valid Doppler points show large radial-velocity residuals.</li>
            <li>Spectrum width and RHOHV are context only, not proof.</li>
          </ul>
          <div id="dopplerSummaryTarget" style="margin-top: 1.5rem; overflow-x: auto;"></div>
        </div>
      </div>
      <div class="table-wrap tab-panel" id="tablePanel" style="display:none;"></div>
      <div class="table-wrap tab-panel" id="windPanel" style="display:none;">
        <div style="padding: 1rem; color: var(--ink);">
          <h3 style="margin-top: 0;">Atmospheric Wind Consistency</h3>
          <p>This panel compares the candidate's geometric motion to NOAA HRRR/RAP atmospheric winds.</p>
          <ul>
            <li>If a candidate is a drifting balloon, its speed and heading must roughly match the winds at its altitude.</li>
            <li>Large discrepancies indicate the tracklet is likely an altitude-matching artifact, not a physical object drifting with the wind.</li>
          </ul>
          <div id="windSummaryTarget" style="margin-top: 1.5rem; overflow-x: auto;"></div>
        </div>
      </div>
    </section>
  </div>
  <footer>
    Data source: K7UAZ launch-day telemetry and archived NEXRAD Level II derived candidates.
    Review status: near-track radar features requiring visual inspection and multi-radar confirmation.
  </footer>
  <script>
    const dashboardData = {payload};
    const siteColors = dashboardData.colors;
    let selectedRank = 1;
    let map;
    let candidateLayer;
    let linkLayer;
    let altitudeChart;
    let mismatchChart;
    let speedChart;

    function fmt(value, digits = 1, suffix = "") {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return Number(value).toFixed(digits) + suffix;
    }}
    function timeMs(value) {{ return new Date(value).getTime(); }}
    function timeLabel(value) {{
      return new Date(value).toISOString().substring(11, 16) + "Z";
    }}
    function chartX(value) {{ return timeMs(value); }}
    function tickTime(value) {{ return new Date(value).toISOString().substring(11, 16); }}
    function selectedItem() {{
      return dashboardData.review_items.find(item => item.review_rank === selectedRank)
        || dashboardData.review_items[0];
    }}

    function initSummary() {{
      document.getElementById("caseMetric").textContent = dashboardData.case.case_id;
      document.getElementById("topMetric").textContent = dashboardData.case.top_candidate || "n/a";
      document.getElementById("countMetric").textContent =
        dashboardData.case.shown_review_items + " of " + dashboardData.case.total_review_items;
      const majorGap = dashboardData.telemetry_gaps[0];
      document.getElementById("gapMetric").textContent =
        majorGap ? majorGap.start_time_utc.substring(11, 16) + "Z to "
          + majorGap.end_time_utc.substring(11, 16) + "Z" : "none";
    }}

    function initMap() {{
      const centerLat = dashboardData.telemetry.reduce((s, p) => s + p.lat_deg, 0)
        / dashboardData.telemetry.length;
      const centerLon = dashboardData.telemetry.reduce((s, p) => s + p.lon_deg, 0)
        / dashboardData.telemetry.length;
      map = L.map("map", {{ scrollWheelZoom: true }}).setView([centerLat, centerLon], 9);
      L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 18,
        attribution: "OpenStreetMap"
      }}).addTo(map);
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
        {{ attribution: "Esri", maxZoom: 18 }}
      ).addTo(map);

      dashboardData.grid_squares.forEach(grid => {{
        L.polygon(grid.coordinates, {{
          color: "#5d6673",
          weight: 1,
          fillColor: "#9aa5b1",
          fillOpacity: 0.16
        }}).bindTooltip("Maidenhead grid " + grid.grid).addTo(map);
      }});

      const telem = dashboardData.telemetry;
      for (let i = 0; i < telem.length - 1; i++) {{
        const gap = telem[i + 1].is_gap_from_previous;
        L.polyline(
          [[telem[i].lat_deg, telem[i].lon_deg], [telem[i + 1].lat_deg, telem[i + 1].lon_deg]],
          {{ color: "#202020", weight: 4, opacity: gap ? 0.42 : 0.75, dashArray: gap ? "10 8" : null }}
        ).bindTooltip(gap ? "Telemetry gap bridged by interpolation" : "Telemetry segment").addTo(map);
      }}
      telem.forEach(point => {{
        L.circleMarker([point.lat_deg, point.lon_deg], {{
          radius: 4,
          color: "#111",
          fillColor: "#fff",
          fillOpacity: 1
        }}).bindTooltip(
          "Telemetry " + timeLabel(point.time_utc) + "<br>Alt " + fmt(point.alt_m, 0, " m")
          + "<br>Speed " + fmt(point.speed_kmh, 0, " km/h")
        ).addTo(map);
      }});

      dashboardData.radar_sites.forEach(site => {{
        L.circleMarker([site.lat, site.lon], {{
          radius: 7,
          color: site.color,
          fillColor: "#fff",
          fillOpacity: 1,
          weight: 2
        }}).bindTooltip(site.site + " radar").addTo(map);
        dashboardData.range_rings_km.forEach(radius => {{
          L.circle([site.lat, site.lon], {{
            radius: radius * 1000,
            color: site.color,
            fill: false,
            opacity: 0.18,
            weight: 1
          }}).addTo(map);
        }});
      }});

      candidateLayer = L.layerGroup().addTo(map);
      linkLayer = L.layerGroup().addTo(map);
    }}

    function updateMap(item) {{
      candidateLayer.clearLayers();
      linkLayer.clearLayers();
      const bounds = [];
      const centroids = {{}};
      item.tracklet_ids.forEach(trackletId => {{
        const pts = item.points_by_tracklet[trackletId] || [];
        if (!pts.length) return;
        const site = pts[0].radar_site;
        const color = siteColors[site] || "#555";
        const latLngs = pts.map(p => [p.lat, p.lon]);
        latLngs.forEach(ll => bounds.push(ll));
        centroids[trackletId] = [
          pts.reduce((s, p) => s + p.lat, 0) / pts.length,
          pts.reduce((s, p) => s + p.lon, 0) / pts.length
        ];
        L.polyline(latLngs, {{ color, weight: 5, opacity: 0.9 }}).addTo(candidateLayer);
        pts.forEach(p => {{
          L.circleMarker([p.lat, p.lon], {{
            radius: 6,
            color,
            fillColor: color,
            fillOpacity: 0.9
          }}).bindPopup(
            "<b>" + trackletId + "</b><br>" +
            "Radar: " + site + "<br>" +
            "UTC: " + p.scan_time_utc + "<br>" +
            "Candidate alt: " + fmt(p.alt_m, 0, " m") + "<br>" +
            "Expected alt: " + fmt(p.expected_alt_m, 0, " m") + "<br>" +
            "Mismatch: " + fmt(p.vertical_mismatch_m, 0, " m") + "<br>" +
            "Reflectivity: " + fmt(p.reflectivity_dbz, 1, " dBZ") + "<br>" +
            "Candidate speed: " + fmt(p.candidate_segment_speed_kmh, 1, " km/h")
          ).addTo(candidateLayer);
        }});
      }});
      if (item.tracklet_ids.length >= 2) {{
        const first = centroids[item.tracklet_ids[0]];
        const second = centroids[item.tracklet_ids[1]];
        if (first && second) {{
          L.polyline([first, second], {{
            color: "#7b3294",
            weight: 3,
            opacity: 0.85,
            dashArray: "8 8"
          }}).bindTooltip("Cross-radar candidate association " + item.item_id).addTo(linkLayer);
        }}
      }}
      if (bounds.length) map.fitBounds(bounds, {{ padding: [35, 35], maxZoom: 11 }});
    }}

    function renderRankButtons() {{
      const container = document.getElementById("rankList");
      container.innerHTML = "";
      dashboardData.review_items.forEach(item => {{
        const button = document.createElement("button");
        button.className = "rank-btn" + (item.review_rank === selectedRank ? " active" : "");
        button.innerHTML = "<strong>#" + item.review_rank + " " + item.item_id + "</strong><br>"
          + "<span>" + item.tracklet_ids.join("; ") + "</span>";
        button.addEventListener("click", () => {{
          selectedRank = item.review_rank;
          updateSelection();
        }});
        container.appendChild(button);
      }});
    }}

    function renderDetails(item) {{
      document.getElementById("selectedReason").textContent = item.review_reason;
      const s = item.stats;
      const details = [
        ["Priority", fmt(item.review_priority_score, 3)],
        ["Median vertical mismatch", fmt(s.median_abs_vertical_mismatch_m, 0, " m")],
        ["Max vertical mismatch", fmt(s.max_abs_vertical_mismatch_m, 0, " m")],
        ["Median radar speed", fmt(s.median_candidate_speed_kmh, 1, " km/h")],
        ["Max radar speed", fmt(s.max_candidate_speed_kmh, 1, " km/h")],
        ["Median corridor distance", fmt(s.median_corridor_distance_km, 1, " km")],
        ["Reflectivity range", fmt(s.reflectivity_min_dbz, 1) + " to " + fmt(s.reflectivity_max_dbz, 1) + " dBZ"],
        ["Low-confidence speed pts", String(s.low_confidence_speed_points)]
      ];
      document.getElementById("detailGrid").innerHTML = details.map(d =>
        '<div class="detail"><span>' + d[0] + '</span><strong>' + d[1] + '</strong></div>'
      ).join("");
    }}

    function baseChartOptions(yTitle) {{
      return {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: "bottom" }},
          tooltip: {{
            callbacks: {{
              title: items => items.length ? tickTime(items[0].parsed.x) + "Z" : ""
            }}
          }}
        }},
        scales: {{
          x: {{
            type: "linear",
            ticks: {{ callback: tickTime }},
            title: {{ display: true, text: "UTC" }}
          }},
          y: {{ title: {{ display: true, text: yTitle }} }}
        }}
      }};
    }}

    function destroyCharts() {{
      [altitudeChart, mismatchChart, speedChart].forEach(chart => {{
        if (chart) chart.destroy();
      }});
    }}

    function renderCharts(item) {{
      destroyCharts();
      const telemetry = dashboardData.telemetry;
      const telemetryAlt = telemetry.map(p => ({{ x: chartX(p.time_utc), y: p.alt_m }}));
      const pointsBySite = {{}};
      item.points.forEach(p => {{
        if (!pointsBySite[p.radar_site]) pointsBySite[p.radar_site] = [];
        pointsBySite[p.radar_site].push(p);
      }});
      const candidateAltSets = Object.entries(pointsBySite).map(([site, pts]) => ({{
        label: site + " candidate altitude",
        data: pts.map(p => ({{ x: chartX(p.scan_time_utc), y: p.alt_m }})),
        showLine: true,
        borderColor: siteColors[site] || "#555",
        backgroundColor: siteColors[site] || "#555",
        pointRadius: 5
      }}));
      const band = delta => telemetry.map(p => ({{ x: chartX(p.time_utc), y: p.alt_m + delta }}));
      altitudeChart = new Chart(document.getElementById("altitudeChart"), {{
        type: "scatter",
        data: {{
          datasets: [
            {{ label: "Balloon altitude", data: telemetryAlt, type: "line", borderColor: "#111", pointRadius: 3 }},
            {{ label: "+/-250 m band", data: band(250), type: "line", borderColor: "#18864b", borderDash: [4, 4], pointRadius: 0 }},
            {{ label: "-250 m band", data: band(-250), type: "line", borderColor: "#18864b", borderDash: [4, 4], pointRadius: 0 }},
            {{ label: "+1000 m reference", data: band(1000), type: "line", borderColor: "#d29922", borderDash: [6, 6], pointRadius: 0 }},
            {{ label: "-1000 m reference", data: band(-1000), type: "line", borderColor: "#d29922", borderDash: [6, 6], pointRadius: 0 }},
            ...candidateAltSets
          ]
        }},
        options: baseChartOptions("Altitude (m)")
      }});

      const mismatchSets = Object.entries(pointsBySite).map(([site, pts]) => ({{
        label: site + " mismatch",
        data: pts.map(p => ({{ x: chartX(p.scan_time_utc), y: p.vertical_mismatch_m }})),
        showLine: true,
        borderColor: siteColors[site] || "#555",
        backgroundColor: siteColors[site] || "#555",
        pointRadius: 5
      }}));
      mismatchChart = new Chart(document.getElementById("mismatchChart"), {{
        type: "scatter",
        data: {{
          datasets: [
            {{ label: "0 m", data: telemetry.map(p => ({{ x: chartX(p.time_utc), y: 0 }})), type: "line", borderColor: "#111", pointRadius: 0 }},
            ...mismatchSets
          ]
        }},
        options: baseChartOptions("Vertical mismatch (m)")
      }});

      const speedSets = Object.entries(pointsBySite).map(([site, pts]) => ({{
        label: site + " candidate speed",
        data: pts.filter(p => p.candidate_segment_speed_kmh !== null).map(p => ({{
          x: chartX(p.scan_time_utc),
          y: p.candidate_segment_speed_kmh
        }})),
        showLine: true,
        borderColor: siteColors[site] || "#555",
        backgroundColor: siteColors[site] || "#555",
        pointRadius: 5
      }}));
      speedChart = new Chart(document.getElementById("speedChart"), {{
        type: "scatter",
        data: {{
          datasets: [
            {{
              label: "Telemetry speed_kmh",
              data: telemetry.filter(p => p.speed_kmh !== null).map(p => ({{ x: chartX(p.time_utc), y: p.speed_kmh }})),
              type: "line",
              borderColor: "#111",
              pointRadius: 3
            }},
            {{
              label: "Derived grid-center speed",
              data: telemetry.filter(p => p.grid_center_speed_kmh !== null).map(p => ({{ x: chartX(p.time_utc), y: p.grid_center_speed_kmh }})),
              type: "line",
              borderColor: "#7b3294",
              borderDash: [5, 5],
              pointRadius: 3
            }},
            ...speedSets
          ]
        }},
        options: baseChartOptions("Speed (km/h)")
      }});
    }}

    function renderPointTable(item) {{
      const rows = item.points.map(p => `
        <tr>
          <td>${{p.tracklet_id}}</td>
          <td><span class="badge ${{p.radar_site === "KEMX" ? "site-kemx" : "site-kiwa"}}">${{p.radar_site}}</span></td>
          <td>${{timeLabel(p.scan_time_utc)}}Z</td>
          <td>${{fmt(p.alt_m, 0)}}</td>
          <td>${{fmt(p.expected_alt_m, 0)}}</td>
          <td>${{fmt(p.vertical_mismatch_m, 0)}}</td>
          <td>${{fmt(p.candidate_segment_speed_kmh, 1)}}</td>
          <td>${{fmt(p.expected_speed_kmh, 1)}}</td>
          <td>${{p.is_low_confidence_speed ? "gap/low confidence" : "normal"}}</td>
        </tr>`).join("");
      document.getElementById("tablePanel").innerHTML = `
        <table>
          <thead><tr>
            <th>Tracklet</th><th>Radar</th><th>UTC</th><th>Candidate alt m</th>
            <th>Expected alt m</th><th>Mismatch m</th><th>Radar speed km/h</th>
            <th>Telemetry speed km/h</th><th>Speed confidence</th>
          </tr></thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function renderWindSummary(item) {{
      if (!dashboardData.wind_points || dashboardData.wind_points.length === 0) {{
        document.getElementById("windSummaryTarget").innerHTML = "<p>No wind data available.</p>";
        return;
      }}
      
      const relatedPoints = dashboardData.wind_points.filter(wp => item.tracklet_ids.includes(wp.tracklet_id));
      if (relatedPoints.length === 0) {{
        document.getElementById("windSummaryTarget").innerHTML = "<p>No wind data for this candidate.</p>";
        return;
      }}
      
      const rows = relatedPoints.map(wp => `
        <tr>
          <td>${{wp.scan_time_utc}}</td>
          <td>${{fmt(wp.candidate_speed_kmh, 1)}}</td>
          <td>${{fmt(wp.candidate_bearing_deg, 1)}}&deg;</td>
          <td>${{fmt(wp.hrrr_wind_speed_kmh, 1)}}</td>
          <td>${{fmt(wp.hrrr_wind_to_direction_deg, 1)}}&deg;</td>
          <td>${{fmt(wp.speed_difference_kmh, 1)}}</td>
          <td>${{fmt(wp.bearing_difference_deg, 1)}}&deg;</td>
          <td><span class="badge" style="background:#fef0f0; color:#c53030; border:1px solid #fed7d7;">${{wp.wind_consistency_label}}</span></td>
        </tr>
      `).join("");
      
      document.getElementById("windSummaryTarget").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Scan Time</th>
              <th>Candidate Speed (km/h)</th>
              <th>Candidate Bearing</th>
              <th>Wind Speed (km/h)</th>
              <th>Wind Bearing</th>
              <th>Speed Diff (km/h)</th>
              <th>Bearing Diff</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function renderDopplerSummary(item) {{
      if (!dashboardData.doppler_summaries || dashboardData.doppler_summaries.length === 0) {{
        document.getElementById("dopplerSummaryTarget").innerHTML = "<p>No Doppler data available.</p>";
        return;
      }}
      
      const relatedSummaries = dashboardData.doppler_summaries.filter(ds => item.tracklet_ids.includes(ds.tracklet_id));
      if (relatedSummaries.length === 0) {{
        document.getElementById("dopplerSummaryTarget").innerHTML = "<p>No Doppler data for this candidate.</p>";
        return;
      }}
      
      const rows = relatedSummaries.map(ds => `
        <tr>
          <td><strong>${{ds.tracklet_id}}</strong></td>
          <td>${{ds.n_valid_doppler_points}} / ${{ds.n_points}}</td>
          <td>${{fmt(ds.median_observed_radial_velocity_ms, 1)}}</td>
          <td>${{fmt(ds.median_expected_radial_velocity_ms, 1)}}</td>
          <td>${{fmt(ds.median_abs_radial_velocity_residual_ms, 1)}}</td>
          <td><span class="badge" style="background:#fef0f0; color:#c53030; border:1px solid #fed7d7;">${{ds.doppler_consistency_label}}</span></td>
          <td>${{ds.doppler_notes || ""}}</td>
        </tr>
      `).join("");
      
      document.getElementById("dopplerSummaryTarget").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Tracklet</th>
              <th>Valid Doppler Points</th>
              <th>Median Observed RV (m/s)</th>
              <th>Median Expected RV (m/s)</th>
              <th>Median Abs Residual (m/s)</th>
              <th>Label</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function updateSelection() {{
      const item = selectedItem();
      renderRankButtons();
      renderDetails(item);
      updateMap(item);
      renderCharts(item);
      renderMismatchTable(item);
      renderSpeedTable(item);
      renderDopplerSummary(item);
      renderWindSummary(item);
      renderPointTable(item);
    }}

    document.querySelectorAll(".tab-btn").forEach(button => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(panel => panel.style.display = "none");
        button.classList.add("active");
        document.getElementById(button.dataset.tab + "Panel").style.display = "block";
        setTimeout(() => {{
          if (altitudeChart) altitudeChart.resize();
          if (mismatchChart) mismatchChart.resize();
          if (speedChart) speedChart.resize();
        }}, 50);
      }});
    }});

    initSummary();
    initMap();
    updateSelection();
  </script>
</body>
</html>
"""


def write_dashboard(config_path: Path, top_n: int = 10) -> tuple[Path, Path, Path]:
    case_dir = config_path.parent
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    review_dir.mkdir(parents=True, exist_ok=True)
    data = build_dashboard_data(config_path, top_n=top_n)

    data_path = review_dir / "review_packet_dashboard_data.json"
    html_path = review_dir / "review_packet_dashboard.html"
    index_path = review_dir / "index.html"
    html = render_dashboard_html(data)
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    index_path.write_text(html, encoding="utf-8")
    return html_path, data_path, index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--top-n", type=int, default=10, help="Number of review rows to include")
    args = parser.parse_args()

    html_path, data_path, index_path = write_dashboard(args.config, top_n=args.top_n)
    print(f"Wrote primary review dashboard: {index_path}")
    print(f"Wrote review dashboard: {html_path}")
    print(f"Wrote dashboard data: {data_path}")

    # 1. Generate Radar Science Visual Lab
    try:
        from scripts.make_radar_science_lab import make_radar_science_lab
        make_radar_science_lab(args.config, rank=1)
    except Exception as e:
        print(f"Warning: Failed to auto-generate radar science lab: {e}")

    # 2. Publish all assets to docs/review_packet
    case_dir = args.config.parent
    review_dir = case_dir / "outputs" / "discovery" / "review_packet"
    docs_dir = ROOT / "docs" / "review_packet"
    docs_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    if review_dir.exists():
        for item in review_dir.iterdir():
            dst = docs_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        print(f"Published all review packet assets to {docs_dir}")

    # 3. Validate assets using the validation module
    try:
        from scripts.validate_review_packet_assets import validate_directory
        print("\n--- Running Asset Validation ---")
        success = validate_directory(docs_dir)
        if not success:
            print("Error: Asset validation failed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error executing validator: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
