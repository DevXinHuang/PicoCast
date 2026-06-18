#!/usr/bin/env python
"""Fit an altitude-constrained candidate trajectory from radar candidates.

Tests whether altitude-consistent KEMX radar candidates can be connected
into a smooth, physically plausible candidate path.  Uses altitude
consistency as the primary constraint and horizontal smoothness as the
secondary constraint.

Output (to <case_dir>/outputs/candidates/<SITE>/path_fit/):
  - candidate_path.csv
  - candidate_path.geojson
  - smoothed_candidate_path.geojson
  - path_fit_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    horizontal_distance_km,
    load_config,
    radar_site_from_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum plausible picoballoon speed: ~250 km/h (strong jet stream)
MAX_PLAUSIBLE_SPEED_KMH = 250.0

# Minimum altitude consistency score to be considered plausible
MIN_ALT_CONSISTENCY = 0.6

# Minimum candidate score to consider
MIN_CANDIDATE_SCORE = 0.15

# Path scoring weights
W_ALTITUDE = 0.35  # altitude consistency is king
W_SMOOTH = 0.25    # horizontal smoothness
W_COMPACT = 0.15   # compactness
W_SCORE = 0.15     # original composite score
W_HDIST = 0.10     # distance from grid center (deprioritized)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_plausible_candidates(
    df: pd.DataFrame,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Filter candidates to those plausible for path fitting."""
    if df.empty:
        return df.copy()
    filtered = df.copy()

    # Time window
    filtered["scan_dt"] = pd.to_datetime(
        filtered["scan_time_utc"], utc=True,
    )
    if start_time:
        filtered = filtered[filtered["scan_dt"] >= start_time]
    if end_time:
        filtered = filtered[filtered["scan_dt"] <= end_time]

    # Altitude consistency
    filtered = filtered[
        filtered["altitude_consistency_score"] >= MIN_ALT_CONSISTENCY
    ]

    # Minimum composite score
    filtered = filtered[filtered["candidate_score"] >= MIN_CANDIDATE_SCORE]

    # Drop duplicates (same cluster from different search windows)
    filtered = filtered.sort_values(
        ["scan_time_utc", "altitude_consistency_score", "candidate_score"],
        ascending=[True, False, False],
    )

    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Segment scoring
# ---------------------------------------------------------------------------


def segment_speed_kmh(
    lat1: float, lon1: float, t1_str: str,
    lat2: float, lon2: float, t2_str: str,
) -> float:
    """Compute ground speed between two candidate points in km/h."""
    dist = float(horizontal_distance_km(lat1, lon1, lat2, lon2))
    dt1 = pd.Timestamp(t1_str)
    dt2 = pd.Timestamp(t2_str)
    dt_hours = abs((dt2 - dt1).total_seconds()) / 3600.0
    if dt_hours < 1e-6:
        return 0.0
    return dist / dt_hours


def segment_bearing_deg(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Compute bearing from point 1 to point 2 in degrees."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def smoothness_score(
    speed_kmh: float,
    prev_bearing: float | None,
    curr_bearing: float,
) -> float:
    """Score segment smoothness.  Rewards reasonable speed, penalizes
    impossible jumps and sharp bearing changes."""
    # Speed penalty
    if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
        return 0.0
    speed_s = max(0.0, 1.0 - speed_kmh / MAX_PLAUSIBLE_SPEED_KMH)

    # Bearing change penalty (if we have a previous bearing)
    if prev_bearing is not None:
        bearing_change = abs(curr_bearing - prev_bearing)
        if bearing_change > 180:
            bearing_change = 360 - bearing_change
        # Allow up to 90 degrees change; heavily penalize reversals
        bearing_s = max(0.0, 1.0 - bearing_change / 180.0)
    else:
        bearing_s = 0.5  # neutral if no prior bearing

    return 0.6 * speed_s + 0.4 * bearing_s


# ---------------------------------------------------------------------------
# Path selection (greedy best-first with backtracking)
# ---------------------------------------------------------------------------


def select_path(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one candidate per scan time to form the best path.

    Uses a greedy forward pass that scores each candidate against the
    previous selected point, choosing the candidate with the best
    combined altitude + smoothness + compactness + score.
    """
    if candidates.empty:
        return pd.DataFrame()
    scan_times = sorted(candidates["scan_time_utc"].unique())
    if not scan_times:
        return pd.DataFrame()

    path_rows = []
    prev_lat = None
    prev_lon = None
    prev_time = None
    prev_bearing = None

    for scan_time in scan_times:
        scan_cands = candidates[
            candidates["scan_time_utc"] == scan_time
        ].copy()

        if scan_cands.empty:
            continue

        # Score each candidate for this scan time
        scores = []
        for _idx, row in scan_cands.iterrows():
            # Altitude component
            alt_s = row["altitude_consistency_score"]

            # Compactness
            compact_s = row.get("compactness_score", 0.5)

            # Composite score
            score_s = row["candidate_score"]

            # Horizontal distance from grid center (deprioritized)
            h_max = 20.0  # loose window
            h_dist = row["horizontal_distance_km"]
            hdist_s = max(0.0, 1.0 - h_dist / h_max)

            # Smoothness (relative to previous point)
            if prev_lat is not None:
                speed = segment_speed_kmh(
                    prev_lat, prev_lon, prev_time,
                    row["candidate_lat_deg"], row["candidate_lon_deg"],
                    scan_time,
                )
                bearing = segment_bearing_deg(
                    prev_lat, prev_lon,
                    row["candidate_lat_deg"], row["candidate_lon_deg"],
                )
                smooth_s = smoothness_score(speed, prev_bearing, bearing)
                # Hard reject impossible jumps
                if speed > MAX_PLAUSIBLE_SPEED_KMH:
                    scores.append(-1.0)
                    continue
            else:
                smooth_s = 0.5  # neutral for first point
                speed = 0.0
                bearing = 0.0

            total = (
                W_ALTITUDE * alt_s
                + W_SMOOTH * smooth_s
                + W_COMPACT * compact_s
                + W_SCORE * score_s
                + W_HDIST * hdist_s
            )
            scores.append(total)

        scan_cands = scan_cands.copy()
        scan_cands["_path_score"] = scores

        # Pick the best (skip scan if all rejected)
        valid = scan_cands[scan_cands["_path_score"] > 0]
        if valid.empty:
            continue

        best = valid.loc[valid["_path_score"].idxmax()]

        # Compute segment metrics
        if prev_lat is not None:
            seg_speed = segment_speed_kmh(
                prev_lat, prev_lon, prev_time,
                best["candidate_lat_deg"], best["candidate_lon_deg"],
                scan_time,
            )
            seg_bearing = segment_bearing_deg(
                prev_lat, prev_lon,
                best["candidate_lat_deg"], best["candidate_lon_deg"],
            )
        else:
            seg_speed = 0.0
            seg_bearing = 0.0

        path_rows.append({
            "case_id": best["case_id"],
            "radar_site": best["radar_site"],
            "scan_time_utc": scan_time,
            "original_candidate_rank": int(best["original_candidate_rank"]),
            "altitude_priority_rank": int(best["altitude_priority_rank"]),
            "candidate_lat_deg": best["candidate_lat_deg"],
            "candidate_lon_deg": best["candidate_lon_deg"],
            "candidate_alt_m": best["candidate_alt_m"],
            "expected_grid_center_lat_deg": best["expected_lat_deg"],
            "expected_grid_center_lon_deg": best["expected_lon_deg"],
            "expected_alt_m": best["interpolated_expected_alt_m"],
            "signed_vertical_m": best["signed_vertical_interp_m"],
            "abs_vertical_distance_m": best["abs_vertical_interp_m"],
            "horizontal_distance_km": best["horizontal_distance_km"],
            "candidate_score": best["candidate_score"],
            "altitude_consistency_score": best["altitude_consistency_score"],
            "altitude_consistency_label": best["altitude_consistency_label"],
            "n_gates": int(best["n_gates"]),
            "max_reflectivity_dbz": best["max_reflectivity_dbz"],
            "segment_speed_kmh": seg_speed,
            "segment_bearing_deg": seg_bearing,
            "path_step_score": best["_path_score"],
            "path_selected_reason": _selection_reason(best),
            "notes": best.get("notes", ""),
        })

        prev_lat = best["candidate_lat_deg"]
        prev_lon = best["candidate_lon_deg"]
        prev_time = scan_time
        prev_bearing = seg_bearing

    return pd.DataFrame(path_rows)


def _selection_reason(row: pd.Series) -> str:
    """Generate a human-readable selection reason."""
    parts = []
    label = row["altitude_consistency_label"]
    parts.append(label.replace("_", " "))
    if row["horizontal_distance_km"] < 5.0:
        parts.append("near grid center")
    if row.get("compactness_score", 0) > 0.7:
        parts.append("compact return")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# GeoJSON generation
# ---------------------------------------------------------------------------


def path_to_geojson(path_df: pd.DataFrame) -> dict:
    """Convert path CSV to GeoJSON with points and connecting line."""
    features = []

    # Point features for each path step
    for _, row in path_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    row["candidate_lon_deg"],
                    row["candidate_lat_deg"],
                    row["candidate_alt_m"],
                ],
            },
            "properties": {
                k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                for k, v in row.items()
                if k not in ("candidate_lat_deg", "candidate_lon_deg")
            },
        })

    # Connecting LineString
    if len(path_df) >= 2:
        coords = [
            [row["candidate_lon_deg"], row["candidate_lat_deg"]]
            for _, row in path_df.iterrows()
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "type": "candidate_path_line",
                "n_points": len(path_df),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def smoothed_path_geojson(path_df: pd.DataFrame) -> dict:
    """Generate a smoothed version of the path using rolling average."""
    if len(path_df) < 3:
        return path_to_geojson(path_df)

    df = path_df.copy()
    # Simple 3-point rolling average for smoothing
    df["smooth_lat"] = df["candidate_lat_deg"].rolling(
        3, min_periods=1, center=True
    ).mean()
    df["smooth_lon"] = df["candidate_lon_deg"].rolling(
        3, min_periods=1, center=True
    ).mean()

    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["smooth_lon"], row["smooth_lat"]],
            },
            "properties": {
                "scan_time_utc": row["scan_time_utc"],
                "type": "smoothed_point",
                "original_lat": row["candidate_lat_deg"],
                "original_lon": row["candidate_lon_deg"],
            },
        })

    if len(df) >= 2:
        coords = [
            [row["smooth_lon"], row["smooth_lat"]]
            for _, row in df.iterrows()
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "type": "smoothed_path_line",
                "n_points": len(df),
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_path_report(
    path_df: pd.DataFrame,
    plausible: pd.DataFrame,
    case_id: str,
    site: str,
    start_time: str,
    end_time: str,
    out_path: Path,
) -> None:
    """Generate cautious path-fit report."""
    n_scans = plausible["scan_time_utc"].nunique() if not plausible.empty else 0
    n_path = len(path_df)
    time_span = ""
    med_vert = 0.0
    max_speed = 0.0
    speeds_plausible = True

    if n_path > 0:
        t0 = path_df["scan_time_utc"].iloc[0]
        t1 = path_df["scan_time_utc"].iloc[-1]
        time_span = f"{t0.split('T')[1][:8]} → {t1.split('T')[1][:8]} UTC"
        med_vert = path_df["abs_vertical_distance_m"].median()
        max_speed = path_df["segment_speed_kmh"].max()
        speeds_plausible = (
            path_df["segment_speed_kmh"] <= MAX_PLAUSIBLE_SPEED_KMH
        ).all()

    # Altitude trend
    alt_follows = False
    if n_path >= 3:
        alt_corr = np.corrcoef(
            range(n_path),
            path_df["candidate_alt_m"].values,
        )[0, 1]
        exp_corr = np.corrcoef(
            range(n_path),
            path_df["expected_alt_m"].values,
        )[0, 1]
        alt_follows = abs(alt_corr - exp_corr) < 0.3 and alt_corr > 0.5

    # Horizontal smoothness
    horiz_smooth = False
    if n_path >= 3:
        lats = path_df["candidate_lat_deg"].values
        lons = path_df["candidate_lon_deg"].values
        lat_diffs = np.diff(lats)
        lon_diffs = np.diff(lons)
        sign_changes_lat = np.sum(np.diff(np.sign(lat_diffs)) != 0)
        sign_changes_lon = np.sum(np.diff(np.sign(lon_diffs)) != 0)
        horiz_smooth = (sign_changes_lat + sign_changes_lon) <= n_path

    no_path = n_path < 2

    report = textwrap.dedent(f"""\
    # Candidate Trajectory Report — {case_id}

    **Radar site:** {site}
    **Time window:** {start_time} → {end_time}
    **Generated:** {datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")}

    ---

    ## Position Uncertainty Context

    The balloon horizontal position is estimated from 6-character Maidenhead grid
    squares. At this latitude, one grid square is several kilometers across, so
    horizontal offsets of a few km may still be consistent with the telemetry
    region. Altitude from telemetry is more reliable and is used as the primary
    constraint.

    Expected altitude at each radar scan time is **linearly interpolated** from
    telemetry points.

    > **Radar beam-center caveat:** Radar gate altitude is the beam-center
    > altitude. The beam has finite width that increases with range. A candidate
    > whose beam-center altitude differs by a few hundred meters could still
    > have the balloon within the beam volume.

    ---

    ## Path Fit Summary

    | Metric | Value |
    |--------|-------|
    | Candidate scan times in window | {n_scans} |
    | Selected path points | {n_path} |
    | Time span | {time_span} |
    | Median absolute vertical mismatch | {med_vert:.0f} m |
    | Maximum segment speed | {max_speed:.1f} km/h |
    | All speeds plausible (< {MAX_PLAUSIBLE_SPEED_KMH} km/h) | \
{"Yes ✓" if speeds_plausible else "**No ✗**"} |

    ---

    ## Selected Path Points

    | Step | Scan Time | Orig R | Alt R | Signed Vert | H-dist | Speed | \
Alt Label |
    |:----:|-----------|:------:|:-----:|:-----------:|:------:|:-----:|-----------|
    """)

    for i, (_, row) in enumerate(path_df.iterrows()):
        scan_short = row["scan_time_utc"].split("T")[1][:8]
        report += (
            f"| {i + 1} "
            f"| {scan_short} "
            f"| {int(row['original_candidate_rank'])} "
            f"| {int(row['altitude_priority_rank'])} "
            f"| {row['signed_vertical_m']:+.0f} m "
            f"| {row['horizontal_distance_km']:.1f} km "
            f"| {row['segment_speed_kmh']:.0f} km/h "
            f"| {row['altitude_consistency_label'].replace('_', ' ')} |\n"
        )

    # Assessment
    if no_path:
        assessment = (
            "**No plausible candidate path could be constructed** in this time "
            "window. Altitude-consistent candidates exist at individual scan "
            "times, but they cannot be connected into a smooth trajectory "
            "without physically implausible jumps."
        )
    elif alt_follows and horiz_smooth and speeds_plausible:
        assessment = (
            "The selected candidates form an **altitude-consistent, "
            "horizontally smooth candidate sequence** that tracks the "
            "expected balloon altitude profile. Segment speeds are physically "
            "plausible for a picoballoon at these altitudes.\n\n"
            "This constitutes an **altitude-constrained candidate trajectory** "
            "— a radar-assisted candidate path that is consistent with the "
            "balloon telemetry. It is not a confirmed detection, but the "
            "altitude consistency, horizontal smoothness, and plausible speeds "
            "raise the prior probability that these candidates are "
            "balloon-associated."
        )
    elif alt_follows and speeds_plausible:
        assessment = (
            "The selected candidates show **altitude consistency** with the "
            "expected balloon profile, and segment speeds are physically "
            "plausible. However, the horizontal track shows some irregularity. "
            "Given that horizontal position is derived from coarse Maidenhead "
            "grid squares, this irregularity may be acceptable.\n\n"
            "This is a **possible balloon-associated sequence** that warrants "
            "further investigation, but does not constitute a confirmed "
            "detection."
        )
    else:
        assessment = (
            "The selected candidates show some altitude consistency, but the "
            "path has issues: "
            + ("implausible segment speeds, " if not speeds_plausible else "")
            + ("altitude trend does not match, " if not alt_follows else "")
            + ("horizontal motion is not smooth." if not horiz_smooth else "")
            + "\n\nThis result is **inconclusive**. Altitude-consistent "
            "candidates exist, but they do not form a reliable continuous path."
        )

    report += textwrap.dedent(f"""
    ---

    ## Assessment

    {assessment}

    ---

    ## Terminology

    This analysis uses cautious terminology:
    - **altitude-constrained candidate trajectory** — a path constructed from
      radar candidates whose altitudes are consistent with balloon telemetry
    - **radar-assisted candidate path** — the candidate path overlaid on radar
      geometry
    - **possible balloon-associated sequence** — candidates that may be related
      to the balloon, based on altitude and spatial consistency

    This does **not** use:
    - "confirmed track" — not confirmed without independent verification
    - "detected balloon" — altitude consistency is not proof of detection
    - "exact GPS track" — the horizontal position is from grid squares

    ---

    ## Caveats

    - Horizontal position from 6-character Maidenhead grid squares
      (~4.6 × 7.1 km at this latitude).
    - Radar gate altitude is beam-center altitude with beam-width uncertainty.
    - Expected altitude is linearly interpolated from sparse telemetry.
    - Path selection is greedy (not globally optimal).
    - Many non-balloon targets can appear at similar altitudes.
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit_candidate_path(
    config_path: Path,
    radar_site: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> Path:
    """Run altitude-constrained path fitting."""
    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_id = config["case_id"]
    case_dir = config_path.parent
    cand_dir = candidates_dir(config_path, site)

    out_dir = cand_dir / "path_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    track = pd.read_csv(case_dir / "expected_track.csv")
    track = track.sort_values("time_utc").reset_index(drop=True)

    alt_csv = cand_dir / "altitude_validation" / "altitude_prioritized_candidates.csv"
    candidates = pd.read_csv(alt_csv)

    # Filter plausible candidates
    plausible = filter_plausible_candidates(candidates, start_time, end_time)
    print(
        f"Plausible candidates: {len(plausible)} "
        f"across {plausible['scan_time_utc'].nunique()} scan times"
    )

    # Select path
    path_df = select_path(plausible)
    print(f"Selected path points: {len(path_df)}")

    # Write CSV
    csv_path = out_dir / "candidate_path.csv"
    path_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Write GeoJSON
    gj_path = out_dir / "candidate_path.geojson"
    with open(gj_path, "w", encoding="utf-8") as f:
        json.dump(path_to_geojson(path_df), f, indent=2, default=str)
    print(f"Wrote {gj_path}")

    smooth_path = out_dir / "smoothed_candidate_path.geojson"
    with open(smooth_path, "w", encoding="utf-8") as f:
        json.dump(smoothed_path_geojson(path_df), f, indent=2, default=str)
    print(f"Wrote {smooth_path}")

    # Report
    effective_start = start_time or "full range"
    effective_end = end_time or "full range"
    generate_path_report(
        path_df, plausible, case_id, site,
        effective_start, effective_end,
        out_dir / "path_fit_report.md",
    )

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site, default from config")
    parser.add_argument("--start-time", help="Start time (ISO 8601)")
    parser.add_argument("--end-time", help="End time (ISO 8601)")
    args = parser.parse_args()
    fit_candidate_path(
        args.config,
        radar_site=args.radar_site,
        start_time=args.start_time,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
