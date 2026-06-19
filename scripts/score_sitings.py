#!/usr/bin/env python3
"""
score_sitings.py — Multi-factor balloon siting identification scorer for PicoCAST.

Reads plausible tracklets, telemetry comparison, and cross-radar associations for a
given case and computes a weighted 0-1 identification score for each tracklet.

Usage:
    python scripts/score_sitings.py --case-id k7uaz_20260322

Outputs:
    cases/<case_id>/outputs/discovery/siting_scores.csv
    cases/<case_id>/outputs/discovery/siting_scores_geojson.json  (for map embedding)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "altitude_match": 0.25,      # lower median altitude mismatch → higher score
    "horizontal_proximity": 0.20, # lower mean corridor distance → higher score
    "smoothness": 0.20,          # lower spaghetti_score (smoother path) → higher score
    "speed_ratio": 0.15,         # speed_ratio close to 1.0 → higher score
    "cross_radar": 0.15,         # corroborated by ≥1 other radar → higher score
    "duration_coverage": 0.05,   # longer overlap duration → higher score
}

# Normalization reference values (tuned for picoballoon 10-12 km float scenarios)
ALT_MISMATCH_GOOD_M = 50.0      # ≤50 m → 1.0
ALT_MISMATCH_BAD_M = 500.0      # ≥500 m → 0.0
CORRIDOR_GOOD_KM = 3.0          # ≤3 km → 1.0
CORRIDOR_BAD_KM = 25.0          # ≥25 km → 0.0
SPAGHETTI_GOOD = 60.0           # ≤60 → 1.0
SPAGHETTI_BAD = 300.0           # ≥300 → 0.0
SPEED_RATIO_IDEAL = 1.0         # ideal ratio
SPEED_RATIO_TOLERANCE = 0.4     # within ±0.4 of 1.0 → degraded linearly
DURATION_MAX_MIN = 70.0         # reference full-window duration


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def score_altitude_match(median_mismatch_m: float) -> float:
    """Lower mismatch → higher score."""
    if median_mismatch_m <= ALT_MISMATCH_GOOD_M:
        return 1.0
    if median_mismatch_m >= ALT_MISMATCH_BAD_M:
        return 0.0
    return 1.0 - (median_mismatch_m - ALT_MISMATCH_GOOD_M) / (ALT_MISMATCH_BAD_M - ALT_MISMATCH_GOOD_M)


def score_horizontal_proximity(mean_corridor_km: float) -> float:
    """Closer to corridor → higher score."""
    if mean_corridor_km <= CORRIDOR_GOOD_KM:
        return 1.0
    if mean_corridor_km >= CORRIDOR_BAD_KM:
        return 0.0
    return 1.0 - (mean_corridor_km - CORRIDOR_GOOD_KM) / (CORRIDOR_BAD_KM - CORRIDOR_GOOD_KM)


def score_smoothness(spaghetti_score: float) -> float:
    """Lower spaghetti score (smoother path) → higher score."""
    if spaghetti_score <= SPAGHETTI_GOOD:
        return 1.0
    if spaghetti_score >= SPAGHETTI_BAD:
        return 0.0
    return 1.0 - (spaghetti_score - SPAGHETTI_GOOD) / (SPAGHETTI_BAD - SPAGHETTI_GOOD)


def score_speed_ratio(speed_ratio: float) -> float:
    """Speed ratio close to 1.0 → higher score."""
    deviation = abs(speed_ratio - SPEED_RATIO_IDEAL)
    if deviation >= SPEED_RATIO_TOLERANCE:
        return 0.0
    return 1.0 - (deviation / SPEED_RATIO_TOLERANCE)


def score_cross_radar(tracklet_id: str, associations_df: pd.DataFrame) -> float:
    """Returns 1.0 if this tracklet is in a strong cross-radar association, 0.5 if moderate, 0.0 if none."""
    if associations_df.empty:
        return 0.0
    matches = associations_df[associations_df["tracklet_ids"].str.contains(tracklet_id, na=False)]
    if matches.empty:
        return 0.0
    if (matches["association_label"] == "strong_cross_radar_candidate").any():
        return 1.0
    return 0.5


def score_duration_coverage(duration_min: float) -> float:
    """Longer duration relative to full window → higher score."""
    return clamp(duration_min / DURATION_MAX_MIN)


def compute_composite_score(row: pd.Series, telemetry_df: pd.DataFrame, assoc_df: pd.DataFrame) -> dict:
    tid = row["tracklet_id"]

    # Look up telemetry comparison row
    telem = telemetry_df[telemetry_df["tracklet_id"] == tid]
    mean_corridor_km = float(telem["mean_distance_to_corridor_km"].iloc[0]) if not telem.empty else 15.0
    speed_ratio = float(telem["speed_consistency_ratio"].iloc[0]) if not telem.empty else 1.0
    overlap_min = float(telem["overlap_duration_min"].iloc[0]) if not telem.empty else float(row.get("duration_min", 35.0))

    median_mismatch_m = float(row.get("median_abs_vertical_mismatch_m", 200.0))
    spaghetti_score = float(row.get("spaghetti_score", 150.0))
    duration_min = float(row.get("duration_min", 35.0))

    # Individual factor scores
    s_alt = score_altitude_match(median_mismatch_m)
    s_horiz = score_horizontal_proximity(mean_corridor_km)
    s_smooth = score_smoothness(spaghetti_score)
    s_speed = score_speed_ratio(speed_ratio)
    s_cross = score_cross_radar(tid, assoc_df)
    s_dur = score_duration_coverage(overlap_min)

    composite = (
        WEIGHTS["altitude_match"] * s_alt
        + WEIGHTS["horizontal_proximity"] * s_horiz
        + WEIGHTS["smoothness"] * s_smooth
        + WEIGHTS["speed_ratio"] * s_speed
        + WEIGHTS["cross_radar"] * s_cross
        + WEIGHTS["duration_coverage"] * s_dur
    )

    # Identification tier
    if composite >= 0.75:
        tier = "high_confidence_siting"
    elif composite >= 0.55:
        tier = "moderate_confidence_siting"
    elif composite >= 0.35:
        tier = "low_confidence_siting"
    else:
        tier = "speculative_siting"

    return {
        "tracklet_id": tid,
        "radar_site": row.get("radar_site", ""),
        "n_points": int(row.get("n_points", 0)),
        "start_time_utc": row.get("start_time_utc", ""),
        "end_time_utc": row.get("end_time_utc", ""),
        "duration_min": duration_min,
        "median_alt_mismatch_m": median_mismatch_m,
        "mean_corridor_km": mean_corridor_km,
        "spaghetti_score": spaghetti_score,
        "speed_ratio": speed_ratio,
        "cross_radar_score": s_cross,
        "factor_altitude": round(s_alt, 4),
        "factor_horizontal": round(s_horiz, 4),
        "factor_smoothness": round(s_smooth, 4),
        "factor_speed": round(s_speed, 4),
        "factor_cross_radar": round(s_cross, 4),
        "factor_duration": round(s_dur, 4),
        "identification_score": round(composite, 4),
        "identification_tier": tier,
    }


def build_tracklet_geojson(scores_df: pd.DataFrame, points_df: pd.DataFrame) -> dict:
    """Build a GeoJSON FeatureCollection of tracklet line-strings enriched with scores."""
    features = []
    for _, row in scores_df.iterrows():
        tid = row["tracklet_id"]
        pts = points_df[points_df["tracklet_id"] == tid].sort_values("scan_time_utc")
        if pts.empty:
            continue
        coords = [[float(r["cluster_lon_deg"]), float(r["cluster_lat_deg"]), float(r["cluster_alt_m"])]
                  for _, r in pts.iterrows()]
        times = pts["scan_time_utc"].tolist()
        alts = pts["cluster_alt_m"].tolist()
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "tracklet_id": tid,
                "radar_site": row["radar_site"],
                "identification_score": float(row["identification_score"]),
                "identification_tier": row["identification_tier"],
                "duration_min": float(row["duration_min"]),
                "median_alt_mismatch_m": float(row["median_alt_mismatch_m"]),
                "mean_corridor_km": float(row["mean_corridor_km"]),
                "spaghetti_score": float(row["spaghetti_score"]),
                "speed_ratio": float(row["speed_ratio"]),
                "start_time_utc": row["start_time_utc"],
                "end_time_utc": row["end_time_utc"],
                "scan_times": times,
                "point_alts_m": [float(a) for a in alts],
                "centroid_lat": float(pts["cluster_lat_deg"].mean()),
                "centroid_lon": float(pts["cluster_lon_deg"].mean()),
                "centroid_alt_m": float(pts["cluster_alt_m"].mean()),
            },
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def main(case_id: str, project_root: Path) -> None:
    discovery_dir = project_root / "cases" / case_id / "outputs" / "discovery"

    # Load inputs
    tracklets_path = discovery_dir / "plausible_tracklets.csv"
    telemetry_path = discovery_dir / "regional_tracklet_telemetry_comparison.csv"
    assoc_path = discovery_dir / "plausible_cross_radar_associations.csv"
    points_path = discovery_dir / "plausible_tracklet_points.csv"

    print(f"Loading tracklets from {tracklets_path}")
    tracklets_df = pd.read_csv(tracklets_path)
    print(f"  {len(tracklets_df)} tracklets")

    print(f"Loading telemetry comparison from {telemetry_path}")
    telemetry_df = pd.read_csv(telemetry_path)

    print(f"Loading cross-radar associations from {assoc_path}")
    assoc_df = pd.read_csv(assoc_path)

    print(f"Loading tracklet points from {points_path}")
    points_df = pd.read_csv(points_path)

    # Compute scores
    print("\nComputing multi-factor identification scores...")
    rows = []
    for _, row in tracklets_df.iterrows():
        rows.append(compute_composite_score(row, telemetry_df, assoc_df))

    scores_df = pd.DataFrame(rows).sort_values("identification_score", ascending=False).reset_index(drop=True)
    scores_df.insert(0, "siting_rank", range(1, len(scores_df) + 1))

    # Save CSV
    out_csv = discovery_dir / "siting_scores.csv"
    scores_df.to_csv(out_csv, index=False)
    print(f"\n✓ Wrote {out_csv}  ({len(scores_df)} rows)")

    # Print summary table
    print("\n── Top 10 Siting Candidates ─────────────────────────────────────────")
    print(f"{'Rank':<5} {'Tracklet':<14} {'Radar':<6} {'Score':<7} {'Tier'}")
    print("─" * 65)
    for _, r in scores_df.head(10).iterrows():
        print(f"{int(r.siting_rank):<5} {r.tracklet_id:<14} {r.radar_site:<6} {r.identification_score:<7.4f} {r.identification_tier}")

    # Build GeoJSON for map
    print("\nBuilding tracklet GeoJSON for map embedding...")
    geojson = build_tracklet_geojson(scores_df, points_df)
    out_geojson = discovery_dir / "siting_scores_geojson.json"
    with open(out_geojson, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))
    print(f"✓ Wrote {out_geojson}  ({len(geojson['features'])} features)")

    print("\n✓ Scoring complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PicoCAST multi-factor balloon siting scorer")
    parser.add_argument("--case-id", default="k7uaz_20260322", help="Case ID (default: k7uaz_20260322)")
    parser.add_argument("--project-root", default=".", help="Path to project root")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        print(f"ERROR: project root {root} does not exist", file=sys.stderr)
        sys.exit(1)

    main(args.case_id, root)
