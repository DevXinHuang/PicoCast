#!/usr/bin/env python
"""Score near-track radar gate clusters for candidate inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    SEARCH_WINDOWS,
    bounded_score,
    candidate_label,
    candidates_dir,
    horizontal_distance_km,
    load_config,
    radar_site_from_config,
    reflectivity_score,
)

CANDIDATE_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "search_window",
    "cluster_id",
    "expected_lat_deg",
    "expected_lon_deg",
    "expected_alt_m",
    "candidate_lat_deg",
    "candidate_lon_deg",
    "candidate_alt_m",
    "horizontal_distance_km",
    "vertical_distance_m",
    "n_gates",
    "max_reflectivity_dbz",
    "mean_reflectivity_dbz",
    "p95_reflectivity_dbz",
    "velocity_mean_ms",
    "spectrum_width_mean_ms",
    "rhohv_mean",
    "distance_score",
    "altitude_score",
    "reflectivity_score",
    "compactness_score",
    "isolation_score",
    "temporal_continuity_score",
    "candidate_score",
    "candidate_rank",
    "candidate_label",
    "notes",
]


def score_candidates(clusters: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Score cluster rows using transparent, inspection-oriented components."""

    if clusters.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    summary_lookup = summaries.set_index(["scan_time_utc", "search_window"])
    rows = []
    for _, cluster in clusters.iterrows():
        summary = summary_lookup.loc[(cluster["scan_time_utc"], cluster["search_window"])]
        summary_notes = str(summary.get("notes", ""))
        notes = str(cluster["notes"])
        if summary_notes and summary_notes != "nan":
            notes = f"{notes}; {summary_notes}" if notes else summary_notes
        window = SEARCH_WINDOWS[str(cluster["search_window"])]
        horizontal_distance = float(
            horizontal_distance_km(
                float(cluster["cluster_center_lat_deg"]),
                float(cluster["cluster_center_lon_deg"]),
                float(summary["expected_lat_deg"]),
                float(summary["expected_lon_deg"]),
            )
        )
        vertical_distance = abs(
            float(cluster["cluster_center_alt_m"]) - float(summary["expected_alt_m"])
        )
        distance_score = bounded_score(horizontal_distance, 0.0, window["horizontal_radius_km"])
        altitude_score = bounded_score(
            vertical_distance,
            0.0,
            window["vertical_radius_km"] * 1000.0,
        )
        refl_score = reflectivity_score(
            float(cluster["cluster_max_reflectivity_dbz"]),
            float(cluster["cluster_p95_reflectivity_dbz"]),
        )
        compactness_score = bounded_score(
            float(cluster["compactness_km"]),
            0.0,
            max(1.0, window["horizontal_radius_km"] / 2.0),
        )
        isolation_score = bounded_score(float(cluster["n_gates"]), 1.0, 80.0)
        temporal_score = 0.0
        rows.append(
            {
                "case_id": cluster["case_id"],
                "radar_site": cluster["radar_site"],
                "scan_time_utc": cluster["scan_time_utc"],
                "scan_filename": cluster["scan_filename"],
                "search_window": cluster["search_window"],
                "cluster_id": cluster["cluster_id"],
                "expected_lat_deg": float(summary["expected_lat_deg"]),
                "expected_lon_deg": float(summary["expected_lon_deg"]),
                "expected_alt_m": float(summary["expected_alt_m"]),
                "candidate_lat_deg": float(cluster["cluster_center_lat_deg"]),
                "candidate_lon_deg": float(cluster["cluster_center_lon_deg"]),
                "candidate_alt_m": float(cluster["cluster_center_alt_m"]),
                "horizontal_distance_km": horizontal_distance,
                "vertical_distance_m": vertical_distance,
                "n_gates": int(cluster["n_gates"]),
                "max_reflectivity_dbz": float(cluster["cluster_max_reflectivity_dbz"]),
                "mean_reflectivity_dbz": float(cluster["cluster_mean_reflectivity_dbz"]),
                "p95_reflectivity_dbz": float(cluster["cluster_p95_reflectivity_dbz"]),
                "velocity_mean_ms": value_or_nan(cluster["cluster_velocity_mean_ms"]),
                "spectrum_width_mean_ms": value_or_nan(cluster["cluster_spectrum_width_mean_ms"]),
                "rhohv_mean": value_or_nan(cluster["cluster_rhohv_mean"]),
                "distance_score": distance_score,
                "altitude_score": altitude_score,
                "reflectivity_score": refl_score,
                "compactness_score": compactness_score,
                "isolation_score": isolation_score,
                "temporal_continuity_score": temporal_score,
                "candidate_score": 0.0,
                "candidate_rank": 0,
                "candidate_label": "no_candidate",
                "notes": notes,
            }
        )

    scored = pd.DataFrame(rows)
    scored["temporal_continuity_score"] = temporal_continuity_scores(scored)
    base_score = (
        0.24 * scored["distance_score"]
        + 0.20 * scored["altitude_score"]
        + 0.20 * scored["reflectivity_score"]
        + 0.14 * scored["compactness_score"]
        + 0.10 * scored["isolation_score"]
        + 0.12 * scored["temporal_continuity_score"]
    )
    broad_feature_penalty = 0.35 + 0.65 * scored["isolation_score"]
    telemetry_window_penalty = np.where(
        scored["notes"].str.contains("outside telemetry window", na=False),
        0.25,
        1.0,
    )
    scored["candidate_score"] = base_score * broad_feature_penalty * telemetry_window_penalty
    scored = scored.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    scored["candidate_rank"] = np.arange(1, len(scored) + 1)
    scored["candidate_label"] = scored["candidate_score"].map(candidate_label)
    return scored[CANDIDATE_COLUMNS]


def temporal_continuity_scores(scored: pd.DataFrame) -> pd.Series:
    """Reward candidates with nearby clusters in adjacent scan times."""

    if scored.empty:
        return pd.Series(dtype=float)
    scan_times = pd.to_datetime(scored["scan_time_utc"], utc=True)
    scores = []
    for idx, row in scored.iterrows():
        same_window = scored["search_window"] == row["search_window"]
        time_delta_min = (scan_times - scan_times.iloc[idx]).abs().dt.total_seconds() / 60.0
        nearby_time = (time_delta_min > 0.0) & (time_delta_min <= 16.0)
        nearby_space = (
            horizontal_distance_km(
                scored["candidate_lat_deg"].to_numpy(dtype=float),
                scored["candidate_lon_deg"].to_numpy(dtype=float),
                float(row["candidate_lat_deg"]),
                float(row["candidate_lon_deg"]),
            )
            <= 15.0
        )
        scores.append(float((same_window & nearby_time & nearby_space).any()))
    return pd.Series(scores, index=scored.index)


def value_or_nan(value) -> float:
    """Convert aggregate values to plain floats."""

    if pd.isna(value):
        return float("nan")
    return float(value)


def score_case_candidates(config_path: Path, radar_site: str | None = None) -> tuple[Path, Path]:
    """Score clusters and write candidate score tables."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    output_dir = candidates_dir(config_path, site)
    clusters = pd.read_csv(output_dir / "gate_clusters.csv")
    summaries = pd.read_csv(output_dir / "scan_gate_summary.csv")
    scored = score_candidates(clusters, summaries)
    candidate_path = output_dir / "candidate_scores.csv"
    top_path = output_dir / "top_candidates.csv"
    scored.to_csv(candidate_path, index=False)
    scored.head(10).to_csv(top_path, index=False)
    print(f"Wrote {candidate_path}")
    print(f"Wrote {top_path}")
    return candidate_path, top_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    args = parser.parse_args()
    score_case_candidates(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
