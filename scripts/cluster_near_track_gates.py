#!/usr/bin/env python
"""Cluster valid nearby radar gates into inspectable candidate features."""

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
    candidates_dir,
    load_config,
    local_xy_km,
    radar_site_from_config,
)

CLUSTER_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "scan_filename",
    "search_window",
    "cluster_id",
    "n_gates",
    "cluster_center_lat_deg",
    "cluster_center_lon_deg",
    "cluster_center_alt_m",
    "cluster_min_distance_to_expected_km",
    "cluster_min_vertical_distance_m",
    "cluster_max_reflectivity_dbz",
    "cluster_mean_reflectivity_dbz",
    "cluster_p95_reflectivity_dbz",
    "cluster_velocity_mean_ms",
    "cluster_spectrum_width_mean_ms",
    "cluster_rhohv_mean",
    "compactness_km",
    "notes",
]


def cluster_group(group: pd.DataFrame) -> pd.DataFrame:
    """Cluster one scan/search-window group of valid-reflectivity gates."""

    valid = group.dropna(subset=["reflectivity_dbz"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    window_name = str(valid["search_window"].iloc[0])
    window = SEARCH_WINDOWS[window_name]
    x_km, y_km = local_xy_km(
        valid["gate_lat_deg"],
        valid["gate_lon_deg"],
        float(valid["expected_lat_deg"].iloc[0]),
        float(valid["expected_lon_deg"].iloc[0]),
    )
    z_km = (
        valid["gate_alt_m"].to_numpy(dtype=float) - float(valid["expected_alt_m"].iloc[0])
    ) / 1000.0
    coords = np.column_stack([x_km, y_km, z_km])
    labels = dbscan_labels(coords, eps_km=max(0.75, window["horizontal_radius_km"] / 4.0))

    clustered = valid.copy()
    clustered["_cluster_label"] = labels
    cluster_labels = sorted(label for label in set(labels) if label >= 0)
    if not cluster_labels:
        clustered = (
            clustered.sort_values("reflectivity_dbz", ascending=False)
            .head(10)
            .assign(_cluster_label=lambda frame: np.arange(len(frame)))
        )
        notes = "single-gate fallback; no multi-gate DBSCAN cluster"
    else:
        clustered = clustered[clustered["_cluster_label"].isin(cluster_labels)]
        notes = "DBSCAN cluster"

    rows = []
    for sequence, (_, cluster) in enumerate(clustered.groupby("_cluster_label"), start=1):
        rows.append(summarize_cluster(cluster, sequence=sequence, notes=notes))
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)


def dbscan_labels(coords: np.ndarray, eps_km: float) -> np.ndarray:
    """Cluster coordinates using DBSCAN, with a simple fallback if sklearn is unavailable."""

    try:
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=eps_km, min_samples=2).fit_predict(coords)
    except Exception:  # noqa: BLE001 - fallback keeps end-of-day workflow moving.
        return simple_connected_labels(coords, eps_km=eps_km)


def simple_connected_labels(coords: np.ndarray, eps_km: float) -> np.ndarray:
    """Very small connected-distance grouping fallback."""

    labels = np.full(len(coords), -1, dtype=int)
    label = 0
    for idx in range(len(coords)):
        if labels[idx] >= 0:
            continue
        distances = np.linalg.norm(coords - coords[idx], axis=1)
        members = np.where(distances <= eps_km)[0]
        if len(members) < 2:
            continue
        labels[members] = label
        label += 1
    return labels


def summarize_cluster(cluster: pd.DataFrame, *, sequence: int, notes: str) -> dict:
    """Create one cluster summary row."""

    scan_time = str(cluster["scan_time_utc"].iloc[0])
    window_name = str(cluster["search_window"].iloc[0])
    stamp = scan_time.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    cluster_id = f"{stamp}_{window_name}_{sequence:03d}"
    center_lat = float(cluster["gate_lat_deg"].mean())
    center_lon = float(cluster["gate_lon_deg"].mean())
    center_alt = float(cluster["gate_alt_m"].mean())
    x_km, y_km = local_xy_km(
        cluster["gate_lat_deg"],
        cluster["gate_lon_deg"],
        center_lat,
        center_lon,
    )
    z_km = (cluster["gate_alt_m"].to_numpy(dtype=float) - center_alt) / 1000.0
    compactness = float(np.sqrt(x_km**2 + y_km**2 + z_km**2).max()) if len(cluster) else np.nan

    return {
        "case_id": cluster["case_id"].iloc[0],
        "radar_site": cluster["radar_site"].iloc[0],
        "scan_time_utc": scan_time,
        "scan_filename": cluster["scan_filename"].iloc[0],
        "search_window": window_name,
        "cluster_id": cluster_id,
        "n_gates": int(len(cluster)),
        "cluster_center_lat_deg": center_lat,
        "cluster_center_lon_deg": center_lon,
        "cluster_center_alt_m": center_alt,
        "cluster_min_distance_to_expected_km": float(cluster["horizontal_distance_km"].min()),
        "cluster_min_vertical_distance_m": float(cluster["vertical_distance_m"].min()),
        "cluster_max_reflectivity_dbz": float(cluster["reflectivity_dbz"].max()),
        "cluster_mean_reflectivity_dbz": float(cluster["reflectivity_dbz"].mean()),
        "cluster_p95_reflectivity_dbz": float(cluster["reflectivity_dbz"].quantile(0.95)),
        "cluster_velocity_mean_ms": value_or_nan(cluster["velocity_ms"].mean()),
        "cluster_spectrum_width_mean_ms": value_or_nan(cluster["spectrum_width_ms"].mean()),
        "cluster_rhohv_mean": value_or_nan(cluster["cross_correlation_ratio"].mean()),
        "compactness_km": compactness,
        "notes": notes,
    }


def value_or_nan(value) -> float:
    """Convert aggregate values to plain floats."""

    if pd.isna(value):
        return float("nan")
    return float(value)


def cluster_case_gates(config_path: Path, radar_site: str | None = None) -> Path:
    """Cluster extracted near-track gates for one case/radar site."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    output_dir = candidates_dir(config_path, site)
    gates_path = output_dir / "near_track_gates.csv"
    gates = pd.read_csv(gates_path)
    frames = []
    for _, group in gates.groupby(["scan_time_utc", "search_window"], dropna=False):
        clustered = cluster_group(group)
        if not clustered.empty:
            frames.append(clustered)
    clusters = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=CLUSTER_COLUMNS)
    )
    output_path = output_dir / "gate_clusters.csv"
    clusters.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    args = parser.parse_args()
    cluster_case_gates(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
