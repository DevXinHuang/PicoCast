"""Shared helpers for near-track radar candidate triage scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEARCH_WINDOWS = {
    "tight": {"horizontal_radius_km": 5.0, "vertical_radius_km": 2.0},
    "normal": {"horizontal_radius_km": 10.0, "vertical_radius_km": 3.0},
    "loose": {"horizontal_radius_km": 20.0, "vertical_radius_km": 5.0},
}


def load_config(config_path: Path) -> dict:
    """Read a PicoCAST case YAML config."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def radar_site_from_config(config: dict, radar_site: str | None) -> str:
    """Resolve the requested radar site."""

    if radar_site:
        return radar_site
    if "nexrad" in config and "primary_radar_site" in config["nexrad"]:
        return str(config["nexrad"]["primary_radar_site"])
    return str(config["primary_radar_site"])


def case_id_from_config(config: dict) -> str:
    """Return the case identifier."""

    return str(config["case_id"])


def candidates_dir(config_path: Path, radar_site: str) -> Path:
    """Return the candidate-output directory for a radar site."""

    return config_path.parent / "outputs" / "candidates" / radar_site


def matches_path(config_path: Path, radar_site: str) -> Path:
    """Return the scan-track match CSV path for a radar site."""

    return config_path.parent / "nexrad" / radar_site / "index" / "scan_track_matches.csv"


def horizontal_distance_km(
    lat1_deg: np.ndarray | float,
    lon1_deg: np.ndarray | float,
    lat2_deg: np.ndarray | float,
    lon2_deg: np.ndarray | float,
) -> np.ndarray:
    """Compute haversine distance in kilometers."""

    earth_radius_km = 6371.0088
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def local_xy_km(
    lat_deg: pd.Series | np.ndarray,
    lon_deg: pd.Series | np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local x/y coordinates in kilometers around an origin."""

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(origin_lat_deg))
    x = (lon - origin_lon_deg) * km_per_deg_lon
    y = (lat - origin_lat_deg) * km_per_deg_lat
    return x, y


def bounded_score(value: float, best: float, worst: float) -> float:
    """Return a 0-1 score where best is 1 and worst-or-beyond is 0."""

    if np.isnan(value):
        return 0.0
    if worst == best:
        return 1.0
    return float(np.clip((worst - value) / (worst - best), 0.0, 1.0))


def reflectivity_score(
    max_reflectivity_dbz: float | None,
    p95_reflectivity_dbz: float | None,
) -> float:
    """Score valid reflectivity without treating high dBZ as proof."""

    values = [
        v
        for v in [max_reflectivity_dbz, p95_reflectivity_dbz]
        if v is not None and not np.isnan(v)
    ]
    if not values:
        return 0.0
    representative = max(values)
    return float(np.clip((representative + 10.0) / 35.0, 0.0, 1.0))


def candidate_label(score: float) -> str:
    """Return a cautious candidate label from a score."""

    if score < 0.25:
        return "no_candidate"
    if score < 0.45:
        return "weak_candidate"
    if score < 0.65:
        return "moderate_candidate"
    return "strong_candidate"
